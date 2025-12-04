"""Training loop for hybrid diffusion language model with block diffusion."""

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config, TrainingConfig, DiffusionConfig
from ..model.block_diffusion import BlockDiffusion
from ..model.transformer import HybridDiffusionTransformer
from .utils import save_checkpoint, format_number


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a cosine learning rate schedule with linear warmup."""
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Trainer for hybrid diffusion language model with block diffusion."""

    def __init__(
        self,
        model: HybridDiffusionTransformer,
        block_diffusion: BlockDiffusion,
        train_loader: DataLoader,
        config: Config,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.block_diffusion = block_diffusion
        self.train_loader = train_loader
        self.config = config
        self.device = device

        # Config shortcuts
        self.train_config = config.training
        self.diff_config = config.diffusion
        self.model_config = config.model

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.train_config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.train_config.weight_decay,
            eps=1e-8,
        )

        # Setup scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.train_config.warmup_steps,
            num_training_steps=self.train_config.max_steps,
        )

        # Mixed precision
        self.use_amp = (
            self.train_config.mixed_precision != "no" and torch.cuda.is_available()
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        if self.train_config.mixed_precision == "bf16":
            self.autocast_dtype = torch.bfloat16
        else:
            self.autocast_dtype = torch.float16

        # Gradient checkpointing
        if self.train_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Logging
        self.wandb_run = None
        self.tb_writer = None
        self._setup_logging()

        # State
        self.global_step = 0
        self.total_tokens = 0

    def _setup_logging(self):
        """Setup WandB and TensorBoard logging."""
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.train_config.wandb_project,
                name=self.train_config.wandb_run_name,
                config={
                    "model": {
                        k: v for k, v in self.config.model.__dict__.items()
                        if not callable(v)
                    },
                    "training": self.config.training.__dict__,
                    "diffusion": self.config.diffusion.__dict__,
                    "data": self.config.data.__dict__,
                },
            )
        except ImportError:
            print("WandB not available, skipping")
        except Exception as e:
            print(f"WandB init failed: {e}")

        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = Path(self.train_config.output_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(log_dir))
        except ImportError:
            print("TensorBoard not available, skipping")
        except Exception as e:
            print(f"TensorBoard init failed: {e}")

    def _log_metrics(self, metrics: dict, step: int):
        """Log metrics to WandB and TensorBoard."""
        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)

        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)

    def train_step(self, batch: dict) -> dict:
        """Execute a single training step with block diffusion.

        Args:
            batch: Dictionary with 'input_ids' tensor

        Returns:
            Dictionary with loss and metrics
        """
        input_ids = batch["input_ids"].to(self.device)

        # Prepare block diffusion batch
        block_batch = self.block_diffusion.prepare_training_batch(
            input_ids,
            min_ratio=self.diff_config.min_masking_ratio,
            max_ratio=self.diff_config.max_masking_ratio,
        )

        masked_ids = block_batch["input_ids"]
        target_ids = block_batch["target_ids"]
        mask_indicator = block_batch["mask_indicator"]

        # Forward pass with mixed precision
        if self.use_amp:
            with autocast(device_type="cuda", dtype=self.autocast_dtype):
                result = self.model(masked_ids, target_ids=target_ids)
                logits = result["logits"]

                # Main diffusion loss
                loss = self.block_diffusion.compute_loss(
                    logits, target_ids, mask_indicator
                )

                # Add MTP auxiliary loss if available
                if "mtp_loss" in result:
                    loss = loss + result["mtp_loss"]
        else:
            result = self.model(masked_ids, target_ids=target_ids)
            logits = result["logits"]

            loss = self.block_diffusion.compute_loss(
                logits, target_ids, mask_indicator
            )

            if "mtp_loss" in result:
                loss = loss + result["mtp_loss"]

        # Compute accuracy on masked positions
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = (predictions == target_ids) & mask_indicator
            if mask_indicator.sum() > 0:
                accuracy = correct.sum().float() / mask_indicator.sum().float()
            else:
                accuracy = torch.tensor(0.0)

        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "masking_ratio": block_batch["masking_ratio"].mean().item(),
            "masking_ratio_min": block_batch["masking_ratio"].min().item(),
            "masking_ratio_max": block_batch["masking_ratio"].max().item(),
            "num_masked": mask_indicator.sum().item(),
            "mtp_loss": result.get("mtp_loss", torch.tensor(0.0)).item(),
        }

    def train(self):
        """Main training loop."""
        self.model.train()

        # Create checkpoint directory
        ckpt_dir = Path(self.train_config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Progress bar
        pbar = tqdm(
            total=self.train_config.max_steps,
            desc="Training",
            initial=self.global_step,
        )

        # Training state
        accumulated_loss = 0.0
        accumulated_accuracy = 0.0
        accumulation_count = 0

        # Tokens per second tracking
        last_log_time = time.time()
        last_log_tokens = self.total_tokens

        self.optimizer.zero_grad()
        data_iter = iter(self.train_loader)

        while self.global_step < self.train_config.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Forward and backward
            metrics = self.train_step(batch)
            loss = metrics["loss"] / self.train_config.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += metrics["loss"].item()
            accumulated_accuracy += metrics["accuracy"]
            accumulation_count += 1

            self.total_tokens += batch["input_ids"].numel()

            # Gradient accumulation step
            if accumulation_count >= self.train_config.gradient_accumulation_steps:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.train_config.grad_clip
                )

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                avg_loss = accumulated_loss / accumulation_count
                avg_accuracy = accumulated_accuracy / accumulation_count

                # Logging
                if self.global_step % self.train_config.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]

                    # Calculate tokens per second
                    current_time = time.time()
                    elapsed_time = current_time - last_log_time
                    tokens_since_last_log = self.total_tokens - last_log_tokens
                    tokens_per_second = tokens_since_last_log / elapsed_time if elapsed_time > 0 else 0
                    last_log_time = current_time
                    last_log_tokens = self.total_tokens

                    log_metrics = {
                        "train/loss": avg_loss,
                        "train/accuracy": avg_accuracy,
                        "train/learning_rate": lr,
                        "train/tokens": self.total_tokens,
                        "train/tokens_per_second": tokens_per_second,
                        "train/masking_ratio": metrics["masking_ratio"],
                        "train/masking_ratio_min": metrics.get("masking_ratio_min", 0.0),
                        "train/masking_ratio_max": metrics.get("masking_ratio_max", 0.0),
                        "train/mtp_loss": metrics["mtp_loss"],
                    }
                    self._log_metrics(log_metrics, self.global_step)

                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        acc=f"{avg_accuracy:.4f}",
                        tps=f"{tokens_per_second:.0f}",
                        lr=f"{lr:.2e}",
                    )

                # Save checkpoint
                if (
                    self.global_step > 0
                    and self.global_step % self.train_config.save_every == 0
                ):
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.global_step,
                        avg_loss,
                        self.config,
                        ckpt_dir / f"checkpoint_{self.global_step}.pt",
                    )

                # Reset accumulators
                accumulated_loss = 0.0
                accumulated_accuracy = 0.0
                accumulation_count = 0

                self.global_step += 1
                pbar.update(1)

        pbar.close()

        # Save final checkpoint
        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.global_step,
            avg_loss,
            self.config,
            ckpt_dir / "checkpoint_final.pt",
        )

        if self.wandb_run:
            import wandb
            wandb.finish()

        if self.tb_writer:
            self.tb_writer.close()

        print(f"Training complete! Total tokens: {format_number(self.total_tokens)}")
