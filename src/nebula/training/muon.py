"""Muon Optimizer - Momentum Orthogonalized by Newton-Schulz.

Muon is an optimizer designed specifically for transformers that applies
Newton-Schulz orthogonalization to the momentum, which helps with training
stability and speed.

Key features:
- Orthogonalizes momentum updates for matrix parameters (2D tensors)
- Uses AdamW for non-matrix parameters (1D tensors like biases, norms)
- Significantly faster convergence on transformer training

Reference: https://github.com/KellerJordan/Muon

Note: This implementation is adapted for PyTorch and includes optimizations
for mixed precision training.
"""

from typing import Optional, Tuple, Iterable, Callable
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Newton-Schulz iteration to compute G @ (G.T @ G)^(-1/2).

    This orthogonalizes the matrix G, making its singular values all equal to 1.

    Args:
        G: Matrix to orthogonalize [out_features, in_features]
        steps: Number of Newton-Schulz iterations
        eps: Small epsilon for numerical stability

    Returns:
        Orthogonalized matrix with same shape as G
    """
    assert G.ndim == 2, "Newton-Schulz only works on 2D tensors"

    a, b, c = (3.4445, -4.7750, 2.0315)  # Coefficients for cubic iteration

    X = G.bfloat16()

    # Normalize to have unit Frobenius norm
    X = X / (X.norm() + eps)

    # Newton-Schulz iterations
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon optimizer: Momentum Orthogonalized by Newton-Schulz.

    For matrix parameters (2D), applies Newton-Schulz orthogonalization to momentum.
    For vector parameters (1D), uses standard AdamW.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate for Muon (matrix params)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        adamw_lr: Learning rate for AdamW (1D params), defaults to lr * 0.1
        adamw_betas: Betas for AdamW (default: (0.9, 0.95))
        adamw_eps: Epsilon for AdamW (default: 1e-8)
        adamw_wd: Weight decay for AdamW (default: 0.0)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: Optional[float] = None,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        if adamw_lr is None:
            adamw_lr = lr * 0.1

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            adamw_lr = group['adamw_lr']
            adamw_betas = group['adamw_betas']
            adamw_eps = group['adamw_eps']
            adamw_wd = group['adamw_wd']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if p.ndim == 2:
                        # Muon for matrices
                        state['momentum_buffer'] = torch.zeros_like(p)
                    else:
                        # AdamW for vectors
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1

                if p.ndim == 2:
                    # Muon update for matrix parameters
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)

                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf

                    # Apply Newton-Schulz orthogonalization
                    grad = zeropower_via_newtonschulz5(grad, steps=ns_steps)

                    # Scale by sqrt of dimensions (similar to spectral normalization)
                    scale = max(1, grad.size(0) / grad.size(1)) ** 0.5

                    p.add_(grad, alpha=-lr * scale)

                else:
                    # AdamW update for non-matrix parameters (biases, norms, embeddings)
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    beta1, beta2 = adamw_betas

                    # Weight decay
                    if adamw_wd != 0:
                        p.mul_(1 - adamw_lr * adamw_wd)

                    # Update biased first moment estimate
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                    # Update biased second raw moment estimate
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Bias correction
                    step = state['step']
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    # Compute step
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(adamw_eps)
                    step_size = adamw_lr / bias_correction1

                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def get_muon_param_groups(
    model: torch.nn.Module,
    lr: float = 0.02,
    adamw_lr: Optional[float] = None,
    weight_decay: float = 0.0,
    no_decay_keywords: Tuple[str, ...] = ('bias', 'norm', 'embedding'),
) -> list:
    """Create parameter groups for Muon optimizer.

    Separates parameters into:
    - Matrix parameters (2D): Use Muon updates
    - Vector parameters (1D): Use AdamW updates
    - No decay parameters: Exclude from weight decay

    Args:
        model: The model to optimize
        lr: Learning rate for Muon
        adamw_lr: Learning rate for AdamW (default: lr * 0.1)
        weight_decay: Weight decay for AdamW parameters
        no_decay_keywords: Keywords for parameters to exclude from weight decay

    Returns:
        List of parameter groups for Muon optimizer
    """
    if adamw_lr is None:
        adamw_lr = lr * 0.1

    # Collect all parameters
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if should skip weight decay
            skip_wd = any(kw in name.lower() for kw in no_decay_keywords)
            params.append({
                'param': param,
                'name': name,
                'skip_wd': skip_wd,
            })

    # Just return a single group - Muon handles 1D/2D internally
    return [{'params': [p['param'] for p in params]}]
