"""GPT-2 tokenizer extended with [MASK] token for diffusion."""

from typing import List, Optional, Union

from transformers import GPT2TokenizerFast


class DiffusionTokenizer:
    """GPT-2 tokenizer with added [MASK] token for masked diffusion."""

    MASK_TOKEN = "[MASK]"
    MASK_TOKEN_ID = 50257  # GPT-2 vocab size is 50257, so [MASK] gets id 50257

    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # Add [MASK] token
        self.tokenizer.add_special_tokens({"mask_token": self.MASK_TOKEN})

        # Set padding token to EOS (GPT-2 doesn't have a pad token by default)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size including [MASK] token."""
        return len(self.tokenizer)

    @property
    def mask_token_id(self) -> int:
        """Return the [MASK] token id."""
        return self.tokenizer.mask_token_id

    @property
    def pad_token_id(self) -> int:
        """Return the padding token id."""
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        """Return the EOS token id."""
        return self.tokenizer.eos_token_id

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        """Encode text to token ids."""
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
        )

    def decode(
        self,
        token_ids: Union[List[int], "torch.Tensor"],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token ids to text."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        texts: List[str],
        max_length: int,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> dict:
        """Batch encode texts."""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
