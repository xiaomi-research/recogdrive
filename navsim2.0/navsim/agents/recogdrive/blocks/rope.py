import torch
from torch import nn
from typing import Tuple

class RotaryEmbedding(nn.Module):
    """
    The Rotary Position Embedding (RoPE) module.

    This implementation uses a pre-computed cache of sine and cosine values to
    efficiently apply rotary embeddings to query and key tensors. It can
    dynamically expand the cache if a sequence longer than the initial
    `max_position_embeddings` is encountered.

    Attributes:
        dim (int): The dimension of the head the RoPE is applied to.
        max_position_embeddings (int): The maximum sequence length for the pre-computed cache.
        theta (float): The base for the geometric progression of frequencies.
        inv_freq (torch.Tensor): A buffer holding the inverse frequencies.
        cos_cached (torch.Tensor): A buffer holding the cached cosine values.
        sin_cached (torch.Tensor): A buffer holding the cached sine values.
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        theta: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device):
        """
        Updates the sine and cosine cache.

        Args:
            seq_len (int): The new maximum sequence length.
            device (torch.device): The device to store the cache on.
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates rotary embeddings for the given positions.

        Args:
            x (torch.Tensor): A dummy tensor used only to get the device and dtype.
            position_ids (torch.LongTensor): The positions of the tokens in the
                sequence. Shape: (batch_size, sequence_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the cosine and
                sine embeddings. Shape of each: (batch_size, 1, sequence_length, dim).
        """
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)

        cos = self.cos_cached.gather(
            2, position_ids.unsqueeze(1).unsqueeze(3).expand(-1, -1, -1, self.dim)
        )
        sin = self.sin_cached.gather(
            2, position_ids.unsqueeze(1).unsqueeze(3).expand(-1, -1, -1, self.dim)
        )
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor.

    Splits the last dimension of the tensor into two halves, negates the
    second half, and then concatenates them back together.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The tensor with half of its dimensions rotated.
    """

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


