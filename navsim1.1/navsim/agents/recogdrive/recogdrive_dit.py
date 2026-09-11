# ------------------------------------------------------------------------
# Modified from LightningDiT(https://github.com/hustvl/LightningDiT)
# Copyright (c) Xiaomi Corporation. All rights reserved.
# ------------------------------------------------------------------------
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np

from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)

from .blocks.rmsnorm import RMSNorm
from .blocks.attention import Attention
from .blocks.rope import RotaryEmbedding
from .blocks.encoder import SwiGLUFFN

class TimestepEncoder(nn.Module):
    """Encodes scalar timesteps into a high-dimensional vector."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = self.timestep_embedder.linear_1.weight.dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb



class FinalLayer(nn.Module):
    """
    The final output layer of the DiT model.

    Encapsulates the final adaptive layer normalization and the projection to
    the output dimension.
    """
    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_dim)
        self.modulation_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size )
        )

    @torch.compile
    def modulate(self, x, shift, scale):
        if shift is None:
            return x * (1 + scale.unsqueeze(1))
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation_proj(conditioning).chunk(2, dim=1)
        x = self.modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        norm_type: str = "layer_norm", 
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        if norm_type == "layer_norm":
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(dim)

        self.attn = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
        )

        if norm_type == "layer_norm":
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm2 = RMSNorm(dim)

        self.ffn = SwiGLUFFN(
            dim,
            bias=True,
        )

        self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim, bias=True)
            )

    @torch.compile
    def modulate(self, x, shift, scale):
        if shift is None:
            return x * (1 + scale.unsqueeze(1))
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        rotary_embedder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        mod_params = self.adaLN_modulation(conditioning)
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = \
            mod_params.chunk(6, dim=1)

        normed_states = self.norm1(hidden_states)
        modulated_states = self.modulate(normed_states, shift_attn, scale_attn)

        attn_output = self.attn(
            modulated_states,
            encoder_hidden_states=encoder_hidden_states,
            rotary_embedder=rotary_embedder
        )
        hidden_states = hidden_states + gate_attn.unsqueeze(1) * attn_output

        normed_states = self.norm2(hidden_states)
        modulated_states = self.modulate(normed_states, shift_ffn, scale_ffn)
        ffn_output = self.ffn(modulated_states)

        hidden_states = hidden_states + gate_ffn.unsqueeze(1) * ffn_output
        
        return hidden_states

class LightningDiT(nn.Module):

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 48,
        output_dim: int = 512,
        num_layers: int = 16,
        dropout: float = 0.0,
        attention_bias: bool = True,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        interleave_attention: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.output_dim = output_dim
        self.interleave_attention = interleave_attention

        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim
        )

        self.rotary_embedder = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=8, 
        )

        self.transformer_blocks = nn.ModuleList([
            LightningDiTBlock(
                dim=self.inner_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                attention_bias=attention_bias,
                norm_type=norm_type,
                norm_eps=norm_eps,
                cross_attention_dim=self.inner_dim if (idx % 2 != 0 or not interleave_attention) else None
            ) for idx in range(num_layers)
        ])
                                           
        self.final_layer = FinalLayer(self.inner_dim, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initializes weights for stable training.
        - Initializes positional embeddings with sine/cosine values.
        - Zeroes out the weights of modulation and final output layers.
        """
        def zero_out_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        for block in self.transformer_blocks:
            block.adaLN_modulation.apply(zero_out_init)
        self.final_layer.modulation_proj.apply(zero_out_init)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        conditioning_features: torch.Tensor,
        timesteps: torch.LongTensor,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the UnifiedDiT model.

        Args:
            hidden_states (torch.Tensor): Input sequence. Shape: (B, N, D).
            encoder_hidden_states (torch.Tensor): Context sequence for cross-attention.
            conditioning_features (torch.Tensor): Additional conditioning features.
            timesteps (torch.LongTensor): Diffusion timesteps.
            return_hidden_states (bool): If True, returns the output and a list of
                all intermediate hidden states.

        Returns:
            The final output tensor, or a tuple of the output and all hidden states.
        """
        seq_len = hidden_states.shape[1]
        #hidden_states = hidden_states + self.pos_embed[:, :seq_len, :]

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        conditioning_features = conditioning_features.contiguous()

        time_embedding = self.timestep_encoder(timesteps)
        conditioning = time_embedding + conditioning_features

        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            use_cross_attention = not (idx % 2 == 0 and self.interleave_attention)
            current_encoder_states = encoder_hidden_states if use_cross_attention else None
            
            hidden_states = block(
                hidden_states,
                conditioning=conditioning,
                encoder_hidden_states=current_encoder_states,
                rotary_embedder=self.rotary_embedder,
            )
            all_hidden_states.append(hidden_states)

        output = self.final_layer(hidden_states, conditioning)
        
        return (output, all_hidden_states) if return_hidden_states else output



