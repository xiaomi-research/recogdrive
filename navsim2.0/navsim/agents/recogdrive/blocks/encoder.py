import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)



class SinusoidalPositionalEncoding(nn.Module):
    """
    Generates a sinusoidal encoding of shape (B, T, D) given timesteps of
    shape (B, T).
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps (torch.Tensor): A tensor of shape (B, T).

        Returns:
            torch.Tensor: The positional encoding. Shape: (B, T, D).
        """
        timesteps = timesteps.float()
        device = timesteps.device
        half_dim = self.embedding_dim // 2

        exponent = -torch.log(torch.tensor(10000.0)) / half_dim
        exponent = torch.arange(half_dim, dtype=torch.float32, device=device) * exponent

        freqs = timesteps.unsqueeze(-1) * torch.exp(exponent).unsqueeze(0)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)

        encoding = torch.cat([sin, cos], dim=-1)
        return encoding


class ActionEncoder(nn.Module):
    """
    Encodes a sequence of actions and their corresponding timesteps into a
    fixed-size embedding sequence.
    """
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

        self.fc1 = nn.Linear(action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions (torch.Tensor): Action sequence. Shape: (B, T, action_dim).
            timesteps (torch.Tensor): Timestep for each action. Shape: (B, T).

        Returns:
            torch.Tensor: The encoded action sequence. Shape: (B, T, hidden_size).
        """
        B, T, _ = actions.shape

        action_embedding = self.fc1(actions)

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        time_embedding = self.pos_encoding(timesteps).to(dtype=action_embedding.dtype)

        x = torch.cat([action_embedding, time_embedding], dim=-1)
        x = self.act(self.fc2(x))
        x = self.fc3(x)

        return x


class StateAttentionEncoder(nn.Module):
    """
    Encodes a flat state vector into a fixed-size embedding using a
    multi-head attention pooling mechanism.
    """
    def __init__(
        self,
        state_dim: int,
        embed_dim: int,
        num_kinematic_states: int,
        state_dropout: float = 0.75,
        num_heads: int = 4
    ):
        super().__init__()
        assert state_dim > num_kinematic_states, \
            "state_dim must be greater than num_kinematic_states"

        self.state_dim = state_dim
        self.num_kinematic_states = num_kinematic_states
        self.state_dropout = state_dropout

        self.linears = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(state_dim)
        ])
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.pos_embed = nn.Parameter(torch.Tensor(1, state_dim, embed_dim))
        self.query = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes learnable embeddings."""
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The flat state vector. Shape: (B, state_dim).

        Returns:
            torch.Tensor: The encoded state embedding. Shape: (B, embed_dim).
        """
        x_embed_list = [
            linear(x[:, i, None]) for i, linear in enumerate(self.linears)
        ]
        x_embed = torch.stack(x_embed_list, dim=1)
        x_embed = x_embed + self.pos_embed

        key_padding_mask = None
        if self.training and self.state_dropout > 0:
            kinematic_mask = torch.rand(
                (x_embed.shape[0], self.num_kinematic_states), device=x.device
            ) < self.state_dropout

            num_command_states = self.state_dim - self.num_kinematic_states
            command_mask = torch.zeros(
                (x_embed.shape[0], num_command_states),
                device=x.device,
                dtype=torch.bool
            )
            key_padding_mask = torch.cat([kinematic_mask, command_mask], dim=1)

        query = self.query.expand(x_embed.shape[0], -1, -1)
        x_state, _ = self.attn(
            query=query,
            key=x_embed,
            value=x_embed,
            key_padding_mask=key_padding_mask,
        )

        return x_state.squeeze(1)