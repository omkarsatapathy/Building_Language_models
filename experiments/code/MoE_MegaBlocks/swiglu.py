import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu_hidden_dim(in_features: int, multiple_of: int = 32) -> int:

    hidden = int(2 / 3 * 4 * in_features)
    hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
    return hidden


class SwiGLUFFN(nn.Module):
    """SwiGLU feedforward network (FFN) block."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or swiglu_hidden_dim(in_features)

        # gate and value projections (no bias, LLaMA-style)
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_value = nn.Linear(in_features, hidden_features, bias=False)
        # output projection 
        self.w_out = nn.Linear(hidden_features, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        value = self.w_value(x)
        hidden = F.silu(gate) * value            # SwiGLU: silu(gate) * value
        return self.dropout(self.w_out(hidden))

