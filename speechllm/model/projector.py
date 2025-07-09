import torch
import torch.nn as nn
from typing import Optional

class MLPProjector(nn.Module):
    """
    Simple MLP projector for mapping between different hidden dimensions.

    Args:
        input_dim (int): Input dimension size
        output_dim (int): Output dimension size
        hidden_size (int, optional): Hidden layer dimension. Defaults to max(input_dim, output_dim).
        hidden_layers (int, optional): Number of hidden layers. Defaults to 1.
        activation (str, optional): Activation function name. Defaults to "relu".
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        residual (bool, optional): Whether to add a linear residual if input_dim != output_dim. Defaults to False.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: Optional[int] = None,
        hidden_layers: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size or max(input_dim, output_dim)
        self.residual = residual and (input_dim == output_dim or True)
        act = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "silu": nn.SiLU,
            "swish": nn.SiLU,
        }[activation.lower()]

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, self.hidden_size))
        layers.append(act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        # Output layer
        layers.append(nn.Linear(self.hidden_size, output_dim))
        self.mlp = nn.Sequential(*layers)

        if self.residual and input_dim != output_dim:
            self.res_proj = nn.Linear(input_dim, output_dim)
        else:
            self.res_proj = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        if self.residual:
            if self.res_proj is not None:
                res = self.res_proj(x)
            else:
                res = x
            out = out + res
        return out

