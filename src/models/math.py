import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,  # 隐藏层层数
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (
            num_layers - 1
        )  # [hidden_dim] * (num_layers - 1) = [hidden_dim, hidden_dim, hidden_dim, ...]
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP: ReLU on all layers except the last.

        Args:
            x: Input tensor of shape ``(..., input_dim)``.

        Returns:
            Output tensor of shape ``(..., output_dim)``.
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
