from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CNN3DEncoder(nn.Module):
    """
    3D CNN encoder for static pore structure and remaining-oil morphology.

    The encoder downsamples the input volume with three conv-pool blocks and
    squeezes features to a 512-d vector via adaptive pooling, so it can handle
    different spatial sizes (e.g. 256x256x64 for production or smaller shapes
    in tests).
    """

    output_dim: int = 512

    def __init__(self, in_channels: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, self.output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        feats = self.features(x)
        return self.projection(feats)


class LSTMEncoder(nn.Module):
    """
    Bidirectional two-layer LSTM encoder for dynamic displacement sequences.

    Input shape: (batch, seq_len, feature_dim)
    Output: 512-d vector summarizing temporal evolution.
    """

    output_dim: int = 512

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, self.output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        # We only need the last timestep hidden state from both directions.
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # Take the last layer from both directions and concatenate.
        forward_final = h_n[-2]
        backward_final = h_n[-1]
        final_state = torch.cat([forward_final, backward_final], dim=1)
        return self.projection(final_state)


class ChannelAttention(nn.Module):
    """
    Lightweight channel attention to re-weight fused static + dynamic features.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.attn = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Softmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.attn(x)
        return x * weights


class FusionModel(nn.Module):
    """
    CNN + LSTM feature fusion with Darcy-law physical constraint.

    The forward pass returns predictions for saturation, seepage velocity, and
    dynamic/static transition probability. Velocity is softly corrected with a
    Darcy-based estimate using permeability, viscosity, and pressure gradient.
    """

    def __init__(
        self,
        static_channels: int = 1,
        dynamic_features: int = 5,
        physics_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.physics_weight = physics_weight
        self.static_encoder = CNN3DEncoder(in_channels=static_channels)
        self.dynamic_encoder = LSTMEncoder(input_size=dynamic_features)
        fused_dim = self.static_encoder.output_dim + self.dynamic_encoder.output_dim
        self.attention = ChannelAttention(fused_dim)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.saturation_head = nn.Linear(512, 1)
        self.velocity_head = nn.Linear(512, 1)
        self.transition_head = nn.Linear(512, 1)

    @staticmethod
    def _darcy_velocity(
        permeability: Tensor, viscosity: Tensor, pressure_gradient: Tensor
    ) -> Tensor:
        """
        Compute seepage velocity based on Darcy's law v = μ * K * ∇P.
        Shapes are broadcast-safe; inputs are expected to be (batch, 1) or
        broadcastable to the prediction shape.
        """
        return viscosity * permeability * pressure_gradient

    def forward(
        self,
        static_volume: Tensor,
        dynamic_sequence: Tensor,
        permeability: Tensor,
        viscosity: Tensor,
        pressure_gradient: Tensor,
    ) -> Dict[str, Tensor]:
        static_feat = self.static_encoder(static_volume)
        dynamic_feat = self.dynamic_encoder(dynamic_sequence)
        fused = torch.cat([static_feat, dynamic_feat], dim=1)
        fused = self.attention(fused)
        fused = self.head(fused)

        saturation = torch.sigmoid(self.saturation_head(fused))
        velocity_pred = F.relu(self.velocity_head(fused))
        transition_prob = torch.sigmoid(self.transition_head(fused))

        darcy_velocity = self._darcy_velocity(permeability, viscosity, pressure_gradient)
        corrected_velocity = torch.clamp(
            (1 - self.physics_weight) * velocity_pred
            + self.physics_weight * darcy_velocity,
            min=0.0,
        )

        return {
            "saturation": saturation,
            "velocity": corrected_velocity,
            "transition_prob": transition_prob,
            "darcy_velocity": darcy_velocity,
        }


class PhysicalConstraintLoss(nn.Module):
    """
    Composite loss combining data fit (MSE) and Darcy consistency.
    """

    def __init__(self, lambda_darcy: float = 0.05) -> None:
        super().__init__()
        self.lambda_darcy = lambda_darcy
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        permeability: Tensor,
        viscosity: Tensor,
        pressure_gradient: Tensor,
    ) -> Tensor:
        loss = 0.0
        if "saturation" in targets:
            loss = loss + self.mse(predictions["saturation"], targets["saturation"])
        if "velocity" in targets:
            loss = loss + self.mse(predictions["velocity"], targets["velocity"])
        if "transition_prob" in targets:
            loss = loss + self.mse(
                predictions["transition_prob"], targets["transition_prob"]
            )

        darcy_velocity = FusionModel._darcy_velocity(
            permeability, viscosity, pressure_gradient
        )
        loss_darcy = self.mse(predictions["velocity"], darcy_velocity)
        return loss + self.lambda_darcy * loss_darcy


def predict_with_model(
    model: FusionModel,
    static_volume: Tensor,
    dynamic_sequence: Tensor,
    permeability: Tensor,
    viscosity: Tensor,
    pressure_gradient: Tensor,
    targets: Optional[Dict[str, Tensor]] = None,
) -> Dict[str, Tensor]:
    """
    Convenience wrapper to run the model and optionally compute loss.
    """
    outputs = model(
        static_volume=static_volume,
        dynamic_sequence=dynamic_sequence,
        permeability=permeability,
        viscosity=viscosity,
        pressure_gradient=pressure_gradient,
    )
    if targets is not None:
        criterion = PhysicalConstraintLoss()
        outputs["loss"] = criterion(
            outputs, targets, permeability, viscosity, pressure_gradient
        )
    return outputs
