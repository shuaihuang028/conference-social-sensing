"""ResNet visual encoder fused with normalized group geometry."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional
from torchvision.models import ResNet18_Weights, resnet18


class VisualGeometryGroupClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_geometry_features: int,
        num_labels: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.30,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        visual_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(visual_dim + num_geometry_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, images: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        visual_features = self.backbone(images)
        visual_features = functional.normalize(visual_features, p=2, dim=1)
        fused = torch.cat([visual_features, geometry], dim=1)
        return self.classifier(fused)
