import torch
import torch.nn as nn

from .easr_blocks import AdaptiveFeatureAggregation, ProgressiveFeatureRefinement


class SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EASRModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        dropout: float = 0.3,
        pfr_iterations: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.feature_aggregation = AdaptiveFeatureAggregation(
            channels_list=[64, 128, 256, 512]
        )
        self.progressive_refinement = ProgressiveFeatureRefinement(
            channels=512, iterations=pfr_iterations, norm_type="layernorm"
        )
        self.selayer = SELayer(512, reduction=8)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        features = [feat1, feat2, feat3, feat4]
        aggregated = self.feature_aggregation(features)
        refined = self.progressive_refinement(aggregated)
        refined = self.selayer(refined)
        refined = self.avgpool(refined)
        refined = refined.view(refined.size(0), -1)
        refined = self.dropout(refined)
        return self.fc(refined)


