import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import DeformConv2d  # type: ignore
except (ImportError, OSError):  # torchvision compiled ops missing
    DeformConv2d = None


class DynamicWeightGenerator(nn.Module):
    def __init__(self, channels_list, hidden_dim: int = 256) -> None:
        super().__init__()
        parts = []
        for channels in channels_list:
            parts.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channels, hidden_dim // 4, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
            )
        self.global_descriptors = nn.ModuleList(parts)
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, len(channels_list)),
            nn.Softmax(dim=-1),
        )

    def forward(self, features_list):
        descriptors = []
        for extractor, feat in zip(self.global_descriptors, features_list):
            desc = extractor(feat).flatten(1)
            descriptors.append(desc)
        combined = torch.cat(descriptors, dim=1)
        return self.weight_generator(combined)


class DeformableConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        if DeformConv2d is None:
            self.offset_conv = None
            self.deform_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        else:
            self.offset_conv = nn.Conv2d(
                in_channels,
                2 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                padding=padding,
            )
            self.deform_conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if self.offset_conv is not None:
            nn.init.constant_(self.offset_conv.weight, 0)
            nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.offset_conv is not None:
            offset = self.offset_conv(x)
            x = self.deform_conv(x, offset)
        else:
            x = self.deform_conv(x)
        x = self.bn(x)
        return self.relu(x)


class AdaptiveFeatureAggregation(nn.Module):
    def __init__(self, channels_list, out_channels: int = 512) -> None:
        super().__init__()
        self.adaptors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, 128, kernel_size=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )
                for channels in channels_list
            ]
        )
        self.weight_generator = DynamicWeightGenerator(channels_list)
        self.fusion_conv = DeformableConvBlock(128, out_channels)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features_list):
        target_size = features_list[-1].shape[2:]
        adapted = []
        for adaptor, feat in zip(self.adaptors, features_list):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=True
                )
            adapted.append(adaptor(feat))

        weights = self.weight_generator(features_list)
        fused = torch.zeros_like(adapted[0])
        for index, feat in enumerate(adapted):
            weight = weights[:, index : index + 1, None, None]
            fused = fused + feat * weight
        fused = self.fusion_conv(fused)
        if not self.training and fused.size(0) % 2 == 0 and fused.size(1) > 1:
            fused = fused[:, 1:, :, :]
        return self.refine_conv(fused)


class ProgressiveFeatureRefinement(nn.Module):
    def __init__(self, channels: int, iterations: int = 4, norm_type: str = "layernorm") -> None:
        super().__init__()
        self.iterations = iterations
        self.norm_type = norm_type
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        self.final_weight_refined = nn.Parameter(torch.tensor(0.7))
        self.final_weight_original = nn.Parameter(torch.tensor(0.3))
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, channels // 8), num_channels=channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_adjust = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            self._make_norm(channels),
            nn.ReLU(inplace=True),
        )

    def _make_norm(self, channels: int) -> nn.Module:
        if self.norm_type == "groupnorm":
            groups = min(8, channels)
            return nn.GroupNorm(num_groups=groups, num_channels=channels)
        if self.norm_type == "layernorm":
            return nn.GroupNorm(num_groups=1, num_channels=channels)
        return nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x
        for _ in range(self.iterations):
            spatial = self.spatial_attention(x)
            channel = self.channel_attention(x)
            combined = spatial * x + channel * x
            residual = self.residual_adjust(combined)
            weight = torch.sigmoid(self.residual_weight)
            x = x + weight * residual
        weight_refined = torch.sigmoid(self.final_weight_refined)
        weight_original = torch.sigmoid(self.final_weight_original)
        total = weight_refined + weight_original + 1e-8
        return (weight_refined / total) * x + (weight_original / total) * original


