"""
Task-specific head modules for PointNet++.

Provides classification, detection, and segmentation heads that
operate on features extracted by the PointNet++ backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Classification head for point cloud classification.

    Architecture: FC(512) -> BN -> ReLU -> Dropout(0.4) ->
                  FC(256) -> BN -> ReLU -> Dropout(0.4) ->
                  FC(num_classes)

    Args:
        in_channels: Number of input feature channels
        num_classes: Number of output classes
        dropout: Dropout probability (default 0.4)
    """

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Global feature vector, shape (B, in_channels)

        Returns:
            Class logits, shape (B, num_classes)
        """
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


class DetectionHead(nn.Module):
    """
    3D object detection head.

    Predicts (x, y, z, w, h, l, yaw, class_scores) for each proposal point.
    Uses shared FC layers followed by separate branches for:
    - Center regression (x, y, z)
    - Size regression (w, h, l)
    - Angle prediction (bin classification + residual regression)
    - Object class scores

    Args:
        in_channels: Number of input feature channels per point
        num_classes: Number of object classes
        num_angle_bins: Number of angle bins for bin-based angle prediction
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_angle_bins: int = 12,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_angle_bins = num_angle_bins

        # Shared layers
        self.shared_fc1 = nn.Linear(in_channels, 256)
        self.shared_bn1 = nn.BatchNorm1d(256)
        self.shared_fc2 = nn.Linear(256, 256)
        self.shared_bn2 = nn.BatchNorm1d(256)

        # Center branch (x, y, z)
        self.center_fc1 = nn.Linear(256, 128)
        self.center_bn1 = nn.BatchNorm1d(128)
        self.center_fc2 = nn.Linear(128, 3)

        # Size branch (w, h, l)
        self.size_fc1 = nn.Linear(256, 128)
        self.size_bn1 = nn.BatchNorm1d(128)
        self.size_fc2 = nn.Linear(128, 3)

        # Angle branch: bin classification + per-bin residual
        self.angle_fc1 = nn.Linear(256, 128)
        self.angle_bn1 = nn.BatchNorm1d(128)
        self.angle_cls_fc = nn.Linear(128, num_angle_bins)
        self.angle_res_fc = nn.Linear(128, num_angle_bins)

        # Class branch
        self.cls_fc1 = nn.Linear(256, 128)
        self.cls_bn1 = nn.BatchNorm1d(128)
        self.cls_fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Per-point features, shape (B, N, in_channels)

        Returns:
            Dictionary with:
                'center': (B, N, 3) - center offsets
                'size': (B, N, 3) - size predictions (w, h, l)
                'angle_cls': (B, N, num_angle_bins) - angle bin logits
                'angle_res': (B, N, num_angle_bins) - angle residuals
                'cls_scores': (B, N, num_classes) - class logits
        """
        B, N, _ = x.shape

        # Flatten for batch norm: (B*N, C)
        x_flat = x.reshape(B * N, -1)

        # Shared layers
        shared = F.relu(self.shared_bn1(self.shared_fc1(x_flat)))
        shared = F.relu(self.shared_bn2(self.shared_fc2(shared)))

        # Center branch
        center = F.relu(self.center_bn1(self.center_fc1(shared)))
        center = self.center_fc2(center)
        center = center.view(B, N, 3)

        # Size branch
        size = F.relu(self.size_bn1(self.size_fc1(shared)))
        size = self.size_fc2(size)
        size = size.view(B, N, 3)

        # Angle branch
        angle_feat = F.relu(self.angle_bn1(self.angle_fc1(shared)))
        angle_cls = self.angle_cls_fc(angle_feat).view(B, N, self.num_angle_bins)
        angle_res = self.angle_res_fc(angle_feat).view(B, N, self.num_angle_bins)

        # Classification branch
        cls_feat = F.relu(self.cls_bn1(self.cls_fc1(shared)))
        cls_scores = self.cls_fc2(cls_feat).view(B, N, self.num_classes)

        return {
            "center": center,
            "size": size,
            "angle_cls": angle_cls,
            "angle_res": angle_res,
            "cls_scores": cls_scores,
        }


class SegmentationHead(nn.Module):
    """
    Per-point segmentation head using Conv1d layers.

    Applies a series of 1D convolutions to per-point features
    to produce per-point class scores.

    Args:
        in_channels: Number of input feature channels
        num_seg_classes: Number of segmentation classes
        hidden_channels: List of hidden layer channel sizes
    """

    def __init__(
        self,
        in_channels: int,
        num_seg_classes: int,
        hidden_channels: list = None,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [256, 256, 128]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        last_channel = in_channels
        for hc in hidden_channels:
            self.convs.append(nn.Conv1d(last_channel, hc, 1))
            self.bns.append(nn.BatchNorm1d(hc))
            last_channel = hc

        self.drop = nn.Dropout(0.5)
        self.final_conv = nn.Conv1d(last_channel, num_seg_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Per-point features, shape (B, N, in_channels)

        Returns:
            Per-point class logits, shape (B, N, num_seg_classes)
        """
        # (B, N, C) -> (B, C, N) for Conv1d
        x = x.permute(0, 2, 1)

        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))

        x = self.drop(x)
        x = self.final_conv(x)  # (B, num_seg_classes, N)

        # Back to (B, N, num_seg_classes)
        x = x.permute(0, 2, 1)

        return x
