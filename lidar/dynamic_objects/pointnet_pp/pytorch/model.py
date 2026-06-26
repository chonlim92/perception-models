"""
Full PointNet++ model variants for classification, detection, and segmentation.

This module assembles the PointNet++ backbone (set abstraction layers) with
task-specific heads to create end-to-end models for 3D understanding tasks.
"""

import torch
import torch.nn as nn

from .pointnet_modules import (
    PointNetSetAbstraction,
    PointNetSetAbstractionMsg,
    PointNetFeaturePropagation,
)
from .heads import ClassificationHead, DetectionHead, SegmentationHead


class PointNetPPClassification(nn.Module):
    """
    PointNet++ for point cloud classification.

    Architecture:
        SA1: 4096 -> 1024 points (radius=0.1, nsample=32)
        SA2: 1024 -> 256 points (radius=0.2, nsample=64)
        SA3: 256 -> 64 points (radius=0.4, nsample=128)
        SA4: Global feature (group_all)
        Classification Head: FC layers -> num_classes

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels per point (default 3 for xyz only,
                     use 6 for xyz+normals, etc.)
        use_msg: Whether to use multi-scale grouping (default False)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        use_msg: bool = False,
    ):
        super().__init__()
        self.use_msg = use_msg

        # Additional features beyond xyz
        extra_channels = in_channels - 3

        if use_msg:
            self.sa1 = PointNetSetAbstractionMsg(
                npoint=4096,
                radius_list=[0.05, 0.1],
                nsample_list=[16, 32],
                in_channel=extra_channels,
                mlp_list=[[16, 16, 32], [32, 32, 64]],
            )
            sa1_out = 32 + 64  # sum of last channels from each scale

            self.sa2 = PointNetSetAbstractionMsg(
                npoint=1024,
                radius_list=[0.1, 0.2],
                nsample_list=[32, 64],
                in_channel=sa1_out,
                mlp_list=[[64, 64, 128], [64, 96, 128]],
            )
            sa2_out = 128 + 128

            self.sa3 = PointNetSetAbstractionMsg(
                npoint=256,
                radius_list=[0.2, 0.4],
                nsample_list=[64, 128],
                in_channel=sa2_out,
                mlp_list=[[128, 196, 256], [128, 196, 256]],
            )
            sa3_out = 256 + 256

            self.sa4 = PointNetSetAbstraction(
                npoint=None,
                radius=None,
                nsample=None,
                in_channel=sa3_out + 3,
                mlp=[256, 512, 1024],
                group_all=True,
            )
        else:
            self.sa1 = PointNetSetAbstraction(
                npoint=4096,
                radius=0.1,
                nsample=32,
                in_channel=extra_channels + 3,
                mlp=[64, 64, 128],
            )
            self.sa2 = PointNetSetAbstraction(
                npoint=1024,
                radius=0.2,
                nsample=64,
                in_channel=128 + 3,
                mlp=[128, 128, 256],
            )
            self.sa3 = PointNetSetAbstraction(
                npoint=256,
                radius=0.4,
                nsample=128,
                in_channel=256 + 3,
                mlp=[256, 256, 512],
            )
            self.sa4 = PointNetSetAbstraction(
                npoint=None,
                radius=None,
                nsample=None,
                in_channel=512 + 3,
                mlp=[256, 512, 1024],
                group_all=True,
            )

        self.head = ClassificationHead(1024, num_classes)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            xyz: Point coordinates, shape (B, N, 3)
            features: Additional per-point features, shape (B, N, C) or None

        Returns:
            Class logits, shape (B, num_classes)
        """
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Global feature: (B, 1, 1024) -> (B, 1024)
        global_feat = l4_points.squeeze(1)

        logits = self.head(global_feat)
        return logits


class PointNetPPDetection(nn.Module):
    """
    PointNet++ for 3D object detection.

    Uses set abstraction layers for feature extraction followed by
    a detection head that predicts 3D bounding boxes.

    Architecture:
        SA1: N -> 4096 points
        SA2: 4096 -> 1024 points
        SA3: 1024 -> 512 points
        SA4: 512 -> 256 points
        Detection Head on 256 proposal points:
            predicts center, size, angle, class per point

    Args:
        num_classes: Number of object classes
        in_channels: Input feature channels (3 for xyz only)
        num_angle_bins: Number of angle bins for heading prediction
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        num_angle_bins: int = 12,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_angle_bins = num_angle_bins

        extra_channels = in_channels - 3

        # Encoder: Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=4096,
            radius=0.2,
            nsample=32,
            in_channel=extra_channels + 3,
            mlp=[64, 64, 128],
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=1024,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=512,
            radius=0.8,
            nsample=64,
            in_channel=256 + 3,
            mlp=[256, 256, 512],
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=256,
            radius=1.2,
            nsample=64,
            in_channel=512 + 3,
            mlp=[512, 512, 1024],
        )

        # Feature propagation for dense features at proposal level
        self.fp4 = PointNetFeaturePropagation(
            in_channel=1024 + 512,
            mlp=[512, 512],
        )
        self.fp3 = PointNetFeaturePropagation(
            in_channel=512 + 256,
            mlp=[256, 256],
        )

        # Detection head operates on sa3-level points (512 proposals)
        self.detection_head = DetectionHead(
            in_channels=256,
            num_classes=num_classes,
            num_angle_bins=num_angle_bins,
        )

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor = None
    ) -> dict:
        """
        Args:
            xyz: Point coordinates, shape (B, N, 3)
            features: Additional per-point features, shape (B, N, C) or None

        Returns:
            Dictionary with detection predictions:
                'center': (B, 512, 3)
                'size': (B, 512, 3)
                'angle_cls': (B, 512, num_angle_bins)
                'angle_res': (B, 512, num_angle_bins)
                'cls_scores': (B, 512, num_classes)
                'proposal_xyz': (B, 512, 3) - proposal point locations
        """
        # Encoder
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Feature propagation (partial decoder for denser features)
        l3_points_fp = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points_fp = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_fp)

        # Detection head on l2-level points (1024 proposals)
        predictions = self.detection_head(l2_points_fp)
        predictions["proposal_xyz"] = l2_xyz

        return predictions


class PointNetPPSegmentation(nn.Module):
    """
    PointNet++ for per-point semantic segmentation.

    Uses an encoder-decoder architecture:
    - Encoder: Set Abstraction layers downsample and extract features
    - Decoder: Feature Propagation layers upsample features back to
      original resolution via inverse-distance weighted interpolation

    Args:
        num_seg_classes: Number of segmentation classes
        in_channels: Input feature channels (3 for xyz only)
    """

    def __init__(self, num_seg_classes: int, in_channels: int = 3):
        super().__init__()
        self.num_seg_classes = num_seg_classes
        extra_channels = in_channels - 3

        # Encoder: progressive downsampling
        self.sa1 = PointNetSetAbstraction(
            npoint=4096,
            radius=0.1,
            nsample=32,
            in_channel=extra_channels + 3,
            mlp=[32, 32, 64],
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=1024,
            radius=0.2,
            nsample=32,
            in_channel=64 + 3,
            mlp=[64, 64, 128],
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=256,
            radius=0.4,
            nsample=32,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=64,
            radius=0.8,
            nsample=32,
            in_channel=256 + 3,
            mlp=[256, 256, 512],
        )

        # Decoder: progressive upsampling with skip connections
        self.fp4 = PointNetFeaturePropagation(
            in_channel=512 + 256,  # from sa4 + skip from sa3
            mlp=[256, 256],
        )
        self.fp3 = PointNetFeaturePropagation(
            in_channel=256 + 128,  # from fp4 + skip from sa2
            mlp=[256, 128],
        )
        self.fp2 = PointNetFeaturePropagation(
            in_channel=128 + 64,  # from fp3 + skip from sa1
            mlp=[128, 128, 128],
        )
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + extra_channels,  # from fp2 + skip from input (or 0)
            mlp=[128, 128, 128],
        )

        # Segmentation head on full-resolution features
        self.seg_head = SegmentationHead(
            in_channels=128,
            num_seg_classes=num_seg_classes,
            hidden_channels=[128, 128],
        )

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            xyz: Point coordinates, shape (B, N, 3)
            features: Additional per-point features, shape (B, N, C) or None

        Returns:
            Per-point segmentation logits, shape (B, N, num_seg_classes)
        """
        # Store original for skip connection
        l0_xyz = xyz
        l0_points = features

        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Decoder with skip connections
        l3_points_dec = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points_dec = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_dec)
        l1_points_dec = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_dec)
        l0_points_dec = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points_dec)

        # Segmentation head
        seg_logits = self.seg_head(l0_points_dec)

        return seg_logits
