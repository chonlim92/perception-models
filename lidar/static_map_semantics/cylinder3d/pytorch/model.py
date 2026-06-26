"""
Cylinder3D Full Model.

Integrates all components into a complete end-to-end model:
    1. CylindricalPartition: Point cloud -> cylindrical voxel representation
    2. Cylinder3DBackbone: Voxel features -> voxel-level semantic predictions
    3. PointRefinementModule: Voxel predictions + point features -> per-point predictions

Reference:
    Zhu et al., "Cylinder3D: An Effective 3D Framework for Driving-scene
    LiDAR Semantic Segmentation", CVPR 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

from .cylindrical_partition import CylindricalPartition
from .backbone import Cylinder3DBackbone
from .point_refinement import PointRefinementModule


class Cylinder3D(nn.Module):
    """
    Complete Cylinder3D model for LiDAR semantic segmentation.

    Pipeline:
        1. CylindricalPartition converts raw points to cylindrical voxel features
        2. Dense voxel volume is constructed from sparse voxel features
        3. Cylinder3DBackbone produces voxel-level predictions
        4. PointRefinementModule refines to per-point predictions

    Args:
        num_classes: Number of semantic classes. Default: 20
        grid_size: Voxel grid dimensions [rho, theta, z]. Default: [480, 360, 32]
        rho_range: Radial distance range. Default: [0.0, 50.0]
        theta_range: Azimuth angle range. Default: [-pi, pi]
        z_range: Height range. Default: [-3.0, 1.0]
        base_channels: Base channel count for backbone. Default: 32
        use_point_refinement: Whether to use point refinement. Default: True
    """

    def __init__(
        self,
        num_classes: int = 20,
        grid_size: Optional[List[int]] = None,
        rho_range: Optional[List[float]] = None,
        theta_range: Optional[List[float]] = None,
        z_range: Optional[List[float]] = None,
        base_channels: int = 32,
        use_point_refinement: bool = True,
    ):
        super().__init__()

        import numpy as np

        if grid_size is None:
            grid_size = [480, 360, 32]
        if rho_range is None:
            rho_range = [0.0, 50.0]
        if theta_range is None:
            theta_range = [-np.pi, np.pi]
        if z_range is None:
            z_range = [-3.0, 1.0]

        self.num_classes = num_classes
        self.grid_size = grid_size
        self.use_point_refinement = use_point_refinement

        # Component 1: Cylindrical Partition
        self.partition = CylindricalPartition(
            grid_size=grid_size,
            rho_range=rho_range,
            theta_range=theta_range,
            z_range=z_range,
        )

        # Point feature dimension from partition
        point_feat_dim = self.partition.point_feature_dim  # 9

        # Component 2: Backbone
        self.backbone = Cylinder3DBackbone(
            input_channels=point_feat_dim,
            num_classes=num_classes,
            base_channels=base_channels,
        )

        # Component 3: Point Refinement
        if use_point_refinement:
            # Decoder output channels = first element of encoder_channels (after full decoder)
            decoder_out_channels = base_channels  # 32
            self.point_refinement = PointRefinementModule(
                voxel_feature_dim=decoder_out_channels,
                point_feature_dim=point_feat_dim,
                num_classes=num_classes,
            )

    def scatter_to_dense_volume(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        grid_size: List[int],
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Convert sparse voxel features to a dense 3D volume.

        Args:
            voxel_features: (M, C) features for M occupied voxels
            voxel_coords: (M, 3) grid coordinates [rho_idx, theta_idx, z_idx]
            grid_size: [D_rho, D_theta, D_z]
            batch_size: Number of samples in batch (default: 1)

        Returns:
            volume: (B, C, D_rho, D_theta, D_z) dense feature volume
        """
        C = voxel_features.shape[1]
        device = voxel_features.device
        dtype = voxel_features.dtype

        volume = torch.zeros(
            batch_size, C, grid_size[0], grid_size[1], grid_size[2],
            device=device, dtype=dtype,
        )

        # Scatter voxel features into the dense volume
        rho_idx = voxel_coords[:, 0].long()
        theta_idx = voxel_coords[:, 1].long()
        z_idx = voxel_coords[:, 2].long()

        # Assuming batch_size=1 for single point cloud
        volume[0, :, rho_idx, theta_idx, z_idx] = voxel_features.t()

        return volume

    def forward(
        self, points: torch.Tensor, num_points: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass of Cylinder3D.

        Args:
            points: (N, 4+) point cloud tensor [x, y, z, intensity, ...]
                    For batched input, all points are concatenated and num_points
                    specifies per-sample counts.
            num_points: (B,) number of points per sample in the batch.
                        If None, assumes single sample (batch_size=1).

        Returns:
            Dictionary containing:
                - 'voxel_logits': (B, num_classes, D_rho, D_theta, D_z) voxel predictions
                - 'point_logits': (N, num_classes) per-point predictions (if point refinement enabled)
                - 'point_to_voxel': (N,) point-to-voxel mapping
        """
        # Determine batch size
        if num_points is None:
            batch_size = 1
            point_splits = [points.shape[0]]
        else:
            batch_size = num_points.shape[0]
            point_splits = num_points.tolist()

        # For simplicity, process each sample independently and stack
        # (In production, you'd batch this more efficiently)
        all_voxel_logits = []
        all_point_logits = []
        all_point_to_voxel = []
        all_point_features = []
        all_voxel_coords = []

        point_offset = 0
        for b in range(batch_size):
            n_pts = point_splits[b]
            sample_points = points[point_offset:point_offset + n_pts]

            # Step 1: Cylindrical partition
            partition_out = self.partition(sample_points)
            voxel_features = partition_out["voxel_features"]
            voxel_coords = partition_out["voxel_coords"]
            point_to_voxel = partition_out["point_to_voxel"]
            point_features = partition_out["point_features"]

            # Step 2: Construct dense volume
            volume = self.scatter_to_dense_volume(
                voxel_features, voxel_coords, self.grid_size, batch_size=1
            )

            # Step 3: Backbone
            backbone_out = self.backbone(volume)
            voxel_logits = backbone_out["voxel_logits"]
            voxel_feat_volume = backbone_out["voxel_features"]

            all_voxel_logits.append(voxel_logits)

            # Step 4: Point refinement
            if self.use_point_refinement:
                point_logits = self.point_refinement(
                    voxel_features_volume=voxel_feat_volume,
                    point_features=point_features,
                    point_to_voxel=point_to_voxel,
                    voxel_coords=voxel_coords,
                    grid_size=self.grid_size,
                )
                all_point_logits.append(point_logits)

            all_point_to_voxel.append(point_to_voxel)
            all_point_features.append(point_features)
            all_voxel_coords.append(voxel_coords)

            point_offset += n_pts

        # Stack batch outputs
        voxel_logits_batch = torch.cat(all_voxel_logits, dim=0)  # (B, C, D, H, W)

        result = {
            "voxel_logits": voxel_logits_batch,
            "point_to_voxel": torch.cat(all_point_to_voxel, dim=0),
        }

        if self.use_point_refinement:
            result["point_logits"] = torch.cat(all_point_logits, dim=0)

        return result


def create_cylinder3d(
    num_classes: int = 20,
    grid_size: Optional[List[int]] = None,
    preset: str = "default",
    **kwargs,
) -> Cylinder3D:
    """
    Factory function to create a Cylinder3D model with common configurations.

    Args:
        num_classes: Number of semantic classes.
        grid_size: Voxel grid size. If None, uses preset default.
        preset: Configuration preset name:
            - 'default': Standard Cylinder3D (480x360x32 grid)
            - 'small': Reduced grid for faster training (240x180x16)
            - 'large': High-resolution grid (480x360x64)
        **kwargs: Additional arguments passed to Cylinder3D constructor.

    Returns:
        Configured Cylinder3D model instance.
    """
    import numpy as np

    presets = {
        "default": {
            "grid_size": [480, 360, 32],
            "rho_range": [0.0, 50.0],
            "theta_range": [-np.pi, np.pi],
            "z_range": [-3.0, 1.0],
            "base_channels": 32,
        },
        "small": {
            "grid_size": [240, 180, 16],
            "rho_range": [0.0, 50.0],
            "theta_range": [-np.pi, np.pi],
            "z_range": [-3.0, 1.0],
            "base_channels": 16,
        },
        "large": {
            "grid_size": [480, 360, 64],
            "rho_range": [0.0, 70.0],
            "theta_range": [-np.pi, np.pi],
            "z_range": [-4.0, 2.0],
            "base_channels": 32,
        },
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(presets.keys())}")

    config = presets[preset]

    if grid_size is not None:
        config["grid_size"] = grid_size

    config.update(kwargs)

    model = Cylinder3D(num_classes=num_classes, **config)

    return model
