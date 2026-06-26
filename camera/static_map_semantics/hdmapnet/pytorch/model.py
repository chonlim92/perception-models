"""
HDMapNet: An Online HD Map Construction and Evaluation Framework.

Main model class that integrates all components:
- Multi-camera backbone feature extraction
- View transformation (IPM or LSS) to BEV
- BEV encoder for feature refinement
- Semantic, instance, and direction prediction heads

Configuration is passed as a dictionary to the constructor.
"""

import torch
import torch.nn as nn

from .backbone import EfficientNetB0Backbone, ResNet50Backbone
from .view_transform import IPMTransform, LSSTransform
from .bev_encoder import BEVEncoder
from .heads import SemanticHead, InstanceHead, DirectionHead


DEFAULT_CONFIG = {
    # Backbone
    "backbone": "efficientnet-b0",  # "efficientnet-b0" or "resnet-50"
    "pretrained_backbone": True,
    "backbone_out_channels": 64,

    # View transform
    "view_transform": "lss",  # "ipm" or "lss"
    "xbound": [-30.0, 30.0, 0.3],   # [min, max, resolution] meters
    "ybound": [-15.0, 15.0, 0.3],   # [min, max, resolution] meters
    "zbound": [-10.0, 10.0, 20.0],  # [min, max, resolution] meters
    "dbound": [4.0, 45.0, 1.0],     # [min, max, resolution] meters (41 bins)

    # Image
    "image_size": [128, 352],  # (H, W) after resize
    "num_cameras": 6,
    "feature_stride": 8,

    # BEV encoder
    "bev_encoder_in_channels": 64,
    "bev_encoder_base_channels": 64,

    # Heads
    "num_classes": 3,  # divider, boundary, crossing
    "embedding_dim": 16,
    "head_mid_channels": 64,
}


class HDMapNet(nn.Module):
    """HDMapNet: Online HD map construction from multi-camera images.

    Architecture:
        Multi-cam images -> Backbone -> View Transform -> BEV Encoder -> Heads
                                                                      |-> Semantic
                                                                      |-> Instance
                                                                      |-> Direction
    """

    def __init__(self, config=None):
        """
        Args:
            config: Configuration dictionary. Missing keys will use DEFAULT_CONFIG values.
        """
        super().__init__()

        # Merge config with defaults
        self.config = {**DEFAULT_CONFIG}
        if config is not None:
            self.config.update(config)

        cfg = self.config

        # Build backbone
        if cfg["backbone"] == "efficientnet-b0":
            self.backbone = EfficientNetB0Backbone(
                pretrained=cfg["pretrained_backbone"],
                out_channels=cfg["backbone_out_channels"],
            )
        elif cfg["backbone"] == "resnet-50":
            self.backbone = ResNet50Backbone(
                pretrained=cfg["pretrained_backbone"],
                out_channels=cfg["backbone_out_channels"],
            )
        else:
            raise ValueError(f"Unknown backbone: {cfg['backbone']}")

        backbone_out_ch = self.backbone.out_channels

        # Build view transform
        image_size = cfg["image_size"]
        xbound = cfg["xbound"]
        ybound = cfg["ybound"]

        if cfg["view_transform"] == "ipm":
            self.view_transform = IPMTransform(
                xbound=xbound,
                ybound=ybound,
                image_size=image_size,
                feature_stride=cfg["feature_stride"],
            )
        elif cfg["view_transform"] == "lss":
            self.view_transform = LSSTransform(
                in_channels=backbone_out_ch,
                xbound=xbound,
                ybound=ybound,
                zbound=cfg["zbound"],
                dbound=cfg["dbound"],
                image_size=image_size,
                feature_stride=cfg["feature_stride"],
            )
        else:
            raise ValueError(f"Unknown view transform: {cfg['view_transform']}")

        # Compute BEV grid size
        self.bev_h = int((ybound[1] - ybound[0]) / ybound[2])
        self.bev_w = int((xbound[1] - xbound[0]) / xbound[2])

        # Build BEV encoder
        self.bev_encoder = BEVEncoder(
            in_channels=cfg["bev_encoder_in_channels"],
            base_channels=cfg["bev_encoder_base_channels"],
        )
        bev_out_ch = self.bev_encoder.out_channels

        # Build prediction heads
        self.semantic_head = SemanticHead(
            in_channels=bev_out_ch,
            num_classes=cfg["num_classes"],
            mid_channels=cfg["head_mid_channels"],
        )
        self.instance_head = InstanceHead(
            in_channels=bev_out_ch,
            embedding_dim=cfg["embedding_dim"],
            mid_channels=cfg["head_mid_channels"],
        )
        self.direction_head = DirectionHead(
            in_channels=bev_out_ch,
            mid_channels=cfg["head_mid_channels"],
        )

    def extract_features(self, images):
        """Extract backbone features from multi-camera images.

        Args:
            images: (B, N_cams, 3, H, W) multi-camera input images.

        Returns:
            features: (B, N_cams, C, fH, fW) feature maps.
        """
        B, N, C_in, H, W = images.shape

        # Reshape to process all cameras together
        x = images.reshape(B * N, C_in, H, W)

        # Extract features
        feats = self.backbone(x)  # (B*N, C, fH, fW)

        # Reshape back
        C_out = feats.shape[1]
        fH, fW = feats.shape[2], feats.shape[3]
        feats = feats.reshape(B, N, C_out, fH, fW)

        return feats

    def forward(self, images, intrinsics, extrinsics):
        """
        Args:
            images: Multi-camera images (B, N_cams, 3, H, W).
            intrinsics: Camera intrinsic matrices (B, N_cams, 3, 3).
            extrinsics: Camera extrinsic matrices (B, N_cams, 4, 4).
                        For IPM: world-to-camera transforms.
                        For LSS: camera-to-ego transforms.

        Returns:
            Dict with predictions:
                - 'semantic': (B, num_classes, bev_h, bev_w) logits
                - 'instance': (B, embedding_dim, bev_h, bev_w) embeddings
                - 'direction': (B, 2, bev_h, bev_w) unit direction vectors
        """
        # Extract multi-camera features
        features = self.extract_features(images)  # (B, N, C, fH, fW)

        # Transform to BEV
        bev_features = self.view_transform(features, intrinsics, extrinsics)  # (B, C, bev_h, bev_w)

        # Encode BEV features
        bev_encoded = self.bev_encoder(bev_features)  # (B, C, bev_h, bev_w)

        # Prediction heads
        semantic = self.semantic_head(bev_encoded)    # (B, num_classes, bev_h, bev_w)
        instance = self.instance_head(bev_encoded)    # (B, embedding_dim, bev_h, bev_w)
        direction = self.direction_head(bev_encoded)  # (B, 2, bev_h, bev_w)

        return {
            "semantic": semantic,
            "instance": instance,
            "direction": direction,
        }

    def get_bev_features(self, images, intrinsics, extrinsics):
        """Get intermediate BEV features (useful for visualization/debugging).

        Args:
            images: Multi-camera images (B, N_cams, 3, H, W).
            intrinsics: Camera intrinsic matrices (B, N_cams, 3, 3).
            extrinsics: Camera extrinsic matrices (B, N_cams, 4, 4).

        Returns:
            Dict with intermediate features:
                - 'cam_features': (B, N, C, fH, fW)
                - 'bev_raw': (B, C, bev_h, bev_w) before encoder
                - 'bev_encoded': (B, C, bev_h, bev_w) after encoder
        """
        features = self.extract_features(images)
        bev_raw = self.view_transform(features, intrinsics, extrinsics)
        bev_encoded = self.bev_encoder(bev_raw)

        return {
            "cam_features": features,
            "bev_raw": bev_raw,
            "bev_encoded": bev_encoded,
        }
