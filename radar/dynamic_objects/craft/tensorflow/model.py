"""
CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer.

TensorFlow 2.x / Keras implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # Camera branch
    "num_cameras": 6,
    "image_height": 256,
    "image_width": 704,
    "fpn_channels": 256,
    "backbone_name": "resnet50",
    # Radar branch
    "pillar_x_size": 0.2,
    "pillar_y_size": 0.2,
    "pillar_z_size": 8.0,
    "x_min": -51.2,
    "x_max": 51.2,
    "y_min": -51.2,
    "y_max": 51.2,
    "z_min": -5.0,
    "z_max": 3.0,
    "max_points_per_pillar": 32,
    "max_pillars": 20000,
    "pillar_feat_dim": 64,
    "bev_backbone_channels": [64, 128, 256],
    # Fusion transformer
    "fusion_embed_dim": 256,
    "fusion_num_heads": 8,
    "fusion_num_layers": 6,
    "fusion_ffn_dim": 512,
    "fusion_dropout": 0.1,
    # Detection head
    "num_classes": 10,
    "num_reg_attrs": 8,  # dx, dy, dz, w, l, h, sin(yaw), cos(yaw)
    "velocity_dim": 2,
    "heatmap_kernel_size": 3,
    # BEV grid for detection head
    "bev_x_cells": 512,
    "bev_y_cells": 512,
}


# ===========================================================================
# Camera branch components
# ===========================================================================


class FPN(layers.Layer):
    """Feature Pyramid Network with lateral connections and top-down pathway."""

    def __init__(self, out_channels: int = 256, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        num_levels = len(input_shape)
        self.lateral_convs = [
            layers.Conv2D(self.out_channels, 1, padding="same", name=f"lateral_{i}")
            for i in range(num_levels)
        ]
        self.output_convs = [
            layers.Conv2D(self.out_channels, 3, padding="same", name=f"fpn_out_{i}")
            for i in range(num_levels)
        ]
        super().build(input_shape)

    def call(self, features: List[tf.Tensor], training: bool = False) -> List[tf.Tensor]:
        """
        Args:
            features: list of feature maps from backbone [C2, C3, C4, C5],
                      each with shape (B, H_i, W_i, C_i).
        Returns:
            List of FPN feature maps, same spatial sizes, all with out_channels.
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            h, w = tf.shape(laterals[i])[1], tf.shape(laterals[i])[2]
            upsampled = tf.image.resize(laterals[i + 1], [h, w], method="nearest")
            laterals[i] = laterals[i] + upsampled

        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs


class CameraBackbone(layers.Layer):
    """ResNet50 backbone with FPN for multi-view image feature extraction."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.fpn_channels = config.get("fpn_channels", 256)

    def build(self, input_shape: tf.TensorShape) -> None:
        # Use Keras ResNet50 pre-trained on ImageNet (without top)
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,  # Will be loaded separately if needed
            input_shape=(
                self.config["image_height"],
                self.config["image_width"],
                3,
            ),
        )
        # Extract intermediate feature maps: C2, C3, C4, C5
        layer_names = [
            "conv2_block3_out",   # C2: stride 4
            "conv3_block4_out",   # C3: stride 8
            "conv4_block6_out",   # C4: stride 16
            "conv5_block3_out",   # C5: stride 32
        ]
        outputs = [base_model.get_layer(name).output for name in layer_names]
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=outputs, name="resnet50_backbone")
        self.fpn = FPN(out_channels=self.fpn_channels, name="fpn")
        super().build(input_shape)

    def call(self, images: tf.Tensor, training: bool = False) -> List[tf.Tensor]:
        """
        Args:
            images: (B, num_cameras, H, W, 3)
        Returns:
            List of FPN feature maps per camera, each (B * num_cameras, H_i, W_i, C).
        """
        b = tf.shape(images)[0]
        n_cams = self.config["num_cameras"]
        h, w = self.config["image_height"], self.config["image_width"]

        # Flatten cameras into batch dimension
        x = tf.reshape(images, [b * n_cams, h, w, 3])
        features = self.backbone(x, training=training)
        fpn_features = self.fpn(features, training=training)
        return fpn_features  # List of (B*N_cams, H_i, W_i, C)


# ===========================================================================
# Radar branch components
# ===========================================================================


class PillarEncoder(layers.Layer):
    """PointPillar-style encoder: per-point MLP then max-pool per pillar."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.feat_dim = config.get("pillar_feat_dim", 64)

    def build(self, input_shape: tf.TensorShape) -> None:
        # Input features per point: (x, y, z, rcs, vr, vr_comp, x_offset, y_offset, z_offset) = 9
        self.linear1 = layers.Dense(64, activation=None, name="pillar_fc1")
        self.bn1 = layers.BatchNormalization(name="pillar_bn1")
        self.linear2 = layers.Dense(self.feat_dim, activation=None, name="pillar_fc2")
        self.bn2 = layers.BatchNormalization(name="pillar_bn2")
        super().build(input_shape)

    def call(
        self,
        pillar_features: tf.Tensor,
        pillar_mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Args:
            pillar_features: (B, max_pillars, max_points_per_pillar, D_in)
            pillar_mask: (B, max_pillars, max_points_per_pillar) bool mask for valid points
        Returns:
            pillar_encodings: (B, max_pillars, feat_dim)
        """
        x = self.linear1(pillar_features)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.linear2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        # Mask invalid points before max pooling
        mask_expanded = tf.expand_dims(tf.cast(pillar_mask, tf.float32), axis=-1)
        x = x * mask_expanded + (1.0 - mask_expanded) * (-1e9)

        # Max-pool over points within each pillar
        pillar_encodings = tf.reduce_max(x, axis=2)  # (B, max_pillars, feat_dim)
        return pillar_encodings


class ScatterBEV(layers.Layer):
    """Scatter pillar features back to BEV pseudo-image grid."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        x_cells = int((config["x_max"] - config["x_min"]) / config["pillar_x_size"])
        y_cells = int((config["y_max"] - config["y_min"]) / config["pillar_y_size"])
        self.grid_x = x_cells
        self.grid_y = y_cells
        self.feat_dim = config.get("pillar_feat_dim", 64)

    def call(
        self,
        pillar_encodings: tf.Tensor,
        pillar_coords: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Args:
            pillar_encodings: (B, max_pillars, feat_dim)
            pillar_coords: (B, max_pillars, 2) integer grid coordinates (ix, iy)
        Returns:
            bev_image: (B, grid_y, grid_x, feat_dim)
        """
        batch_size = tf.shape(pillar_encodings)[0]
        max_pillars = tf.shape(pillar_encodings)[1]

        # Create batch indices
        batch_idx = tf.repeat(
            tf.range(batch_size)[:, tf.newaxis], max_pillars, axis=1
        )  # (B, max_pillars)

        ix = pillar_coords[:, :, 0]  # (B, max_pillars)
        iy = pillar_coords[:, :, 1]  # (B, max_pillars)

        # Clip to grid bounds
        ix = tf.clip_by_value(ix, 0, self.grid_x - 1)
        iy = tf.clip_by_value(iy, 0, self.grid_y - 1)

        # Flatten for scatter
        indices = tf.stack(
            [tf.reshape(batch_idx, [-1]),
             tf.reshape(iy, [-1]),
             tf.reshape(ix, [-1])],
            axis=1,
        )  # (B*max_pillars, 3)

        updates = tf.reshape(pillar_encodings, [-1, self.feat_dim])

        bev_image = tf.scatter_nd(
            indices,
            updates,
            shape=[batch_size, self.grid_y, self.grid_x, self.feat_dim],
        )
        return bev_image


class BEVBackbone(layers.Layer):
    """2D convolutional backbone on the BEV pseudo-image."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.channels = config.get("bev_backbone_channels", [64, 128, 256])

    def build(self, input_shape: tf.TensorShape) -> None:
        self.blocks = []
        for i, ch in enumerate(self.channels):
            block = tf.keras.Sequential([
                layers.Conv2D(ch, 3, strides=2, padding="same", name=f"conv_{i}_0"),
                layers.BatchNormalization(name=f"bn_{i}_0"),
                layers.ReLU(),
                layers.Conv2D(ch, 3, strides=1, padding="same", name=f"conv_{i}_1"),
                layers.BatchNormalization(name=f"bn_{i}_1"),
                layers.ReLU(),
                layers.Conv2D(ch, 3, strides=1, padding="same", name=f"conv_{i}_2"),
                layers.BatchNormalization(name=f"bn_{i}_2"),
                layers.ReLU(),
            ], name=f"bev_block_{i}")
            self.blocks.append(block)

        # Upsample and concatenate for multi-scale BEV features
        self.deblocks = []
        for i, ch in enumerate(self.channels):
            deblock = tf.keras.Sequential([
                layers.Conv2DTranspose(
                    self.channels[-1], 2 ** (len(self.channels) - 1 - i),
                    strides=2 ** (len(self.channels) - 1 - i),
                    padding="same",
                    name=f"deconv_{i}",
                ),
                layers.BatchNormalization(name=f"debn_{i}"),
                layers.ReLU(),
            ], name=f"bev_deblock_{i}")
            self.deblocks.append(deblock)

        self.compress = layers.Conv2D(
            self.channels[-1], 1, padding="same", name="bev_compress"
        )
        super().build(input_shape)

    def call(self, bev_image: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            bev_image: (B, H, W, C_in)
        Returns:
            bev_features: (B, H', W', C_out)
        """
        block_outputs = []
        x = bev_image
        for block in self.blocks:
            x = block(x, training=training)
            block_outputs.append(x)

        # Multi-scale fusion
        upsampled = []
        for deblock, feat in zip(self.deblocks, block_outputs):
            upsampled.append(deblock(feat, training=training))

        # All should be same spatial size now - resize to match smallest
        target_h = tf.shape(upsampled[-1])[1]
        target_w = tf.shape(upsampled[-1])[2]
        aligned = []
        for u in upsampled:
            resized = tf.image.resize(u, [target_h, target_w], method="bilinear")
            aligned.append(resized)

        concat = tf.concat(aligned, axis=-1)
        out = self.compress(concat)
        return out


class RadarBranch(layers.Layer):
    """Complete radar branch: pillar encoding + scatter + BEV backbone."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.pillar_encoder = PillarEncoder(config, name="pillar_encoder")
        self.scatter = ScatterBEV(config, name="scatter_bev")
        self.bev_backbone = BEVBackbone(config, name="bev_backbone")

    def call(
        self,
        pillar_features: tf.Tensor,
        pillar_mask: tf.Tensor,
        pillar_coords: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Args:
            pillar_features: (B, max_pillars, max_pts, D_in)
            pillar_mask: (B, max_pillars, max_pts)
            pillar_coords: (B, max_pillars, 2) - integer grid coords
        Returns:
            bev_features: (B, H_bev, W_bev, C)
        """
        encodings = self.pillar_encoder(pillar_features, pillar_mask, training=training)
        bev_image = self.scatter(encodings, pillar_coords, training=training)
        bev_features = self.bev_backbone(bev_image, training=training)
        return bev_features


# ===========================================================================
# Spatio-Contextual Fusion Transformer
# ===========================================================================


class MultiHeadCrossAttention(layers.Layer):
    """Multi-head cross-attention: queries from one modality, keys/values from another."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout

    def build(self, input_shape: Any) -> None:
        self.q_proj = layers.Dense(self.embed_dim, name="q_proj")
        self.k_proj = layers.Dense(self.embed_dim, name="k_proj")
        self.v_proj = layers.Dense(self.embed_dim, name="v_proj")
        self.out_proj = layers.Dense(self.embed_dim, name="out_proj")
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        key_padding_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Args:
            query: (B, N_q, D)
            key: (B, N_k, D)
            value: (B, N_k, D)
            key_padding_mask: (B, N_k) True = ignore
        Returns:
            (B, N_q, D)
        """
        b = tf.shape(query)[0]
        n_q = tf.shape(query)[1]
        n_k = tf.shape(key)[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (B, num_heads, N, head_dim)
        q = tf.reshape(q, [b, n_q, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.reshape(k, [b, n_k, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.reshape(v, [b, n_k, self.num_heads, self.head_dim])
        v = tf.transpose(v, [0, 2, 1, 3])

        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attn_weights = tf.matmul(q, k, transpose_b=True) / scale  # (B, H, N_q, N_k)

        if key_padding_mask is not None:
            # Expand mask: (B, 1, 1, N_k)
            mask = tf.cast(key_padding_mask[:, tf.newaxis, tf.newaxis, :], tf.float32)
            attn_weights = attn_weights + mask * (-1e9)

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        attn_output = tf.matmul(attn_weights, v)  # (B, H, N_q, head_dim)
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [b, n_q, self.embed_dim])
        return self.out_proj(attn_output)


class TransformerDecoderLayer(layers.Layer):
    """Single transformer decoder layer: self-attn + cross-attn + FFN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout

    def build(self, input_shape: Any) -> None:
        self.self_attn = MultiHeadCrossAttention(
            self.embed_dim, self.num_heads, self.dropout_rate, name="self_attn"
        )
        self.cross_attn = MultiHeadCrossAttention(
            self.embed_dim, self.num_heads, self.dropout_rate, name="cross_attn"
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.norm3 = layers.LayerNormalization(epsilon=1e-5, name="norm3")

        self.ffn = tf.keras.Sequential([
            layers.Dense(self.ffn_dim, activation="relu", name="ffn_fc1"),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embed_dim, name="ffn_fc2"),
            layers.Dropout(self.dropout_rate),
        ], name="ffn")
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(
        self,
        query: tf.Tensor,
        memory: tf.Tensor,
        query_mask: Optional[tf.Tensor] = None,
        memory_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Args:
            query: (B, N_q, D) radar BEV queries
            memory: (B, N_m, D) camera features
            query_mask: optional mask for self-attention
            memory_mask: optional mask for cross-attention keys
        Returns:
            (B, N_q, D)
        """
        # Self-attention
        residual = query
        x = self.norm1(query)
        x = self.self_attn(x, x, x, key_padding_mask=query_mask, training=training)
        x = self.dropout1(x, training=training)
        x = residual + x

        # Cross-attention (radar queries attend to camera memory)
        residual = x
        x_normed = self.norm2(x)
        x_cross = self.cross_attn(
            x_normed, memory, memory, key_padding_mask=memory_mask, training=training
        )
        x = residual + self.dropout2(x_cross, training=training)

        # Feed-forward network
        residual = x
        x = self.norm3(x)
        x = residual + self.ffn(x, training=training)

        return x


class RadarToImageProjection(layers.Layer):
    """Project radar BEV positions into camera image coordinates using calibration."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config

    def call(
        self,
        radar_bev_positions: tf.Tensor,
        lidar_to_cam: tf.Tensor,
        cam_intrinsics: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            radar_bev_positions: (B, N_radar, 3) 3D positions in ego frame
            lidar_to_cam: (B, num_cameras, 4, 4) extrinsic matrices
            cam_intrinsics: (B, num_cameras, 3, 3) intrinsic matrices
        Returns:
            proj_coords: (B, num_cameras, N_radar, 2) pixel coordinates
            valid_mask: (B, num_cameras, N_radar) boolean mask for visible points
        """
        b = tf.shape(radar_bev_positions)[0]
        n_radar = tf.shape(radar_bev_positions)[1]
        n_cams = self.config["num_cameras"]

        # Convert to homogeneous coordinates
        ones = tf.ones([b, n_radar, 1], dtype=radar_bev_positions.dtype)
        pts_homo = tf.concat([radar_bev_positions, ones], axis=-1)  # (B, N, 4)

        # Project to each camera
        # pts_homo: (B, 1, N, 4) -> broadcast with lidar_to_cam: (B, ncams, 4, 4)
        pts_expanded = pts_homo[:, tf.newaxis, :, :]  # (B, 1, N, 4)

        # Transform to camera frame: (B, ncams, 4, 4) @ (B, 1, N, 4)^T
        pts_cam = tf.einsum("bcij,bknj->bcki", lidar_to_cam, pts_expanded)  # (B, ncams, N, 4)
        pts_cam_3d = pts_cam[:, :, :, :3]  # (B, ncams, N, 3)

        # Project using intrinsics: (B, ncams, 3, 3) @ (B, ncams, N, 3)^T
        pts_img = tf.einsum("bcij,bcnj->bcni", cam_intrinsics, pts_cam_3d)  # (B, ncams, N, 3)

        # Normalize by depth
        depth = pts_img[:, :, :, 2:3]  # (B, ncams, N, 1)
        depth = tf.maximum(depth, 1e-5)
        proj_coords = pts_img[:, :, :, :2] / depth  # (B, ncams, N, 2)

        # Valid if in front of camera and within image bounds
        img_h = tf.cast(self.config["image_height"], tf.float32)
        img_w = tf.cast(self.config["image_width"], tf.float32)
        valid_depth = pts_cam_3d[:, :, :, 2] > 0.1
        valid_x = tf.logical_and(proj_coords[:, :, :, 0] >= 0, proj_coords[:, :, :, 0] < img_w)
        valid_y = tf.logical_and(proj_coords[:, :, :, 1] >= 0, proj_coords[:, :, :, 1] < img_h)
        valid_mask = tf.logical_and(valid_depth, tf.logical_and(valid_x, valid_y))

        return proj_coords, valid_mask


class SpatioContextualFusionTransformer(layers.Layer):
    """
    Fusion transformer that fuses radar BEV features with camera image features
    using spatio-contextual cross-attention.
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.embed_dim = config.get("fusion_embed_dim", 256)
        self.num_heads = config.get("fusion_num_heads", 8)
        self.num_layers = config.get("fusion_num_layers", 6)
        self.ffn_dim = config.get("fusion_ffn_dim", 512)
        self.dropout = config.get("fusion_dropout", 0.1)

    def build(self, input_shape: Any) -> None:
        self.radar_proj = layers.Dense(self.embed_dim, name="radar_proj")
        self.camera_proj = layers.Dense(self.embed_dim, name="camera_proj")

        self.projection_layer = RadarToImageProjection(self.config, name="r2i_proj")

        self.decoder_layers = [
            TransformerDecoderLayer(
                self.embed_dim, self.num_heads, self.ffn_dim, self.dropout,
                name=f"decoder_layer_{i}",
            )
            for i in range(self.num_layers)
        ]

        self.positional_encoding = layers.Dense(self.embed_dim, name="pos_enc")
        self.output_norm = layers.LayerNormalization(epsilon=1e-5, name="output_norm")
        super().build(input_shape)

    def _sample_camera_features(
        self,
        camera_features: tf.Tensor,
        proj_coords: tf.Tensor,
        valid_mask: tf.Tensor,
    ) -> tf.Tensor:
        """
        Sample camera features at projected radar positions using bilinear interpolation.

        Args:
            camera_features: (B*num_cams, H_feat, W_feat, C)
            proj_coords: (B, num_cams, N_radar, 2) pixel coords
            valid_mask: (B, num_cams, N_radar)
        Returns:
            sampled_features: (B, N_radar, C) aggregated across cameras
        """
        b = tf.shape(proj_coords)[0]
        n_cams = self.config["num_cameras"]
        n_radar = tf.shape(proj_coords)[2]

        feat_h = tf.shape(camera_features)[1]
        feat_w = tf.shape(camera_features)[2]
        feat_c = tf.shape(camera_features)[3]

        # Reshape camera features: (B, num_cams, H, W, C)
        cam_feats = tf.reshape(camera_features, [b, n_cams, feat_h, feat_w, feat_c])

        # Normalize projection coords to feature map scale
        img_h = tf.cast(self.config["image_height"], tf.float32)
        img_w = tf.cast(self.config["image_width"], tf.float32)
        scale_h = tf.cast(feat_h, tf.float32) / img_h
        scale_w = tf.cast(feat_w, tf.float32) / img_w

        # Scale coordinates to feature map
        coords_scaled = tf.stack([
            proj_coords[:, :, :, 0] * scale_w,
            proj_coords[:, :, :, 1] * scale_h,
        ], axis=-1)  # (B, ncams, N_radar, 2)

        # Bilinear sampling per camera
        all_sampled = []
        for cam_idx in range(n_cams):
            # Get feature map for this camera: (B, H, W, C)
            feat_map = cam_feats[:, cam_idx]
            # Coordinates for this camera: (B, N_radar, 2)
            coords = coords_scaled[:, cam_idx]  # (B, N_radar, 2) - (x, y)

            # Normalize to [-1, 1] for grid_sample-like behavior
            x_norm = 2.0 * coords[:, :, 0] / tf.cast(feat_w - 1, tf.float32) - 1.0
            y_norm = 2.0 * coords[:, :, 1] / tf.cast(feat_h - 1, tf.float32) - 1.0
            grid = tf.stack([x_norm, y_norm], axis=-1)  # (B, N_radar, 2)

            # Reshape grid for dense_image_warp-compatible format or use manual bilinear
            # Manual bilinear interpolation
            x_coord = coords[:, :, 0]  # (B, N_radar)
            y_coord = coords[:, :, 1]

            x0 = tf.cast(tf.floor(x_coord), tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y_coord), tf.int32)
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, 0, feat_w - 1)
            x1 = tf.clip_by_value(x1, 0, feat_w - 1)
            y0 = tf.clip_by_value(y0, 0, feat_h - 1)
            y1 = tf.clip_by_value(y1, 0, feat_h - 1)

            # Gather pixel values
            def _gather_2d(feat: tf.Tensor, y_idx: tf.Tensor, x_idx: tf.Tensor) -> tf.Tensor:
                """Gather from (B, H, W, C) at positions (B, N)."""
                b_size = tf.shape(feat)[0]
                n_pts = tf.shape(y_idx)[1]
                b_idx = tf.repeat(tf.range(b_size)[:, tf.newaxis], n_pts, axis=1)
                indices = tf.stack([b_idx, y_idx, x_idx], axis=-1)  # (B, N, 3)
                return tf.gather_nd(feat, indices)  # (B, N, C)

            f00 = _gather_2d(feat_map, y0, x0)
            f01 = _gather_2d(feat_map, y0, x1)
            f10 = _gather_2d(feat_map, y1, x0)
            f11 = _gather_2d(feat_map, y1, x1)

            # Bilinear weights
            wa = tf.expand_dims((tf.cast(x1, tf.float32) - x_coord) * (tf.cast(y1, tf.float32) - y_coord), -1)
            wb = tf.expand_dims((x_coord - tf.cast(x0, tf.float32)) * (tf.cast(y1, tf.float32) - y_coord), -1)
            wc = tf.expand_dims((tf.cast(x1, tf.float32) - x_coord) * (y_coord - tf.cast(y0, tf.float32)), -1)
            wd = tf.expand_dims((x_coord - tf.cast(x0, tf.float32)) * (y_coord - tf.cast(y0, tf.float32)), -1)

            sampled = f00 * wa + f01 * wb + f10 * wc + f11 * wd  # (B, N_radar, C)
            all_sampled.append(sampled)

        # Stack and aggregate with valid mask: (B, ncams, N_radar, C)
        all_sampled = tf.stack(all_sampled, axis=1)
        mask_expanded = tf.cast(valid_mask, tf.float32)[:, :, :, tf.newaxis]
        # Weighted average across cameras
        numerator = tf.reduce_sum(all_sampled * mask_expanded, axis=1)
        denominator = tf.reduce_sum(mask_expanded, axis=1) + 1e-6
        aggregated = numerator / denominator  # (B, N_radar, C)

        return aggregated

    def call(
        self,
        radar_bev_features: tf.Tensor,
        camera_fpn_features: List[tf.Tensor],
        radar_bev_positions: tf.Tensor,
        lidar_to_cam: tf.Tensor,
        cam_intrinsics: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Args:
            radar_bev_features: (B, H_bev, W_bev, C_radar)
            camera_fpn_features: list of (B*ncams, H_i, W_i, C_fpn)
            radar_bev_positions: (B, N_radar, 3) 3D positions for radar BEV cells
            lidar_to_cam: (B, num_cameras, 4, 4)
            cam_intrinsics: (B, num_cameras, 3, 3)
        Returns:
            fused_bev: (B, H_bev, W_bev, embed_dim)
        """
        b = tf.shape(radar_bev_features)[0]
        h_bev = tf.shape(radar_bev_features)[1]
        w_bev = tf.shape(radar_bev_features)[2]

        # Flatten radar BEV to sequence
        n_radar = h_bev * w_bev
        radar_seq = tf.reshape(radar_bev_features, [b, n_radar, -1])
        radar_seq = self.radar_proj(radar_seq)  # (B, N_radar, embed_dim)

        # Use the highest-resolution FPN feature for cross-attention
        cam_features = camera_fpn_features[0]  # (B*ncams, H_feat, W_feat, C)

        # Project radar positions to camera
        proj_coords, valid_mask = self.projection_layer(
            radar_bev_positions, lidar_to_cam, cam_intrinsics
        )

        # Sample camera features at projected positions
        sampled_cam = self._sample_camera_features(cam_features, proj_coords, valid_mask)
        camera_memory = self.camera_proj(sampled_cam)  # (B, N_radar, embed_dim)

        # Add positional encoding
        positions_flat = tf.reshape(radar_bev_positions, [b, n_radar, 3])
        pos_enc = self.positional_encoding(positions_flat)
        radar_seq = radar_seq + pos_enc

        # Transformer decoder layers
        x = radar_seq
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, camera_memory, training=training)

        x = self.output_norm(x)

        # Reshape back to BEV spatial
        fused_bev = tf.reshape(x, [b, h_bev, w_bev, self.embed_dim])
        return fused_bev


# ===========================================================================
# Detection Head
# ===========================================================================


class AnchorFreeDetectionHead(layers.Layer):
    """
    Anchor-free detection head with center heatmap, box regression, and velocity.
    Similar to CenterPoint-style detection.
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.num_classes = config.get("num_classes", 10)
        self.num_reg_attrs = config.get("num_reg_attrs", 8)
        self.velocity_dim = config.get("velocity_dim", 2)

    def build(self, input_shape: tf.TensorShape) -> None:
        in_channels = input_shape[-1]

        # Shared convolution layers
        self.shared_conv = tf.keras.Sequential([
            layers.Conv2D(256, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(256, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="shared_conv")

        # Heatmap head (per-class center probability)
        self.heatmap_head = tf.keras.Sequential([
            layers.Conv2D(128, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(self.num_classes, 1, padding="same"),
        ], name="heatmap_head")

        # Regression head (offset + size + rotation)
        self.regression_head = tf.keras.Sequential([
            layers.Conv2D(128, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(self.num_reg_attrs, 1, padding="same"),
        ], name="regression_head")

        # Velocity head
        self.velocity_head = tf.keras.Sequential([
            layers.Conv2D(64, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(self.velocity_dim, 1, padding="same"),
        ], name="velocity_head")

        # Height head (z center + height)
        self.height_head = tf.keras.Sequential([
            layers.Conv2D(64, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(2, 1, padding="same"),  # z_center, height
        ], name="height_head")

        super().build(input_shape)

    def call(self, bev_features: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
        """
        Args:
            bev_features: (B, H, W, C)
        Returns:
            dict with keys:
                heatmap: (B, H, W, num_classes) - sigmoid activated
                regression: (B, H, W, num_reg_attrs) - dx, dy, dz, w, l, h, sin, cos
                velocity: (B, H, W, 2) - vx, vy
                height: (B, H, W, 2) - z_center, height
        """
        shared = self.shared_conv(bev_features, training=training)

        heatmap = self.heatmap_head(shared, training=training)
        heatmap = tf.sigmoid(heatmap)

        regression = self.regression_head(shared, training=training)
        velocity = self.velocity_head(shared, training=training)
        height = self.height_head(shared, training=training)

        return {
            "heatmap": heatmap,
            "regression": regression,
            "velocity": velocity,
            "height": height,
        }


# ===========================================================================
# Full CRAFT Model
# ===========================================================================


class CRAFTModel(tf.keras.Model):
    """
    CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer.

    End-to-end model combining:
    1. Camera branch (ResNet50 + FPN)
    2. Radar branch (PointPillar + BEV backbone)
    3. Spatio-Contextual Fusion Transformer
    4. Anchor-free detection head
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Camera branch
        self.camera_backbone = CameraBackbone(self.config, name="camera_backbone")

        # Radar branch
        self.radar_branch = RadarBranch(self.config, name="radar_branch")

        # Fusion transformer
        self.fusion_transformer = SpatioContextualFusionTransformer(
            self.config, name="fusion_transformer"
        )

        # Detection head
        self.detection_head = AnchorFreeDetectionHead(self.config, name="detection_head")

    def _generate_bev_positions(self, batch_size: int, h_bev: int, w_bev: int) -> tf.Tensor:
        """Generate 3D positions for each BEV grid cell center."""
        x_min = self.config["x_min"]
        x_max = self.config["x_max"]
        y_min = self.config["y_min"]
        y_max = self.config["y_max"]

        # Create grid of BEV cell centers
        xs = tf.linspace(
            tf.cast(x_min, tf.float32),
            tf.cast(x_max, tf.float32),
            w_bev,
        )
        ys = tf.linspace(
            tf.cast(y_min, tf.float32),
            tf.cast(y_max, tf.float32),
            h_bev,
        )

        grid_x, grid_y = tf.meshgrid(xs, ys)  # (H, W)
        grid_z = tf.zeros_like(grid_x)  # Assume z=0 for BEV

        positions = tf.stack([grid_x, grid_y, grid_z], axis=-1)  # (H, W, 3)
        positions_flat = tf.reshape(positions, [1, h_bev * w_bev, 3])
        positions_flat = tf.repeat(positions_flat, batch_size, axis=0)
        return positions_flat

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """
        Full forward pass.

        Args:
            inputs: dict containing:
                - images: (B, num_cameras, H, W, 3)
                - radar_pillars: (B, max_pillars, max_pts_per_pillar, D_in)
                - radar_pillar_mask: (B, max_pillars, max_pts_per_pillar)
                - radar_pillar_coords: (B, max_pillars, 2) grid indices
                - lidar_to_cam: (B, num_cameras, 4, 4)
                - cam_intrinsics: (B, num_cameras, 3, 3)
        Returns:
            dict with detection outputs:
                - heatmap: (B, H_det, W_det, num_classes)
                - regression: (B, H_det, W_det, num_reg_attrs)
                - velocity: (B, H_det, W_det, 2)
                - height: (B, H_det, W_det, 2)
        """
        images = inputs["images"]
        radar_pillars = inputs["radar_pillars"]
        radar_pillar_mask = inputs["radar_pillar_mask"]
        radar_pillar_coords = inputs["radar_pillar_coords"]
        lidar_to_cam = inputs["lidar_to_cam"]
        cam_intrinsics = inputs["cam_intrinsics"]

        # 1. Camera branch: extract multi-view features with FPN
        camera_fpn_features = self.camera_backbone(images, training=training)

        # 2. Radar branch: pillar encoding + BEV backbone
        radar_bev_features = self.radar_branch(
            radar_pillars, radar_pillar_mask, radar_pillar_coords, training=training
        )

        # 3. Generate BEV positions for fusion
        b = tf.shape(radar_bev_features)[0]
        h_bev = tf.shape(radar_bev_features)[1]
        w_bev = tf.shape(radar_bev_features)[2]
        bev_positions = self._generate_bev_positions(b, h_bev, w_bev)

        # 4. Spatio-Contextual Fusion Transformer
        fused_bev = self.fusion_transformer(
            radar_bev_features,
            camera_fpn_features,
            bev_positions,
            lidar_to_cam,
            cam_intrinsics,
            training=training,
        )

        # 5. Detection head
        detections = self.detection_head(fused_bev, training=training)
        return detections

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["config"] = self.config
        return config


# ===========================================================================
# Helper: build model from config
# ===========================================================================


def build_craft_model(config: Optional[Dict[str, Any]] = None) -> CRAFTModel:
    """Factory function to build and optionally compile the CRAFT model."""
    model = CRAFTModel(config=config, name="craft")
    return model
