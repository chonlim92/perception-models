"""
RadarPillarNet: PointPillars architecture adapted for radar 3D object detection.

TensorFlow 2.x / Keras implementation.

Components:
- PillarEncoder: PointNet-style per-pillar feature extraction
- PillarScatter: scatter encoded pillars to BEV pseudo-image
- BEVBackbone: multi-scale 2D CNN with FPN-style fusion
- AnchorHead: classification + box regression + velocity + direction heads
- RadarPillarNet: end-to-end model combining all components

Input features per point (9 total):
    x, y, z         - 3D position
    RCS             - radar cross section
    vr              - radial velocity
    dt              - time delta (from multi-sweep accumulation)
    x_c, y_c, z_c  - offset from pillar center
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # Pillar parameters
    "pillar_x_size": 0.4,
    "pillar_y_size": 0.4,
    "pillar_z_size": 8.0,
    "x_min": -51.2,
    "x_max": 51.2,
    "y_min": -51.2,
    "y_max": 51.2,
    "z_min": -5.0,
    "z_max": 3.0,
    "max_points_per_pillar": 32,
    "max_pillars": 16000,
    "input_features_dim": 9,  # x, y, z, RCS, vr, dt, x_c, y_c, z_c
    "pillar_feat_dim": 64,
    # Grid dimensions (derived: (x_max - x_min) / pillar_x_size = 256)
    "grid_x": 256,
    "grid_y": 256,
    # BEV backbone
    "bev_backbone_channels": [64, 128, 256],
    "bev_backbone_num_convs": [3, 5, 5],
    "bev_backbone_strides": [2, 2, 2],
    # FPN deconvolution
    "fpn_upsample_strides": [1, 2, 4],
    "fpn_out_channels": 256,
    # Anchor head
    "num_classes": 10,
    "num_anchors_per_location": 2,  # 0 and 90 degree orientations
    "box_code_size": 7,  # dx, dy, dz, w, l, h, yaw
    "velocity_dim": 2,  # vx, vy
    "num_dir_bins": 2,  # direction classification bins
    # NMS
    "nms_score_threshold": 0.1,
    "nms_iou_threshold": 0.2,
    "max_detections": 500,
}


# ===========================================================================
# PillarEncoder: PointNet-style per-pillar feature extraction
# ===========================================================================


class PillarEncoder(layers.Layer):
    """
    PointNet-based pillar feature encoder.

    For each pillar, applies a shared MLP (Dense -> BN -> ReLU) to all points,
    then max-pools across points to produce a single feature vector per pillar.

    Input features per point (9-dim):
        x, y, z, RCS, vr, dt, x_c, y_c, z_c
    where x_c, y_c, z_c are offsets from pillar center.
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.feat_dim = config.get("pillar_feat_dim", 64)
        self.input_dim = config.get("input_features_dim", 9)

    def build(self, input_shape: tf.TensorShape) -> None:
        # PointNet: Linear(64) -> BN -> ReLU -> MaxPool
        self.linear = layers.Dense(self.feat_dim, use_bias=False, name="pointnet_fc")
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name="pointnet_bn")
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
                D_in = 9: x, y, z, RCS, vr, dt, x_c, y_c, z_c
            pillar_mask: (B, max_pillars, max_points_per_pillar) float32 mask,
                1.0 for valid points, 0.0 for padding
        Returns:
            pillar_encodings: (B, max_pillars, feat_dim)
        """
        # Shared MLP: Dense -> BN -> ReLU
        x = self.linear(pillar_features)  # (B, P, N, 64)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        # Mask out invalid points before max pooling (set to -inf)
        mask_expanded = tf.expand_dims(pillar_mask, axis=-1)  # (B, P, N, 1)
        x = x * mask_expanded + (1.0 - mask_expanded) * (-1e9)

        # Max pool across points dimension
        pillar_encodings = tf.reduce_max(x, axis=2)  # (B, P, 64)
        return pillar_encodings

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["config"] = self.config
        return config


# ===========================================================================
# PillarScatter: scatter pillar features to BEV pseudo-image
# ===========================================================================


class PillarScatter(layers.Layer):
    """
    Scatter pillar encodings into a 2D BEV pseudo-image grid.

    Each pillar's feature vector is placed at its corresponding (x, y) grid cell.
    Empty cells remain zero-filled.
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.grid_x = config.get("grid_x", 256)
        self.grid_y = config.get("grid_y", 256)
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
                where ix in [0, grid_x), iy in [0, grid_y)
        Returns:
            bev_image: (B, grid_y, grid_x, feat_dim)
        """
        batch_size = tf.shape(pillar_encodings)[0]
        max_pillars = tf.shape(pillar_encodings)[1]

        # Create batch indices
        batch_idx = tf.repeat(
            tf.range(batch_size)[:, tf.newaxis], max_pillars, axis=1
        )  # (B, max_pillars)

        ix = pillar_coords[:, :, 0]  # (B, max_pillars) - x grid index
        iy = pillar_coords[:, :, 1]  # (B, max_pillars) - y grid index

        # Clip to valid grid range
        ix = tf.clip_by_value(ix, 0, self.grid_x - 1)
        iy = tf.clip_by_value(iy, 0, self.grid_y - 1)

        # Build scatter indices: (B*max_pillars, 3)
        indices = tf.stack(
            [
                tf.reshape(batch_idx, [-1]),
                tf.reshape(iy, [-1]),
                tf.reshape(ix, [-1]),
            ],
            axis=1,
        )

        updates = tf.reshape(pillar_encodings, [-1, self.feat_dim])

        # Scatter to BEV grid
        bev_image = tf.scatter_nd(
            indices,
            updates,
            shape=[batch_size, self.grid_y, self.grid_x, self.feat_dim],
        )
        return bev_image

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["config"] = self.config
        return config


# ===========================================================================
# BEVBackbone: multi-scale 2D convolutions with FPN fusion
# ===========================================================================


class BEVBackbone(layers.Layer):
    """
    Multi-scale 2D convolutional backbone operating on the BEV pseudo-image.

    Architecture:
    - 3 blocks with increasing channels: 64 -> 128 -> 256
    - Each block has multiple 3x3 convolutions with BN + ReLU
    - First conv in each block uses stride=2 for downsampling
    - FPN-style deconvolution for multi-scale feature fusion
    - Final 1x1 convolution to produce unified BEV features
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.channels = config.get("bev_backbone_channels", [64, 128, 256])
        self.num_convs = config.get("bev_backbone_num_convs", [3, 5, 5])
        self.strides = config.get("bev_backbone_strides", [2, 2, 2])
        self.upsample_strides = config.get("fpn_upsample_strides", [1, 2, 4])
        self.fpn_out_channels = config.get("fpn_out_channels", 256)

    def build(self, input_shape: tf.TensorShape) -> None:
        # Downsampling blocks
        self.blocks = []
        for block_idx, (ch, n_conv, stride) in enumerate(
            zip(self.channels, self.num_convs, self.strides)
        ):
            block_layers = []
            for conv_idx in range(n_conv):
                s = stride if conv_idx == 0 else 1
                block_layers.append(
                    layers.Conv2D(
                        ch, 3, strides=s, padding="same", use_bias=False,
                        name=f"block{block_idx}_conv{conv_idx}",
                    )
                )
                block_layers.append(
                    layers.BatchNormalization(
                        epsilon=1e-3, momentum=0.99,
                        name=f"block{block_idx}_bn{conv_idx}",
                    )
                )
                block_layers.append(layers.ReLU())
            block = tf.keras.Sequential(block_layers, name=f"bev_block_{block_idx}")
            self.blocks.append(block)

        # FPN deconvolution (upsample) blocks
        self.deblocks = []
        for i, (ch, up_stride) in enumerate(zip(self.channels, self.upsample_strides)):
            if up_stride == 1:
                deblock = tf.keras.Sequential([
                    layers.Conv2D(
                        self.fpn_out_channels, 1, strides=1, padding="same",
                        use_bias=False, name=f"deconv_{i}",
                    ),
                    layers.BatchNormalization(
                        epsilon=1e-3, momentum=0.99, name=f"debn_{i}",
                    ),
                    layers.ReLU(),
                ], name=f"bev_deblock_{i}")
            else:
                deblock = tf.keras.Sequential([
                    layers.Conv2DTranspose(
                        self.fpn_out_channels, up_stride, strides=up_stride,
                        padding="same", use_bias=False, name=f"deconv_{i}",
                    ),
                    layers.BatchNormalization(
                        epsilon=1e-3, momentum=0.99, name=f"debn_{i}",
                    ),
                    layers.ReLU(),
                ], name=f"bev_deblock_{i}")
            self.deblocks.append(deblock)

        super().build(input_shape)

    def call(self, bev_image: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            bev_image: (B, H, W, C_in) BEV pseudo-image from PillarScatter
        Returns:
            bev_features: (B, H/2, W/2, fpn_out_channels * num_blocks)
                Multi-scale features concatenated along channel dim
        """
        block_outputs = []
        x = bev_image
        for block in self.blocks:
            x = block(x, training=training)
            block_outputs.append(x)

        # FPN: upsample each block output to the same spatial size and concatenate
        upsampled = []
        for deblock, feat in zip(self.deblocks, block_outputs):
            upsampled.append(deblock(feat, training=training))

        # All upsampled features should have the same spatial dimensions
        # Concatenate along channel dimension
        bev_features = tf.concat(upsampled, axis=-1)  # (B, H', W', fpn_out * 3)
        return bev_features

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["config"] = self.config
        return config


# ===========================================================================
# AnchorHead: detection head with multi-task outputs
# ===========================================================================


class AnchorHead(layers.Layer):
    """
    Anchor-based detection head for RadarPillarNet.

    Produces per-anchor predictions:
    - Classification: num_classes per anchor
    - Box regression: (dx, dy, dz, log(w), log(l), log(h), yaw) per anchor
    - Velocity regression: (vx, vy) per anchor
    - Direction classification: 2 bins per anchor (forward/backward)

    Uses 2 anchors per BEV location (0 and 90 degree orientations).
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.num_classes = config.get("num_classes", 10)
        self.num_anchors = config.get("num_anchors_per_location", 2)
        self.box_code_size = config.get("box_code_size", 7)
        self.velocity_dim = config.get("velocity_dim", 2)
        self.num_dir_bins = config.get("num_dir_bins", 2)

    def build(self, input_shape: tf.TensorShape) -> None:
        in_channels = input_shape[-1]

        # Shared convolution stem
        self.shared_conv = tf.keras.Sequential([
            layers.Conv2D(
                in_channels, 3, padding="same", use_bias=False, name="shared_conv1",
            ),
            layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name="shared_bn1"),
            layers.ReLU(),
        ], name="shared_head")

        # Classification head
        self.cls_head = layers.Conv2D(
            self.num_anchors * self.num_classes,
            1,
            padding="same",
            bias_initializer=tf.constant_initializer(-math.log((1 - 0.01) / 0.01)),
            name="cls_head",
        )

        # Box regression head
        self.box_head = layers.Conv2D(
            self.num_anchors * self.box_code_size,
            1,
            padding="same",
            name="box_head",
        )

        # Velocity regression head
        self.vel_head = layers.Conv2D(
            self.num_anchors * self.velocity_dim,
            1,
            padding="same",
            name="vel_head",
        )

        # Direction classification head
        self.dir_head = layers.Conv2D(
            self.num_anchors * self.num_dir_bins,
            1,
            padding="same",
            name="dir_head",
        )

        super().build(input_shape)

    def call(self, bev_features: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
        """
        Args:
            bev_features: (B, H, W, C) from BEVBackbone
        Returns:
            dict with:
                cls_preds: (B, H, W, num_anchors, num_classes)
                box_preds: (B, H, W, num_anchors, box_code_size)
                vel_preds: (B, H, W, num_anchors, velocity_dim)
                dir_preds: (B, H, W, num_anchors, num_dir_bins)
        """
        batch_size = tf.shape(bev_features)[0]
        h = tf.shape(bev_features)[1]
        w = tf.shape(bev_features)[2]

        x = self.shared_conv(bev_features, training=training)

        # Classification
        cls_preds = self.cls_head(x)  # (B, H, W, A*C)
        cls_preds = tf.reshape(cls_preds, [batch_size, h, w, self.num_anchors, self.num_classes])

        # Box regression
        box_preds = self.box_head(x)  # (B, H, W, A*7)
        box_preds = tf.reshape(box_preds, [batch_size, h, w, self.num_anchors, self.box_code_size])

        # Velocity
        vel_preds = self.vel_head(x)  # (B, H, W, A*2)
        vel_preds = tf.reshape(vel_preds, [batch_size, h, w, self.num_anchors, self.velocity_dim])

        # Direction classification
        dir_preds = self.dir_head(x)  # (B, H, W, A*2)
        dir_preds = tf.reshape(dir_preds, [batch_size, h, w, self.num_anchors, self.num_dir_bins])

        return {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "vel_preds": vel_preds,
            "dir_preds": dir_preds,
        }

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["config"] = self.config
        return config


# ===========================================================================
# RadarPillarNet: full end-to-end model
# ===========================================================================


class RadarPillarNet(tf.keras.Model):
    """
    RadarPillarNet: PointPillars architecture adapted for radar 3D object detection.

    End-to-end model:
        PillarEncoder -> PillarScatter -> BEVBackbone -> AnchorHead

    Supports:
    - Training with anchor-based target assignment
    - Inference with NMS-based post-processing
    - TF-TRT optimization via SavedModel export
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Build sub-components
        self.pillar_encoder = PillarEncoder(self.config, name="pillar_encoder")
        self.pillar_scatter = PillarScatter(self.config, name="pillar_scatter")
        self.bev_backbone = BEVBackbone(self.config, name="bev_backbone")
        self.anchor_head = AnchorHead(self.config, name="anchor_head")

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """
        Forward pass through the full RadarPillarNet.

        Args:
            inputs: dict containing:
                - pillar_features: (B, max_pillars, max_points_per_pillar, 9)
                - pillar_mask: (B, max_pillars, max_points_per_pillar) float mask
                - pillar_coords: (B, max_pillars, 2) int32 grid coords
            training: whether in training mode (affects BN, dropout)
        Returns:
            dict with detection head outputs:
                - cls_preds: (B, H, W, num_anchors, num_classes)
                - box_preds: (B, H, W, num_anchors, box_code_size)
                - vel_preds: (B, H, W, num_anchors, velocity_dim)
                - dir_preds: (B, H, W, num_anchors, num_dir_bins)
        """
        pillar_features = inputs["pillar_features"]
        pillar_mask = inputs["pillar_mask"]
        pillar_coords = inputs["pillar_coords"]

        # 1. Encode pillars with PointNet
        pillar_encodings = self.pillar_encoder(
            pillar_features, pillar_mask, training=training
        )  # (B, P, 64)

        # 2. Scatter to BEV pseudo-image
        bev_image = self.pillar_scatter(
            pillar_encodings, pillar_coords, training=training
        )  # (B, 256, 256, 64)

        # 3. BEV backbone with multi-scale fusion
        bev_features = self.bev_backbone(
            bev_image, training=training
        )  # (B, H', W', C)

        # 4. Detection head
        predictions = self.anchor_head(
            bev_features, training=training
        )

        return predictions

    @tf.function(input_signature=[{
        "pillar_features": tf.TensorSpec([None, None, None, 9], tf.float32),
        "pillar_mask": tf.TensorSpec([None, None, None], tf.float32),
        "pillar_coords": tf.TensorSpec([None, None, 2], tf.int32),
    }])
    def predict_with_nms(
        self,
        inputs: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        """
        Run inference with NMS post-processing.

        This method is decorated with @tf.function for SavedModel export.

        Args:
            inputs: same as call()
        Returns:
            dict with:
                - boxes: (B, max_det, 7) decoded boxes [x, y, z, w, l, h, yaw]
                - scores: (B, max_det) confidence scores
                - labels: (B, max_det) class indices
                - velocities: (B, max_det, 2) predicted velocities
                - num_detections: (B,) number of valid detections per sample
        """
        # Forward pass
        predictions = self(inputs, training=False)

        cls_preds = predictions["cls_preds"]   # (B, H, W, A, C)
        box_preds = predictions["box_preds"]   # (B, H, W, A, 7)
        vel_preds = predictions["vel_preds"]   # (B, H, W, A, 2)
        dir_preds = predictions["dir_preds"]   # (B, H, W, A, 2)

        batch_size = tf.shape(cls_preds)[0]
        h = tf.shape(cls_preds)[1]
        w = tf.shape(cls_preds)[2]
        num_anchors = self.config["num_anchors_per_location"]
        num_classes = self.config["num_classes"]
        max_det = self.config.get("max_detections", 500)

        # Generate anchor grid
        anchors = self._generate_anchors(h, w)  # (H, W, A, 7)

        # Flatten spatial dimensions
        cls_flat = tf.reshape(cls_preds, [batch_size, -1, num_classes])  # (B, H*W*A, C)
        box_flat = tf.reshape(box_preds, [batch_size, -1, 7])
        vel_flat = tf.reshape(vel_preds, [batch_size, -1, 2])
        dir_flat = tf.reshape(dir_preds, [batch_size, -1, 2])
        anchors_flat = tf.reshape(anchors, [1, -1, 7])
        anchors_flat = tf.repeat(anchors_flat, batch_size, axis=0)

        # Decode box predictions to world coordinates
        decoded_boxes = self._decode_boxes(box_flat, anchors_flat)  # (B, N, 7)

        # Apply direction classification to correct yaw
        dir_labels = tf.argmax(dir_flat, axis=-1)  # (B, N)
        # Flip yaw by pi if direction bin is 1
        yaw_correction = tf.cast(dir_labels, tf.float32) * math.pi
        decoded_yaw = decoded_boxes[:, :, 6] + yaw_correction
        # Normalize yaw to [-pi, pi]
        decoded_yaw = tf.math.atan2(tf.sin(decoded_yaw), tf.cos(decoded_yaw))
        decoded_boxes = tf.concat([decoded_boxes[:, :, :6], decoded_yaw[:, :, tf.newaxis]], axis=-1)

        # Classification scores (sigmoid)
        cls_scores = tf.sigmoid(cls_flat)  # (B, N, C)

        # Per-class NMS
        # Get max class score and label for each anchor
        max_scores = tf.reduce_max(cls_scores, axis=-1)  # (B, N)
        max_labels = tf.argmax(cls_scores, axis=-1, output_type=tf.int32)  # (B, N)

        # Apply NMS per batch element
        all_boxes = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=False)
        all_scores = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=False)
        all_labels = tf.TensorArray(dtype=tf.int32, size=batch_size, dynamic_size=False)
        all_vels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=False)
        all_num_det = tf.TensorArray(dtype=tf.int32, size=batch_size, dynamic_size=False)

        score_thresh = self.config.get("nms_score_threshold", 0.1)
        iou_thresh = self.config.get("nms_iou_threshold", 0.2)

        for b in tf.range(batch_size):
            b_scores = max_scores[b]  # (N,)
            b_labels = max_labels[b]
            b_boxes = decoded_boxes[b]  # (N, 7)
            b_vels = vel_flat[b]  # (N, 2)

            # Filter by score threshold
            valid_mask = b_scores > score_thresh
            valid_indices = tf.where(valid_mask)[:, 0]

            b_scores_valid = tf.gather(b_scores, valid_indices)
            b_labels_valid = tf.gather(b_labels, valid_indices)
            b_boxes_valid = tf.gather(b_boxes, valid_indices)
            b_vels_valid = tf.gather(b_vels, valid_indices)

            # Convert to 2D BEV boxes for NMS: [y1, x1, y2, x2]
            cx = b_boxes_valid[:, 0]
            cy = b_boxes_valid[:, 1]
            bw = b_boxes_valid[:, 3]
            bl = b_boxes_valid[:, 4]
            bev_boxes_2d = tf.stack([
                cy - bl / 2, cx - bw / 2,
                cy + bl / 2, cx + bw / 2,
            ], axis=-1)

            # NMS
            nms_indices = tf.image.non_max_suppression(
                bev_boxes_2d,
                b_scores_valid,
                max_output_size=max_det,
                iou_threshold=iou_thresh,
                score_threshold=score_thresh,
            )

            n_det = tf.shape(nms_indices)[0]

            # Gather NMS results
            nms_boxes = tf.gather(b_boxes_valid, nms_indices)
            nms_scores = tf.gather(b_scores_valid, nms_indices)
            nms_labels = tf.gather(b_labels_valid, nms_indices)
            nms_vels = tf.gather(b_vels_valid, nms_indices)

            # Pad to max_det
            pad_size = max_det - n_det
            nms_boxes = tf.pad(nms_boxes, [[0, pad_size], [0, 0]])
            nms_scores = tf.pad(nms_scores, [[0, pad_size]])
            nms_labels = tf.pad(nms_labels, [[0, pad_size]])
            nms_vels = tf.pad(nms_vels, [[0, pad_size], [0, 0]])

            all_boxes = all_boxes.write(b, nms_boxes)
            all_scores = all_scores.write(b, nms_scores)
            all_labels = all_labels.write(b, nms_labels)
            all_vels = all_vels.write(b, nms_vels)
            all_num_det = all_num_det.write(b, n_det)

        return {
            "boxes": all_boxes.stack(),        # (B, max_det, 7)
            "scores": all_scores.stack(),      # (B, max_det)
            "labels": all_labels.stack(),      # (B, max_det)
            "velocities": all_vels.stack(),    # (B, max_det, 2)
            "num_detections": all_num_det.stack(),  # (B,)
        }

    def _generate_anchors(self, h: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
        """
        Generate anchor boxes for the BEV feature map.

        Returns:
            anchors: (H, W, num_anchors, 7)
                Each anchor: [x, y, z, w, l, h, yaw]
        """
        cfg = self.config
        x_min = cfg["x_min"]
        y_min = cfg["y_min"]
        x_max = cfg["x_max"]
        y_max = cfg["y_max"]

        # Compute BEV resolution at feature map level
        # Feature map is downsampled by first stride
        x_res = (x_max - x_min) / tf.cast(w, tf.float32)
        y_res = (y_max - y_min) / tf.cast(h, tf.float32)

        # Grid centers
        xs = tf.cast(tf.range(w), tf.float32) * x_res + x_min + x_res / 2.0
        ys = tf.cast(tf.range(h), tf.float32) * y_res + y_min + y_res / 2.0
        grid_x, grid_y = tf.meshgrid(xs, ys)  # (H, W)

        # Anchor parameters (typical for nuScenes radar detection)
        # Anchor 1: 0 degree, Anchor 2: 90 degree
        anchor_dims = tf.constant([
            [4.7, 2.0, 1.7],  # w, l, h (typical car)
            [4.7, 2.0, 1.7],
        ], dtype=tf.float32)  # (A, 3)

        anchor_yaws = tf.constant([0.0, math.pi / 2.0], dtype=tf.float32)  # (A,)
        anchor_z = tf.constant(-1.0, dtype=tf.float32)  # typical z center

        num_anchors = cfg["num_anchors_per_location"]

        # Expand grid to (H, W, A, 7)
        grid_x_exp = tf.tile(grid_x[:, :, tf.newaxis], [1, 1, num_anchors])
        grid_y_exp = tf.tile(grid_y[:, :, tf.newaxis], [1, 1, num_anchors])
        grid_z_exp = tf.fill([h, w, num_anchors], anchor_z)

        # Tile anchor dims: (A, 3) -> (H, W, A, 3)
        dims_exp = tf.tile(anchor_dims[tf.newaxis, tf.newaxis, :, :], [h, w, 1, 1])
        yaws_exp = tf.tile(anchor_yaws[tf.newaxis, tf.newaxis, :], [h, w, 1])

        anchors = tf.stack([
            grid_x_exp, grid_y_exp, grid_z_exp,
            dims_exp[:, :, :, 0], dims_exp[:, :, :, 1], dims_exp[:, :, :, 2],
            yaws_exp,
        ], axis=-1)  # (H, W, A, 7)

        return anchors

    def _decode_boxes(
        self,
        box_preds: tf.Tensor,
        anchors: tf.Tensor,
    ) -> tf.Tensor:
        """
        Decode box predictions relative to anchors.

        Args:
            box_preds: (B, N, 7) predicted box residuals
                [dx, dy, dz, log(dw), log(dl), log(dh), dyaw]
            anchors: (B, N, 7) anchor parameters
                [xa, ya, za, wa, la, ha, yaw_a]
        Returns:
            decoded: (B, N, 7) decoded boxes in world frame
                [x, y, z, w, l, h, yaw]
        """
        xa, ya, za = anchors[:, :, 0], anchors[:, :, 1], anchors[:, :, 2]
        wa, la, ha = anchors[:, :, 3], anchors[:, :, 4], anchors[:, :, 5]
        yaw_a = anchors[:, :, 6]

        # Anchor diagonal (for normalizing position residuals)
        diag = tf.sqrt(wa ** 2 + la ** 2)

        dx = box_preds[:, :, 0]
        dy = box_preds[:, :, 1]
        dz = box_preds[:, :, 2]
        dw = box_preds[:, :, 3]
        dl = box_preds[:, :, 4]
        dh = box_preds[:, :, 5]
        dyaw = box_preds[:, :, 6]

        # Decode position
        x = xa + dx * diag
        y = ya + dy * diag
        z = za + dz * ha

        # Decode size (exp of log-residual)
        w = wa * tf.exp(dw)
        l = la * tf.exp(dl)
        h = ha * tf.exp(dh)

        # Decode yaw
        yaw = yaw_a + dyaw

        decoded = tf.stack([x, y, z, w, l, h, yaw], axis=-1)
        return decoded

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["config"] = self.config
        return config


# ===========================================================================
# Factory function
# ===========================================================================


def build_radar_pillarnet(config: Optional[Dict[str, Any]] = None) -> RadarPillarNet:
    """
    Factory function to build RadarPillarNet model.

    Args:
        config: model configuration overrides (merged with DEFAULT_CONFIG)
    Returns:
        Constructed RadarPillarNet model
    """
    model = RadarPillarNet(config=config, name="radar_pillarnet")
    return model
