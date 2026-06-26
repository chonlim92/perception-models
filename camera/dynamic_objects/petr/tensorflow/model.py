"""
PETR / StreamPETR model implementation in TensorFlow 2 / Keras.

Implements:
  - ResNet50 backbone with FPN neck
  - 3D Position Embedding via camera frustum -> world coords -> MLP
  - Standard Transformer decoder with cross-attention
  - Detection head (classification + 3D bbox regression with velocity)
  - Optional temporal query propagation (StreamPETR mode)

Reference papers:
  - PETR: Position Embedding Transformation for Multi-View 3D Object Detection (ECCV 2022)
  - PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images (ICCV 2023)
  - StreamPETR: Exploring Object-Centric Temporal Modeling for 3D Object Detection (ICCV 2023)
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple


class FPN(tf.keras.layers.Layer):
    """Feature Pyramid Network neck for multi-scale feature extraction."""

    def __init__(self, out_channels: int = 256, num_outs: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.num_outs = num_outs

    def build(self, input_shape):
        num_inputs = len(input_shape)
        self.lateral_convs = []
        self.fpn_convs = []
        for i in range(num_inputs):
            lateral = tf.keras.layers.Conv2D(
                self.out_channels, 1, padding="same", name=f"lateral_conv_{i}"
            )
            fpn = tf.keras.layers.Conv2D(
                self.out_channels, 3, padding="same", name=f"fpn_conv_{i}"
            )
            self.lateral_convs.append(lateral)
            self.fpn_convs.append(fpn)

        extra_levels = self.num_outs - num_inputs
        self.extra_convs = []
        for i in range(extra_levels):
            extra = tf.keras.layers.Conv2D(
                self.out_channels, 3, strides=2, padding="same", name=f"extra_conv_{i}"
            )
            self.extra_convs.append(extra)

    def call(self, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        for i in range(len(laterals) - 2, -1, -1):
            h, w = tf.shape(laterals[i])[1], tf.shape(laterals[i])[2]
            upsampled = tf.image.resize(laterals[i + 1], [h, w], method="bilinear")
            laterals[i] = laterals[i] + upsampled

        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        if self.extra_convs:
            extra_input = outs[-1]
            for extra_conv in self.extra_convs:
                extra_input = tf.nn.relu(extra_conv(extra_input))
                outs.append(extra_input)

        return outs[: self.num_outs]


class PositionEmbedding3D(tf.keras.layers.Layer):
    """
    3D Position Embedding for PETR.

    Generates a camera frustum of depth/width/height bins, projects them to 3D
    world coordinates using camera intrinsics and extrinsics, then encodes with MLP.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_depth_bins: int = 64,
        depth_range: Tuple[float, float] = (1.0, 61.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_depth_bins = num_depth_bins
        self.depth_start = depth_range[0]
        self.depth_end = depth_range[1]

    def build(self, input_shape):
        self.position_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.embed_dims, activation="relu", name="pe_fc1"),
                tf.keras.layers.Dense(self.embed_dims, activation="relu", name="pe_fc2"),
                tf.keras.layers.Dense(self.embed_dims, name="pe_fc3"),
            ],
            name="position_mlp",
        )

    def _create_frustum(self, height: int, width: int) -> tf.Tensor:
        """Create camera frustum grid of shape (D, H, W, 3) with (u, v, d) coords."""
        depth_bins = tf.linspace(self.depth_start, self.depth_end, self.num_depth_bins)

        u = tf.linspace(0.5, tf.cast(width, tf.float32) - 0.5, width)
        v = tf.linspace(0.5, tf.cast(height, tf.float32) - 0.5, height)

        d_grid, v_grid, u_grid = tf.meshgrid(depth_bins, v, u, indexing="ij")

        frustum = tf.stack([u_grid, v_grid, d_grid], axis=-1)
        return frustum

    def _frustum_to_3d(
        self,
        frustum: tf.Tensor,
        intrinsics: tf.Tensor,
        extrinsics: tf.Tensor,
    ) -> tf.Tensor:
        """
        Transform frustum points from image coords to 3D world coords.

        Args:
            frustum: (D, H, W, 3) frustum points in (u, v, d)
            intrinsics: (B, N, 3, 3) camera intrinsic matrices
            extrinsics: (B, N, 4, 4) camera-to-world transformation matrices

        Returns:
            coords_3d: (B, N, D*H*W, 3) 3D world coordinates
        """
        D, H, W, _ = frustum.shape
        num_points = D * H * W

        points = tf.reshape(frustum, [num_points, 3])
        u, v, d = points[:, 0], points[:, 1], points[:, 2]

        B = tf.shape(intrinsics)[0]
        N = tf.shape(intrinsics)[1]

        fx = intrinsics[:, :, 0, 0]
        fy = intrinsics[:, :, 1, 1]
        cx = intrinsics[:, :, 0, 2]
        cy = intrinsics[:, :, 1, 2]

        u_exp = tf.broadcast_to(u[None, None, :], [B, N, num_points])
        v_exp = tf.broadcast_to(v[None, None, :], [B, N, num_points])
        d_exp = tf.broadcast_to(d[None, None, :], [B, N, num_points])

        x_cam = (u_exp - cx[:, :, None]) * d_exp / fx[:, :, None]
        y_cam = (v_exp - cy[:, :, None]) * d_exp / fy[:, :, None]
        z_cam = d_exp

        points_cam = tf.stack([x_cam, y_cam, z_cam, tf.ones_like(z_cam)], axis=-1)

        points_cam_reshape = tf.reshape(points_cam, [B * N, num_points, 4])
        extrinsics_reshape = tf.reshape(extrinsics, [B * N, 4, 4])

        points_world = tf.matmul(points_cam_reshape, extrinsics_reshape, transpose_b=True)
        points_world = tf.reshape(points_world[:, :, :3], [B, N, num_points, 3])

        return points_world

    def call(
        self,
        feature_shape: Tuple[int, int],
        intrinsics: tf.Tensor,
        extrinsics: tf.Tensor,
    ) -> tf.Tensor:
        """
        Generate 3D position embeddings.

        Args:
            feature_shape: (H, W) spatial dimensions of the feature map
            intrinsics: (B, N, 3, 3) camera intrinsics
            extrinsics: (B, N, 4, 4) cam-to-world extrinsics

        Returns:
            pos_embed: (B, N, D*H*W, embed_dims)
        """
        H, W = feature_shape
        frustum = self._create_frustum(H, W)
        coords_3d = self._frustum_to_3d(frustum, intrinsics, extrinsics)

        x_min, x_max = -61.2, 61.2
        y_min, y_max = -61.2, 61.2
        z_min, z_max = -10.0, 10.0

        coords_normalized = tf.stack(
            [
                (coords_3d[..., 0] - x_min) / (x_max - x_min),
                (coords_3d[..., 1] - y_min) / (y_max - y_min),
                (coords_3d[..., 2] - z_min) / (z_max - z_min),
            ],
            axis=-1,
        )
        coords_normalized = tf.clip_by_value(coords_normalized, 0.0, 1.0)

        pos_embed = self.position_mlp(coords_normalized)
        return pos_embed


class MultiHeadCrossAttention(tf.keras.layers.Layer):
    """Standard multi-head cross-attention for the transformer decoder."""

    def __init__(self, embed_dims: int = 256, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.q_proj = tf.keras.layers.Dense(self.embed_dims, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(self.embed_dims, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.embed_dims, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(self.embed_dims, name="out_proj")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        query_pos: Optional[tf.Tensor] = None,
        key_pos: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Args:
            query: (B, Q, C) query embeddings
            key: (B, K, C) key embeddings
            value: (B, K, C) value embeddings
            query_pos: (B, Q, C) positional encoding for queries
            key_pos: (B, K, C) positional encoding for keys
        """
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        B = tf.shape(query)[0]
        Q = tf.shape(query)[1]
        K = tf.shape(key)[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = tf.reshape(q, [B, Q, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])

        k = tf.reshape(k, [B, K, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])

        v = tf.reshape(v, [B, K, self.num_heads, self.head_dim])
        v = tf.transpose(v, [0, 2, 1, 3])

        attn_weights = tf.matmul(q, k, transpose_b=True) * self.scale
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        attn_output = tf.matmul(attn_weights, v)
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [B, Q, self.embed_dims])

        output = self.out_proj(attn_output)
        return output


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """Standard multi-head self-attention."""

    def __init__(self, embed_dims: int = 256, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.q_proj = tf.keras.layers.Dense(self.embed_dims, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(self.embed_dims, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.embed_dims, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(self.embed_dims, name="out_proj")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(
        self,
        query: tf.Tensor,
        query_pos: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        if query_pos is not None:
            q_input = query + query_pos
        else:
            q_input = query

        B = tf.shape(query)[0]
        Q = tf.shape(query)[1]

        q = self.q_proj(q_input)
        k = self.k_proj(q_input)
        v = self.v_proj(query)

        q = tf.reshape(q, [B, Q, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])

        k = tf.reshape(k, [B, Q, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])

        v = tf.reshape(v, [B, Q, self.num_heads, self.head_dim])
        v = tf.transpose(v, [0, 2, 1, 3])

        attn_weights = tf.matmul(q, k, transpose_b=True) * self.scale
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        attn_output = tf.matmul(attn_weights, v)
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [B, Q, self.embed_dims])

        return self.out_proj(attn_output)


class MotionAwareLayerNorm(tf.keras.layers.Layer):
    """
    Motion-Aware Layer Normalization for StreamPETR.
    Modulates normalized features based on ego-motion information.
    """

    def __init__(self, embed_dims: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims

    def build(self, input_shape):
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln")
        self.motion_proj = tf.keras.layers.Dense(
            self.embed_dims * 2, name="motion_proj"
        )

    def call(self, x: tf.Tensor, motion: tf.Tensor) -> tf.Tensor:
        """
        Args:
            x: (B, N, C) input features
            motion: (B, C) motion embedding (from ego-motion)
        Returns:
            modulated: (B, N, C) motion-modulated features
        """
        normalized = self.norm(x)
        motion_params = self.motion_proj(motion)
        gamma, beta = tf.split(motion_params, 2, axis=-1)
        gamma = gamma[:, None, :] + 1.0
        beta = beta[:, None, :]
        return normalized * gamma + beta


class TransformerDecoderLayer(tf.keras.layers.Layer):
    """Single transformer decoder layer with self-attention, cross-attention, and FFN."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        ffn_dims: int = 1024,
        dropout: float = 0.1,
        use_motion_aware_ln: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.use_motion_aware_ln = use_motion_aware_ln

    def build(self, input_shape):
        self.self_attn = MultiHeadSelfAttention(
            self.embed_dims, name="self_attn"
        )
        self.cross_attn = MultiHeadCrossAttention(
            self.embed_dims, name="cross_attn"
        )

        if self.use_motion_aware_ln:
            self.norm1 = MotionAwareLayerNorm(self.embed_dims, name="norm1")
            self.norm2 = MotionAwareLayerNorm(self.embed_dims, name="norm2")
            self.norm3 = MotionAwareLayerNorm(self.embed_dims, name="norm3")
        else:
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm1")
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm2")
            self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm3")

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(1024, activation="relu", name="ffn_fc1"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(self.embed_dims, name="ffn_fc2"),
                tf.keras.layers.Dropout(0.1),
            ],
            name="ffn",
        )

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        query_pos: Optional[tf.Tensor] = None,
        key_pos: Optional[tf.Tensor] = None,
        motion_embed: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        if self.use_motion_aware_ln and motion_embed is not None:
            q = self.norm1(query, motion_embed)
        else:
            q = self.norm1(query) if not self.use_motion_aware_ln else self.norm1(query, tf.zeros([tf.shape(query)[0], self.embed_dims]))

        q = self.self_attn(q, query_pos=query_pos, training=training)
        query = query + q

        if self.use_motion_aware_ln and motion_embed is not None:
            q2 = self.norm2(query, motion_embed)
        else:
            q2 = self.norm2(query) if not self.use_motion_aware_ln else self.norm2(query, tf.zeros([tf.shape(query)[0], self.embed_dims]))

        q2 = self.cross_attn(
            q2, key, value, query_pos=query_pos, key_pos=key_pos, training=training
        )
        query = query + q2

        if self.use_motion_aware_ln and motion_embed is not None:
            q3 = self.norm3(query, motion_embed)
        else:
            q3 = self.norm3(query) if not self.use_motion_aware_ln else self.norm3(query, tf.zeros([tf.shape(query)[0], self.embed_dims]))

        q3 = self.ffn(q3, training=training)
        query = query + q3

        return query


class TransformerDecoder(tf.keras.layers.Layer):
    """Stack of transformer decoder layers."""

    def __init__(
        self,
        num_layers: int = 6,
        embed_dims: int = 256,
        num_heads: int = 8,
        ffn_dims: int = 1024,
        dropout: float = 0.1,
        use_motion_aware_ln: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.ffn_dims = ffn_dims
        self.dropout = dropout
        self.use_motion_aware_ln = use_motion_aware_ln

    def build(self, input_shape):
        self.layers_list = [
            TransformerDecoderLayer(
                embed_dims=self.embed_dims,
                num_heads=self.num_heads,
                ffn_dims=self.ffn_dims,
                dropout=self.dropout,
                use_motion_aware_ln=self.use_motion_aware_ln,
                name=f"decoder_layer_{i}",
            )
            for i in range(self.num_layers)
        ]

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        query_pos: Optional[tf.Tensor] = None,
        key_pos: Optional[tf.Tensor] = None,
        motion_embed: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> List[tf.Tensor]:
        """Returns list of outputs from each decoder layer for auxiliary losses."""
        intermediate = []
        output = query
        for layer in self.layers_list:
            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                motion_embed=motion_embed,
                training=training,
            )
            intermediate.append(output)
        return intermediate


class DetectionHead(tf.keras.layers.Layer):
    """
    Detection head for 3D object detection.
    Outputs classification scores and 3D bounding box regression (center, size, rotation, velocity).
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dims: int = 256,
        num_reg_layers: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_layers = num_reg_layers

    def build(self, input_shape):
        cls_layers = []
        for i in range(self.num_reg_layers):
            cls_layers.append(
                tf.keras.layers.Dense(self.embed_dims, activation="relu", name=f"cls_fc_{i}")
            )
        cls_layers.append(
            tf.keras.layers.Dense(self.num_classes, name="cls_output")
        )
        self.cls_branch = tf.keras.Sequential(cls_layers, name="cls_branch")

        reg_layers = []
        for i in range(self.num_reg_layers):
            reg_layers.append(
                tf.keras.layers.Dense(self.embed_dims, activation="relu", name=f"reg_fc_{i}")
            )
        reg_layers.append(
            tf.keras.layers.Dense(10, name="reg_output")
        )
        self.reg_branch = tf.keras.Sequential(reg_layers, name="reg_branch")

    def call(self, query_features: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            query_features: (B, Q, C) decoded query features

        Returns:
            cls_scores: (B, Q, num_classes) classification logits
            bbox_preds: (B, Q, 10) regression predictions
                        [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
        """
        cls_scores = self.cls_branch(query_features)
        bbox_preds = self.reg_branch(query_features)
        return cls_scores, bbox_preds


class PETR(tf.keras.Model):
    """
    PETR: Position Embedding Transformation for Multi-View 3D Object Detection.

    Supports:
      - PETR base (temporal=False)
      - StreamPETR with temporal query propagation (temporal=True)

    Config dict keys:
      - num_classes: int (default 10)
      - embed_dims: int (default 256)
      - num_queries: int (default 900)
      - num_decoder_layers: int (default 6)
      - num_heads: int (default 8)
      - ffn_dims: int (default 1024)
      - dropout: float (default 0.1)
      - num_depth_bins: int (default 64)
      - depth_range: tuple (default (1.0, 61.0))
      - temporal: bool (default False)
      - num_propagated_queries: int (default 256) -- for StreamPETR
      - backbone_output_layers: list of str (default ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'])
    """

    def __init__(self, config: Dict, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_classes = config.get("num_classes", 10)
        self.embed_dims = config.get("embed_dims", 256)
        self.num_queries = config.get("num_queries", 900)
        self.num_decoder_layers = config.get("num_decoder_layers", 6)
        self.num_heads = config.get("num_heads", 8)
        self.ffn_dims = config.get("ffn_dims", 1024)
        self.dropout = config.get("dropout", 0.1)
        self.num_depth_bins = config.get("num_depth_bins", 64)
        self.depth_range = config.get("depth_range", (1.0, 61.0))
        self.temporal = config.get("temporal", False)
        self.num_propagated = config.get("num_propagated_queries", 256)
        self.backbone_output_layers = config.get(
            "backbone_output_layers",
            ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"],
        )

        self._build_backbone()
        self._build_neck()
        self._build_pe()
        self._build_decoder()
        self._build_head()
        self._build_query_embedding()
        if self.temporal:
            self._build_temporal_components()

    def _build_backbone(self):
        base_model = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=(None, None, 3)
        )
        outputs = [base_model.get_layer(name).output for name in self.backbone_output_layers]
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=outputs, name="resnet50_backbone")

    def _build_neck(self):
        self.neck = FPN(out_channels=self.embed_dims, num_outs=1, name="fpn_neck")

    def _build_pe(self):
        self.position_embedding = PositionEmbedding3D(
            embed_dims=self.embed_dims,
            num_depth_bins=self.num_depth_bins,
            depth_range=self.depth_range,
            name="pos_embed_3d",
        )

    def _build_decoder(self):
        self.decoder = TransformerDecoder(
            num_layers=self.num_decoder_layers,
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            ffn_dims=self.ffn_dims,
            dropout=self.dropout,
            use_motion_aware_ln=self.temporal,
            name="transformer_decoder",
        )

    def _build_head(self):
        self.detection_head = DetectionHead(
            num_classes=self.num_classes,
            embed_dims=self.embed_dims,
            name="detection_head",
        )

    def _build_query_embedding(self):
        self.query_embedding = self.add_weight(
            name="query_embedding",
            shape=(self.num_queries, self.embed_dims),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.query_pos_embedding = self.add_weight(
            name="query_pos_embedding",
            shape=(self.num_queries, self.embed_dims),
            initializer="glorot_uniform",
            trainable=True,
        )

    def _build_temporal_components(self):
        """Build components specific to StreamPETR temporal modeling."""
        self.ego_motion_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.embed_dims, activation="relu", name="ego_fc1"),
                tf.keras.layers.Dense(self.embed_dims, name="ego_fc2"),
            ],
            name="ego_motion_mlp",
        )
        self.query_memory_proj = tf.keras.layers.Dense(
            self.embed_dims, name="query_memory_proj"
        )
        self.reference_point_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.embed_dims, activation="relu", name="ref_fc1"),
                tf.keras.layers.Dense(3, activation="sigmoid", name="ref_output"),
            ],
            name="reference_point_head",
        )

    def _extract_features(self, images: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Extract multi-view features.

        Args:
            images: (B, N, H, W, 3) multi-view images

        Returns:
            features: (B, N, H', W', C) extracted features
        """
        B = tf.shape(images)[0]
        N = tf.shape(images)[1]
        H = tf.shape(images)[2]
        W = tf.shape(images)[3]

        images_flat = tf.reshape(images, [B * N, H, W, 3])

        backbone_feats = self.backbone(images_flat, training=training)

        fpn_feats = self.neck(backbone_feats)
        feat = fpn_feats[0]

        fH = tf.shape(feat)[1]
        fW = tf.shape(feat)[2]
        C = tf.shape(feat)[3]

        feat = tf.reshape(feat, [B, N, fH, fW, C])
        return feat

    def _propagate_queries(
        self,
        prev_query: tf.Tensor,
        ego_motion_matrix: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Propagate queries from previous frame using ego-motion (StreamPETR).

        Args:
            prev_query: (B, Q_prev, C) queries from previous frame
            ego_motion_matrix: (B, 4, 4) ego-motion transformation

        Returns:
            propagated_query: (B, num_propagated, C)
            motion_embed: (B, C) motion embedding for layer norm modulation
        """
        ego_flat = tf.reshape(ego_motion_matrix[:, :3, :], [-1, 12])
        motion_embed = self.ego_motion_mlp(ego_flat)

        propagated = self.query_memory_proj(prev_query[:, : self.num_propagated, :])

        return propagated, motion_embed

    def call(
        self,
        images: tf.Tensor,
        intrinsics: tf.Tensor,
        extrinsics: tf.Tensor,
        ego_motion: Optional[tf.Tensor] = None,
        prev_query: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """
        Forward pass.

        Args:
            images: (B, N, H, W, 3) multi-view camera images (N=6 for nuScenes)
            intrinsics: (B, N, 3, 3) camera intrinsic matrices
            extrinsics: (B, N, 4, 4) camera-to-world extrinsic matrices
            ego_motion: (B, 4, 4) ego-motion matrix (only for temporal mode)
            prev_query: (B, Q, C) previous frame queries (only for temporal mode)
            training: bool

        Returns:
            dict with keys:
              - 'cls_scores': list of (B, Q, num_classes) per decoder layer
              - 'bbox_preds': list of (B, Q, 10) per decoder layer
              - 'query_output': (B, Q, C) final query features (for temporal propagation)
        """
        features = self._extract_features(images, training=training)

        B = tf.shape(features)[0]
        N = tf.shape(features)[1]
        fH = tf.shape(features)[2]
        fW = tf.shape(features)[3]
        C = tf.shape(features)[4]

        key_value = tf.reshape(features, [B, N * fH * fW, C])

        pos_embed = self.position_embedding((fH, fW), intrinsics, extrinsics)
        key_pos = tf.reshape(pos_embed, [B, -1, self.embed_dims])

        query = tf.broadcast_to(
            self.query_embedding[None, :, :], [B, self.num_queries, self.embed_dims]
        )
        query_pos = tf.broadcast_to(
            self.query_pos_embedding[None, :, :], [B, self.num_queries, self.embed_dims]
        )

        motion_embed = None
        if self.temporal and prev_query is not None and ego_motion is not None:
            propagated_query, motion_embed = self._propagate_queries(prev_query, ego_motion)
            query = tf.concat([propagated_query, query[:, self.num_propagated:, :]], axis=1)

        decoder_outputs = self.decoder(
            query=query,
            key=key_value,
            value=key_value,
            query_pos=query_pos,
            key_pos=key_pos,
            motion_embed=motion_embed,
            training=training,
        )

        all_cls_scores = []
        all_bbox_preds = []
        for layer_output in decoder_outputs:
            cls_scores, bbox_preds = self.detection_head(layer_output)
            all_cls_scores.append(cls_scores)
            all_bbox_preds.append(bbox_preds)

        return {
            "cls_scores": all_cls_scores,
            "bbox_preds": all_bbox_preds,
            "query_output": decoder_outputs[-1],
        }


class HungarianMatcher:
    """
    Hungarian matcher for bipartite matching between predictions and ground truth.
    Uses cost based on classification, L1 bbox distance, and optionally IoU.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def match(
        self,
        cls_scores: tf.Tensor,
        bbox_preds: tf.Tensor,
        gt_labels: tf.Tensor,
        gt_bboxes: tf.Tensor,
    ) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        """
        Perform Hungarian matching for a batch.

        Args:
            cls_scores: (B, Q, num_classes) predicted classification logits
            bbox_preds: (B, Q, 10) predicted bounding boxes
            gt_labels: (B, max_gt) ground truth class labels (padded with -1)
            gt_bboxes: (B, max_gt, 10) ground truth bounding boxes

        Returns:
            List of (pred_indices, gt_indices) tuples per batch element
        """
        B = cls_scores.shape[0]
        matches = []

        for b in range(B):
            valid_mask = gt_labels[b] >= 0
            valid_gt_labels = tf.boolean_mask(gt_labels[b], valid_mask)
            valid_gt_bboxes = tf.boolean_mask(gt_bboxes[b], valid_mask)

            num_gt = tf.shape(valid_gt_labels)[0]
            if num_gt == 0:
                matches.append(
                    (tf.zeros([0], dtype=tf.int32), tf.zeros([0], dtype=tf.int32))
                )
                continue

            cls_prob = tf.nn.softmax(cls_scores[b], axis=-1)
            cls_cost = -tf.gather(cls_prob, valid_gt_labels, axis=1)

            bbox_cost = tf.reduce_sum(
                tf.abs(bbox_preds[b][:, None, :] - valid_gt_bboxes[None, :, :]),
                axis=-1,
            )

            cost_matrix = (
                self.cost_class * cls_cost
                + self.cost_bbox * bbox_cost
            )

            cost_np = cost_matrix.numpy()
            from scipy.optimize import linear_sum_assignment

            row_ind, col_ind = linear_sum_assignment(cost_np)
            matches.append(
                (
                    tf.constant(row_ind, dtype=tf.int32),
                    tf.constant(col_ind, dtype=tf.int32),
                )
            )

        return matches


class PETRLoss(tf.keras.layers.Layer):
    """
    Loss computation for PETR including:
      - Focal loss for classification
      - L1 loss for bounding box regression
      - Applied at each decoder layer (auxiliary loss)
    """

    def __init__(
        self,
        num_classes: int = 10,
        cls_weight: float = 2.0,
        bbox_weight: float = 5.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.matcher = HungarianMatcher(cost_class=cls_weight, cost_bbox=bbox_weight)

    def focal_loss(self, pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """
        Compute sigmoid focal loss.

        Args:
            pred: (N, C) logits
            target: (N,) integer class labels
        """
        num_queries = tf.shape(pred)[0]
        target_onehot = tf.one_hot(target, self.num_classes)

        pred_sigmoid = tf.nn.sigmoid(pred)
        pt = target_onehot * pred_sigmoid + (1.0 - target_onehot) * (1.0 - pred_sigmoid)
        focal_weight = (self.focal_alpha * target_onehot + (1.0 - self.focal_alpha) * (1.0 - target_onehot))
        focal_weight = focal_weight * tf.pow(1.0 - pt, self.focal_gamma)

        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_onehot, logits=pred)
        loss = focal_weight * bce
        return tf.reduce_sum(loss) / tf.cast(num_queries, tf.float32)

    def call(
        self,
        outputs: Dict[str, List[tf.Tensor]],
        gt_labels: tf.Tensor,
        gt_bboxes: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """
        Compute total loss across all decoder layers.

        Args:
            outputs: dict with 'cls_scores' and 'bbox_preds' lists
            gt_labels: (B, max_gt) ground truth labels
            gt_bboxes: (B, max_gt, 10) ground truth boxes

        Returns:
            dict with 'total_loss', 'cls_loss', 'bbox_loss'
        """
        total_cls_loss = 0.0
        total_bbox_loss = 0.0
        num_layers = len(outputs["cls_scores"])

        for layer_idx in range(num_layers):
            cls_scores = outputs["cls_scores"][layer_idx]
            bbox_preds = outputs["bbox_preds"][layer_idx]

            matches = self.matcher.match(cls_scores, bbox_preds, gt_labels, gt_bboxes)

            B = tf.shape(cls_scores)[0]
            for b in range(B):
                pred_idx, gt_idx = matches[b]

                if tf.shape(pred_idx)[0] == 0:
                    continue

                matched_cls = tf.gather(cls_scores[b], pred_idx)
                matched_gt_labels = tf.gather(
                    tf.boolean_mask(gt_labels[b], gt_labels[b] >= 0), gt_idx
                )
                total_cls_loss += self.focal_loss(matched_cls, matched_gt_labels)

                matched_bbox = tf.gather(bbox_preds[b], pred_idx)
                matched_gt_bbox = tf.gather(
                    tf.boolean_mask(gt_bboxes[b], gt_labels[b] >= 0), gt_idx
                )
                total_bbox_loss += tf.reduce_mean(
                    tf.reduce_sum(tf.abs(matched_bbox - matched_gt_bbox), axis=-1)
                )

        total_cls_loss = total_cls_loss / tf.cast(num_layers, tf.float32)
        total_bbox_loss = total_bbox_loss / tf.cast(num_layers, tf.float32)

        total_loss = self.cls_weight * total_cls_loss + self.bbox_weight * total_bbox_loss

        return {
            "total_loss": total_loss,
            "cls_loss": total_cls_loss,
            "bbox_loss": total_bbox_loss,
        }


def build_petr_model(config: Dict) -> PETR:
    """Factory function to build PETR model from config dict."""
    return PETR(config)
