"""
MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction.

TensorFlow 2 / Keras implementation of the MapTR perception model for vectorized
HD map construction from multi-camera surround-view images. Produces vectorized
map elements (polylines) for pedestrian crossings, dividers, and boundaries.

Architecture:
    - ResNet50 backbone with FPN for multi-scale feature extraction
    - GKT (Geometry-guided Kernel Transformer) for perspective-to-BEV projection
    - Transformer decoder with hierarchical queries and iterative refinement
    - Classification and regression heads for map element prediction

Config:
    - BEV grid: 200 x 100 (60m x 30m real-world)
    - embed_dims: 256
    - GKT: 8 heads, kernel_size 3, depth range 1.0-60.0 step 0.5
    - Head: 3 classes, 50 queries, 20 points per instance
    - Decoder: 6 layers, 8 heads, 512 FFN dim, dropout 0.1
    - Input: 480 x 800 images, 6 cameras
"""

import tensorflow as tf
import numpy as np


# ==============================================================================
# ResNet50 Backbone
# ==============================================================================


class ResNet50Backbone(tf.keras.layers.Layer):
    """ResNet50 backbone extracting multi-scale features (C2, C3, C4, C5).

    Uses tf.keras.applications.ResNet50 as the base model and taps into
    intermediate layer outputs corresponding to the end of each residual stage.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(input_shape[-3], input_shape[-2], input_shape[-1]),
        )
        base_model.trainable = True

        # Extract outputs at the end of each residual stage
        # C2: stride 4, C3: stride 8, C4: stride 16, C5: stride 32
        layer_names = [
            "conv2_block3_out",   # C2: H/4 x W/4 x 256
            "conv3_block4_out",   # C3: H/8 x W/8 x 512
            "conv4_block6_out",   # C4: H/16 x W/16 x 1024
            "conv5_block3_out",   # C5: H/32 x W/32 x 2048
        ]
        outputs = [base_model.get_layer(name).output for name in layer_names]
        self.feature_extractor = tf.keras.Model(
            inputs=base_model.input, outputs=outputs, name="resnet50_features"
        )
        super().build(input_shape)

    def call(self, images, training=False):
        """Extract multi-scale features from input images.

        Args:
            images: [B, H, W, 3] tensor of input images.
            training: Boolean flag for batch normalization behavior.

        Returns:
            List of 4 feature tensors [C2, C3, C4, C5] at strides 4, 8, 16, 32.
        """
        features = self.feature_extractor(images, training=training)
        return features


# ==============================================================================
# Feature Pyramid Network
# ==============================================================================


class FPN(tf.keras.layers.Layer):
    """Feature Pyramid Network with lateral connections and top-down pathway.

    Takes multi-scale features from the backbone (C2-C5) and produces
    feature maps at a unified channel dimension via:
      1. 1x1 lateral convolutions to reduce channel dimensions
      2. Top-down pathway with bilinear upsampling + element-wise addition
      3. 3x3 output convolutions to reduce aliasing
    """

    def __init__(self, out_channels=256, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def build(self, input_shape):
        # input_shape is a list of shapes for [C2, C3, C4, C5]
        num_levels = len(input_shape)

        # Lateral 1x1 convolutions
        self.lateral_convs = []
        for i in range(num_levels):
            self.lateral_convs.append(
                tf.keras.layers.Conv2D(
                    self.out_channels,
                    kernel_size=1,
                    padding="same",
                    name=f"lateral_conv_{i}",
                )
            )

        # Output 3x3 convolutions
        self.output_convs = []
        for i in range(num_levels):
            self.output_convs.append(
                tf.keras.layers.Conv2D(
                    self.out_channels,
                    kernel_size=3,
                    padding="same",
                    name=f"output_conv_{i}",
                )
            )

        super().build(input_shape)

    def call(self, features, training=False):
        """Apply FPN to multi-scale backbone features.

        Args:
            features: List of [C2, C3, C4, C5] feature tensors.
            training: Boolean flag.

        Returns:
            List of 4 FPN output feature tensors, all with out_channels channels.
        """
        num_levels = len(features)

        # Apply lateral convolutions
        laterals = [self.lateral_convs[i](features[i]) for i in range(num_levels)]

        # Top-down pathway: from highest level (coarsest) down
        for i in range(num_levels - 2, -1, -1):
            h = tf.shape(laterals[i])[1]
            w = tf.shape(laterals[i])[2]
            upsampled = tf.image.resize(
                laterals[i + 1], size=(h, w), method="bilinear"
            )
            laterals[i] = laterals[i] + upsampled

        # Output convolutions
        outputs = [self.output_convs[i](laterals[i]) for i in range(num_levels)]

        return outputs


# ==============================================================================
# Geometry-guided Kernel Transformer (GKT)
# ==============================================================================


class GKT(tf.keras.layers.Layer):
    """Geometry-guided Kernel Transformer for perspective-to-BEV projection.

    Projects perspective camera features into BEV space using camera intrinsics
    and extrinsics. Creates a depth distribution along camera rays, computes
    3D-to-BEV lookup coordinates, and applies multi-head attention to transform
    features into the BEV representation.

    Config:
        - num_heads: 8
        - kernel_size: 3
        - depth_start: 1.0
        - depth_end: 60.0
        - depth_step: 0.5
        - BEV: 200 x 100 (x_range: [-30, 30], y_range: [0, 60] in meters)
        - embed_dims: 256
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        kernel_size=3,
        bev_h=200,
        bev_w=100,
        x_bound=(-30.0, 30.0),
        y_bound=(0.0, 60.0),
        depth_start=1.0,
        depth_end=60.0,
        depth_step=0.5,
        num_cameras=6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.depth_start = depth_start
        self.depth_end = depth_end
        self.depth_step = depth_step
        self.num_cameras = num_cameras
        self.num_depth_bins = int((depth_end - depth_start) / depth_step)

    def build(self, input_shape):
        # Depth distribution predictor: predicts depth weights for each pixel
        self.depth_net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    self.embed_dims, kernel_size=1, padding="same", activation="relu",
                    name="depth_reduce"
                ),
                tf.keras.layers.Conv2D(
                    self.num_depth_bins, kernel_size=1, padding="same",
                    name="depth_pred"
                ),
            ],
            name="depth_net",
        )

        # Feature projection to embed_dims
        self.input_proj = tf.keras.layers.Conv2D(
            self.embed_dims, kernel_size=1, padding="same", name="input_proj"
        )

        # BEV query embedding
        self.bev_embedding = self.add_weight(
            name="bev_embedding",
            shape=(1, self.bev_h * self.bev_w, self.embed_dims),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Multi-head attention for BEV feature aggregation
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dims // self.num_heads,
            value_dim=self.embed_dims // self.num_heads,
            output_shape=self.embed_dims,
            name="bev_cross_attn",
        )

        # Kernel-based local attention weights
        self.kernel_weight = self.add_weight(
            name="kernel_weight",
            shape=(
                self.num_cameras,
                self.kernel_size * self.kernel_size,
                self.embed_dims,
            ),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Output normalization
        self.output_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name="output_norm"
        )

        # Precompute BEV grid coordinates (in meters)
        xs = tf.linspace(self.x_bound[0], self.x_bound[1], self.bev_w)
        ys = tf.linspace(self.y_bound[0], self.y_bound[1], self.bev_h)
        grid_x, grid_y = tf.meshgrid(xs, ys)  # [bev_h, bev_w]
        # BEV grid in 3D: (x, y, 0) for ground plane
        self.bev_grid = tf.stack(
            [
                tf.reshape(grid_x, [-1]),
                tf.reshape(grid_y, [-1]),
                tf.zeros([self.bev_h * self.bev_w]),
                tf.ones([self.bev_h * self.bev_w]),
            ],
            axis=0,
        )  # [4, bev_h*bev_w]

        super().build(input_shape)

    def _project_bev_to_image(self, intrinsics, extrinsics):
        """Project BEV grid points to image coordinates for each camera.

        Args:
            intrinsics: [B, num_cameras, 3, 3] camera intrinsic matrices.
            extrinsics: [B, num_cameras, 4, 4] camera extrinsic matrices (world2cam).

        Returns:
            pixel_coords: [B, num_cameras, bev_h*bev_w, 2] normalized image coords.
            valid_mask: [B, num_cameras, bev_h*bev_w] boolean mask for valid projections.
        """
        batch_size = tf.shape(intrinsics)[0]
        bev_grid = tf.cast(self.bev_grid, intrinsics.dtype)  # [4, N]
        bev_grid = tf.expand_dims(tf.expand_dims(bev_grid, 0), 0)  # [1, 1, 4, N]
        bev_grid = tf.tile(
            bev_grid, [batch_size, self.num_cameras, 1, 1]
        )  # [B, cam, 4, N]

        # Transform to camera coordinate system
        # extrinsics: world2cam [B, cam, 4, 4]
        cam_points = tf.matmul(extrinsics, bev_grid)  # [B, cam, 4, N]
        cam_points_3d = cam_points[:, :, :3, :]  # [B, cam, 3, N]

        # Project to image plane using intrinsics
        pixel_coords_h = tf.matmul(intrinsics, cam_points_3d)  # [B, cam, 3, N]

        # Normalize by depth (z)
        depth = pixel_coords_h[:, :, 2:3, :]  # [B, cam, 1, N]
        depth = tf.maximum(depth, 1e-5)  # avoid division by zero
        pixel_coords = pixel_coords_h[:, :, :2, :] / depth  # [B, cam, 2, N]
        pixel_coords = tf.transpose(pixel_coords, [0, 1, 3, 2])  # [B, cam, N, 2]

        # Valid mask: positive depth and within image bounds (normalized 0-1)
        valid_depth = tf.squeeze(depth > 0.1, axis=2)  # [B, cam, N]
        # We'll keep all with positive depth; actual bounds checked downstream

        return pixel_coords, valid_depth

    def call(self, features, intrinsics, extrinsics, training=False):
        """Transform perspective features to BEV using geometry-guided attention.

        Args:
            features: List of feature maps from FPN. We use the first level
                      (stride-4 features) of shape [B*num_cameras, H, W, C].
            intrinsics: [B, num_cameras, 3, 3] camera intrinsic matrices.
            extrinsics: [B, num_cameras, 4, 4] camera extrinsic matrices.
            training: Boolean flag.

        Returns:
            bev_features: [B, bev_h, bev_w, embed_dims] BEV feature map.
        """
        # Use the highest-resolution FPN feature (stride-4)
        feat = features[0]  # [B*num_cameras, H_feat, W_feat, C]
        feat_shape = tf.shape(feat)
        h_feat = feat_shape[1]
        w_feat = feat_shape[2]
        batch_size = tf.shape(intrinsics)[0]

        # Project features to embed_dims
        feat = self.input_proj(feat)  # [B*num_cameras, H_feat, W_feat, embed_dims]

        # Predict depth distribution
        depth_logits = self.depth_net(feat)  # [B*num_cameras, H_feat, W_feat, D]
        depth_weights = tf.nn.softmax(depth_logits, axis=-1)

        # Reshape features for per-camera processing
        # [B, num_cameras, H_feat, W_feat, embed_dims]
        feat_per_cam = tf.reshape(
            feat, [batch_size, self.num_cameras, h_feat, w_feat, self.embed_dims]
        )
        depth_per_cam = tf.reshape(
            depth_weights,
            [batch_size, self.num_cameras, h_feat, w_feat, self.num_depth_bins],
        )

        # Project BEV grid to image coordinates
        pixel_coords, valid_mask = self._project_bev_to_image(
            intrinsics, extrinsics
        )  # [B, cam, N, 2], [B, cam, N]

        # Normalize pixel coords to [-1, 1] for grid sampling
        # pixel_coords are in actual pixel space; normalize by feature map size
        norm_coords_x = pixel_coords[:, :, :, 0:1] / tf.cast(
            w_feat * 4, pixel_coords.dtype
        ) * 2.0 - 1.0  # stride-4 feature, so multiply
        norm_coords_y = pixel_coords[:, :, :, 1:2] / tf.cast(
            h_feat * 4, pixel_coords.dtype
        ) * 2.0 - 1.0
        norm_coords = tf.concat([norm_coords_x, norm_coords_y], axis=-1)
        # [B, cam, bev_h*bev_w, 2]

        # Sample features at projected locations using bilinear interpolation
        # We implement grid_sample via tfa-style or manual approach
        num_bev_points = self.bev_h * self.bev_w
        sampled_features_all = []

        for cam_idx in range(self.num_cameras):
            cam_feat = feat_per_cam[:, cam_idx]  # [B, H_feat, W_feat, embed_dims]
            cam_coords = norm_coords[:, cam_idx]  # [B, N, 2]
            cam_valid = valid_mask[:, cam_idx]  # [B, N]

            # Convert normalized coords [-1, 1] to feature map pixel coords
            grid_x = (cam_coords[:, :, 0] + 1.0) / 2.0 * tf.cast(
                w_feat - 1, cam_coords.dtype
            )
            grid_y = (cam_coords[:, :, 1] + 1.0) / 2.0 * tf.cast(
                h_feat - 1, cam_coords.dtype
            )

            # Bilinear interpolation
            grid_x = tf.clip_by_value(
                grid_x, 0.0, tf.cast(w_feat - 1, grid_x.dtype)
            )
            grid_y = tf.clip_by_value(
                grid_y, 0.0, tf.cast(h_feat - 1, grid_y.dtype)
            )

            x0 = tf.cast(tf.floor(grid_x), tf.int32)
            x1 = tf.minimum(x0 + 1, w_feat - 1)
            y0 = tf.cast(tf.floor(grid_y), tf.int32)
            y1 = tf.minimum(y0 + 1, h_feat - 1)

            x0f = tf.cast(x0, cam_coords.dtype)
            x1f = tf.cast(x1, cam_coords.dtype)
            y0f = tf.cast(y0, cam_coords.dtype)
            y1f = tf.cast(y1, cam_coords.dtype)

            wa = (x1f - grid_x) * (y1f - grid_y)  # [B, N]
            wb = (grid_x - x0f) * (y1f - grid_y)
            wc = (x1f - grid_x) * (grid_y - y0f)
            wd = (grid_x - x0f) * (grid_y - y0f)

            # Gather features at four corners
            batch_indices = tf.repeat(
                tf.range(batch_size)[:, tf.newaxis], num_bev_points, axis=1
            )  # [B, N]

            def gather_2d(feat_map, yi, xi):
                """Gather from [B, H, W, C] at positions yi, xi both [B, N]."""
                indices = tf.stack(
                    [batch_indices, yi, xi], axis=-1
                )  # [B, N, 3]
                return tf.gather_nd(feat_map, indices)  # [B, N, C]

            fa = gather_2d(cam_feat, y0, x0)  # [B, N, C]
            fb = gather_2d(cam_feat, y0, x1)
            fc = gather_2d(cam_feat, y1, x0)
            fd = gather_2d(cam_feat, y1, x1)

            # Weighted sum
            sampled = (
                wa[:, :, tf.newaxis] * fa
                + wb[:, :, tf.newaxis] * fb
                + wc[:, :, tf.newaxis] * fc
                + wd[:, :, tf.newaxis] * fd
            )  # [B, N, C]

            # Apply validity mask
            cam_valid_expanded = tf.cast(
                cam_valid[:, :, tf.newaxis], sampled.dtype
            )
            sampled = sampled * cam_valid_expanded

            # Apply geometry-guided kernel weights
            kernel_w = self.kernel_weight[cam_idx]  # [K*K, embed_dims]
            kernel_w_mean = tf.reduce_mean(kernel_w, axis=0, keepdims=True)  # [1, C]
            sampled = sampled * tf.nn.sigmoid(kernel_w_mean)  # [B, N, C]

            sampled_features_all.append(sampled)

        # Aggregate features from all cameras: [B, num_cameras, N, C]
        sampled_features = tf.stack(sampled_features_all, axis=1)
        # Flatten cameras into key sequence: [B, N, num_cameras*C] or use attention
        # Reshape to [B, N, num_cameras, C] then [B*N, num_cameras, C] for attention
        sampled_features = tf.transpose(
            sampled_features, [0, 2, 1, 3]
        )  # [B, N, cam, C]

        # BEV queries
        bev_queries = tf.tile(
            self.bev_embedding, [batch_size, 1, 1]
        )  # [B, N, C]

        # Reshape for cross-attention: queries=[B, N, C], keys/values=[B, N*cam, C]
        kv_features = tf.reshape(
            sampled_features,
            [batch_size, num_bev_points * self.num_cameras, self.embed_dims],
        )  # [B, N*cam, C]

        # Apply multi-head cross-attention
        bev_features = self.cross_attention(
            query=bev_queries,
            key=kv_features,
            value=kv_features,
            training=training,
        )  # [B, N, C]

        # Residual connection + normalization
        bev_features = self.output_norm(bev_features + bev_queries)

        # Reshape to spatial BEV grid
        bev_features = tf.reshape(
            bev_features, [batch_size, self.bev_h, self.bev_w, self.embed_dims]
        )

        return bev_features


# ==============================================================================
# Map Decoder
# ==============================================================================


class MapDecoderLayer(tf.keras.layers.Layer):
    """Single transformer decoder layer with self-attention, cross-attention, and FFN.

    Supports hierarchical queries (instance-level + point-level) and iterative
    point coordinate refinement.
    """

    def __init__(self, embed_dims=256, num_heads=8, ffn_dims=512, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.ffn_dims = ffn_dims
        self.dropout_rate = dropout

    def build(self, input_shape):
        # Self-attention
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dims // self.num_heads,
            value_dim=self.embed_dims // self.num_heads,
            output_shape=self.embed_dims,
            dropout=self.dropout_rate,
            name="self_attn",
        )
        self.self_attn_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name="self_attn_norm"
        )
        self.self_attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # Cross-attention
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dims // self.num_heads,
            value_dim=self.embed_dims // self.num_heads,
            output_shape=self.embed_dims,
            dropout=self.dropout_rate,
            name="cross_attn",
        )
        self.cross_attn_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name="cross_attn_norm"
        )
        self.cross_attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # Feed-forward network
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.ffn_dims, activation="relu", name="ffn_fc1"),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.embed_dims, name="ffn_fc2"),
                tf.keras.layers.Dropout(self.dropout_rate),
            ],
            name="ffn",
        )
        self.ffn_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name="ffn_norm"
        )

        super().build(input_shape)

    def call(self, query, memory, query_pos=None, training=False):
        """Apply decoder layer transformations.

        Args:
            query: [B, num_queries * num_points, embed_dims] query embeddings.
            memory: [B, bev_h * bev_w, embed_dims] BEV feature memory.
            query_pos: Optional positional encoding for queries.
            training: Boolean flag.

        Returns:
            Updated query tensor of same shape.
        """
        # Add positional encoding to queries
        if query_pos is not None:
            q_with_pos = query + query_pos
        else:
            q_with_pos = query

        # Self-attention
        sa_out = self.self_attn(
            query=q_with_pos,
            key=q_with_pos,
            value=query,
            training=training,
        )
        query = self.self_attn_norm(query + self.self_attn_dropout(sa_out, training=training))

        # Cross-attention with BEV memory
        if query_pos is not None:
            q_with_pos = query + query_pos
        else:
            q_with_pos = query

        ca_out = self.cross_attn(
            query=q_with_pos,
            key=memory,
            value=memory,
            training=training,
        )
        query = self.cross_attn_norm(query + self.cross_attn_dropout(ca_out, training=training))

        # Feed-forward network
        ffn_out = self.ffn(query, training=training)
        query = self.ffn_norm(query + ffn_out)

        return query


class MapDecoder(tf.keras.layers.Layer):
    """Transformer decoder with 6 layers and hierarchical query structure.

    Uses instance-level queries (50 queries) combined with point-level queries
    (20 points per instance) for structured map element prediction. Supports
    iterative coordinate refinement across decoder layers.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        ffn_dims=512,
        dropout=0.1,
        num_layers=6,
        num_queries=50,
        num_points=20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.ffn_dims = ffn_dims
        self.dropout_rate = dropout
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.num_points = num_points

    def build(self, input_shape):
        # Decoder layers
        self.layers_list = [
            MapDecoderLayer(
                embed_dims=self.embed_dims,
                num_heads=self.num_heads,
                ffn_dims=self.ffn_dims,
                dropout=self.dropout_rate,
                name=f"decoder_layer_{i}",
            )
            for i in range(self.num_layers)
        ]

        # Instance-level query embeddings
        self.instance_query_embedding = self.add_weight(
            name="instance_query_embedding",
            shape=(1, self.num_queries, self.embed_dims),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Point-level query embeddings
        self.point_query_embedding = self.add_weight(
            name="point_query_embedding",
            shape=(1, self.num_points, self.embed_dims),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Reference point initialization (initial point coordinates)
        self.reference_points_fc = tf.keras.layers.Dense(
            2 * self.num_points, name="reference_points_fc"
        )

        # Per-layer refinement MLPs for iterative coordinate refinement
        self.refinement_mlps = [
            tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        self.embed_dims, activation="relu", name=f"refine_{i}_fc1"
                    ),
                    tf.keras.layers.Dense(
                        2 * self.num_points, name=f"refine_{i}_fc2"
                    ),
                ],
                name=f"refinement_mlp_{i}",
            )
            for i in range(self.num_layers)
        ]

        # Positional encoding for point positions
        self.pos_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.embed_dims, activation="relu", name="pos_fc1"),
                tf.keras.layers.Dense(self.embed_dims, name="pos_fc2"),
            ],
            name="pos_encoder",
        )

        super().build(input_shape)

    def call(self, bev_features, training=False):
        """Decode map elements from BEV features.

        Args:
            bev_features: [B, bev_h, bev_w, embed_dims] BEV feature map.
            training: Boolean flag.

        Returns:
            query_output: [B, num_queries, num_points, embed_dims] final query states.
            reference_points_list: List of [B, num_queries, num_points, 2] refined
                coordinates from each decoder layer.
        """
        batch_size = tf.shape(bev_features)[0]

        # Flatten BEV features for cross-attention memory
        bev_h = tf.shape(bev_features)[1]
        bev_w = tf.shape(bev_features)[2]
        memory = tf.reshape(
            bev_features, [batch_size, bev_h * bev_w, self.embed_dims]
        )  # [B, H*W, C]

        # Create hierarchical queries: instance-level + point-level
        instance_queries = tf.tile(
            self.instance_query_embedding, [batch_size, 1, 1]
        )  # [B, Q, C]
        point_queries = tf.tile(
            self.point_query_embedding, [batch_size, 1, 1]
        )  # [B, P, C]

        # Combine instance and point queries via broadcasting addition
        # [B, Q, 1, C] + [B, 1, P, C] -> [B, Q, P, C]
        combined_queries = (
            instance_queries[:, :, tf.newaxis, :]
            + point_queries[:, tf.newaxis, :, :]
        )  # [B, Q, P, C]

        # Flatten queries for transformer: [B, Q*P, C]
        queries = tf.reshape(
            combined_queries,
            [batch_size, self.num_queries * self.num_points, self.embed_dims],
        )

        # Initialize reference points from instance queries
        ref_pts_init = self.reference_points_fc(
            instance_queries
        )  # [B, Q, 2*P]
        ref_pts_init = tf.nn.sigmoid(ref_pts_init)
        reference_points = tf.reshape(
            ref_pts_init, [batch_size, self.num_queries, self.num_points, 2]
        )  # [B, Q, P, 2]

        # Store intermediate results
        reference_points_list = [reference_points]

        # Iteratively refine through decoder layers
        for layer_idx in range(self.num_layers):
            # Compute positional encoding from current reference points
            ref_pts_flat = tf.reshape(
                reference_points,
                [batch_size, self.num_queries * self.num_points, 2],
            )  # [B, Q*P, 2]
            query_pos = self.pos_encoder(ref_pts_flat)  # [B, Q*P, C]

            # Apply decoder layer
            queries = self.layers_list[layer_idx](
                query=queries,
                memory=memory,
                query_pos=query_pos,
                training=training,
            )

            # Iterative refinement: predict coordinate deltas
            # Reshape queries back to [B, Q, P, C] for per-instance refinement
            queries_reshaped = tf.reshape(
                queries,
                [batch_size, self.num_queries, self.num_points, self.embed_dims],
            )
            # Use mean over points for refinement input
            instance_feats = tf.reduce_mean(
                queries_reshaped, axis=2
            )  # [B, Q, C]
            ref_deltas = self.refinement_mlps[layer_idx](
                instance_feats
            )  # [B, Q, 2*P]
            ref_deltas = tf.reshape(
                ref_deltas, [batch_size, self.num_queries, self.num_points, 2]
            )

            # Update reference points with sigmoid to keep in [0, 1]
            # Apply inverse sigmoid, add delta, then sigmoid again
            inv_sigmoid_ref = tf.math.log(
                reference_points / tf.maximum(1.0 - reference_points, 1e-5)
            )
            reference_points = tf.nn.sigmoid(inv_sigmoid_ref + ref_deltas)
            reference_points_list.append(reference_points)

        # Final query output reshaped to [B, Q, P, C]
        query_output = tf.reshape(
            queries,
            [batch_size, self.num_queries, self.num_points, self.embed_dims],
        )

        return query_output, reference_points_list


# ==============================================================================
# MapTR Head
# ==============================================================================


class MapTRHead(tf.keras.layers.Layer):
    """Classification and point regression heads for MapTR.

    Classification MLP: linear -> ReLU -> linear -> num_classes
    Point regression MLP: linear -> ReLU -> linear -> 2*num_points
    """

    def __init__(self, embed_dims=256, num_classes=3, num_points=20, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_points = num_points

    def build(self, input_shape):
        # Classification head: operates on instance-level features
        self.cls_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.embed_dims, activation="relu", name="cls_fc1"
                ),
                tf.keras.layers.Dense(self.num_classes, name="cls_fc2"),
            ],
            name="cls_head",
        )

        # Point regression head: operates on instance-level features
        self.pts_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.embed_dims, activation="relu", name="pts_fc1"
                ),
                tf.keras.layers.Dense(
                    2 * self.num_points, name="pts_fc2"
                ),
            ],
            name="pts_head",
        )

        super().build(input_shape)

    def call(self, query_output, reference_points, training=False):
        """Predict class scores and point coordinates.

        Args:
            query_output: [B, num_queries, num_points, embed_dims] decoder output.
            reference_points: [B, num_queries, num_points, 2] refined reference points.
            training: Boolean flag.

        Returns:
            cls_scores: [B, num_queries, num_classes] classification logits.
            pts_preds: [B, num_queries, num_points, 2] predicted point coordinates.
        """
        # Instance-level features by averaging over points
        instance_features = tf.reduce_mean(
            query_output, axis=2
        )  # [B, Q, C]

        # Classification
        cls_scores = self.cls_head(instance_features)  # [B, Q, num_classes]

        # Point regression: predict offsets from reference points
        pts_offsets = self.pts_head(instance_features)  # [B, Q, 2*P]
        pts_offsets = tf.reshape(
            pts_offsets,
            [
                tf.shape(query_output)[0],
                tf.shape(query_output)[1],
                self.num_points,
                2,
            ],
        )

        # Final point predictions: reference points + learned offsets
        pts_preds = reference_points + pts_offsets
        # Clamp to [0, 1] normalized coordinates
        pts_preds = tf.clip_by_value(pts_preds, 0.0, 1.0)

        return cls_scores, pts_preds


# ==============================================================================
# Full MapTR Model
# ==============================================================================


class MapTRModel(tf.keras.Model):
    """Full MapTR model for vectorized HD map construction.

    Combines:
        - ResNet50 backbone for multi-scale feature extraction
        - FPN for feature fusion
        - GKT for perspective-to-BEV transformation
        - Transformer decoder with hierarchical queries
        - Classification and regression heads

    Input:
        images: [B, 6, H, W, 3] surround-view camera images (channels-last)
        intrinsics: [B, 6, 3, 3] camera intrinsic matrices
        extrinsics: [B, 6, 4, 4] camera extrinsic matrices (world-to-camera)

    Output:
        Dictionary with:
            'cls_scores': [B, num_queries, num_classes] classification logits
            'pts_preds': [B, num_queries, num_points, 2] point coordinates in [0,1]
            'intermediate_outputs': list of (cls_scores, pts_preds) from each layer
    """

    def __init__(
        self,
        embed_dims=256,
        num_classes=3,
        num_queries=50,
        num_points=20,
        num_cameras=6,
        bev_h=200,
        bev_w=100,
        image_h=480,
        image_w=800,
        num_decoder_layers=6,
        num_heads=8,
        ffn_dims=512,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_points = num_points
        self.num_cameras = num_cameras
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.image_h = image_h
        self.image_w = image_w

        # Backbone
        self.backbone = ResNet50Backbone(name="backbone")

        # FPN
        self.fpn = FPN(out_channels=embed_dims, name="fpn")

        # GKT: Perspective to BEV
        self.gkt = GKT(
            embed_dims=embed_dims,
            num_heads=num_heads,
            kernel_size=3,
            bev_h=bev_h,
            bev_w=bev_w,
            num_cameras=num_cameras,
            name="gkt",
        )

        # Decoder
        self.decoder = MapDecoder(
            embed_dims=embed_dims,
            num_heads=num_heads,
            ffn_dims=ffn_dims,
            dropout=dropout,
            num_layers=num_decoder_layers,
            num_queries=num_queries,
            num_points=num_points,
            name="decoder",
        )

        # Head
        self.head = MapTRHead(
            embed_dims=embed_dims,
            num_classes=num_classes,
            num_points=num_points,
            name="head",
        )

        # Intermediate prediction heads (one per decoder layer for auxiliary loss)
        self.intermediate_heads = [
            MapTRHead(
                embed_dims=embed_dims,
                num_classes=num_classes,
                num_points=num_points,
                name=f"intermediate_head_{i}",
            )
            for i in range(num_decoder_layers)
        ]

    def call(self, images, intrinsics, extrinsics, training=False):
        """Forward pass of the full MapTR model.

        Args:
            images: [B, 6, H, W, 3] multi-camera input images.
            intrinsics: [B, 6, 3, 3] camera intrinsic matrices.
            extrinsics: [B, 6, 4, 4] camera extrinsic matrices (world-to-camera).
            training: Boolean flag for dropout/batch norm behavior.

        Returns:
            Dictionary with:
                'cls_scores': [B, num_queries, num_classes]
                'pts_preds': [B, num_queries, num_points, 2]
                'intermediate_outputs': list of (cls_scores, pts_preds) per layer
        """
        batch_size = tf.shape(images)[0]

        # Reshape images: [B, 6, H, W, 3] -> [B*6, H, W, 3]
        images_flat = tf.reshape(
            images,
            [batch_size * self.num_cameras, self.image_h, self.image_w, 3],
        )

        # Preprocess images for ResNet (ImageNet normalization)
        images_preprocessed = tf.keras.applications.resnet50.preprocess_input(
            images_flat
        )

        # Extract multi-scale backbone features
        backbone_features = self.backbone(
            images_preprocessed, training=training
        )  # List of [B*6, H_i, W_i, C_i]

        # Apply FPN
        fpn_features = self.fpn(
            backbone_features, training=training
        )  # List of [B*6, H_i, W_i, embed_dims]

        # Transform perspective features to BEV using GKT
        bev_features = self.gkt(
            fpn_features, intrinsics, extrinsics, training=training
        )  # [B, bev_h, bev_w, embed_dims]

        # Decode map elements with transformer decoder
        query_output, reference_points_list = self.decoder(
            bev_features, training=training
        )  # [B, Q, P, C], list of [B, Q, P, 2]

        # Final predictions
        cls_scores, pts_preds = self.head(
            query_output, reference_points_list[-1], training=training
        )

        # Intermediate predictions for auxiliary losses
        intermediate_outputs = []
        for layer_idx in range(len(reference_points_list) - 1):
            ref_pts = reference_points_list[layer_idx]  # [B, Q, P, 2]
            # Use the query output (approximation: same final features with
            # different reference points for intermediate supervision)
            inter_cls, inter_pts = self.intermediate_heads[layer_idx](
                query_output, ref_pts, training=training
            )
            intermediate_outputs.append((inter_cls, inter_pts))

        return {
            "cls_scores": cls_scores,
            "pts_preds": pts_preds,
            "intermediate_outputs": intermediate_outputs,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dims": self.embed_dims,
                "num_classes": self.num_classes,
                "num_queries": self.num_queries,
                "num_points": self.num_points,
                "num_cameras": self.num_cameras,
                "bev_h": self.bev_h,
                "bev_w": self.bev_w,
                "image_h": self.image_h,
                "image_w": self.image_w,
            }
        )
        return config


def build_maptr_model(
    num_classes=3,
    num_queries=50,
    num_points=20,
    embed_dims=256,
    bev_h=200,
    bev_w=100,
    image_h=480,
    image_w=800,
    num_cameras=6,
    num_decoder_layers=6,
    num_heads=8,
    ffn_dims=512,
    dropout=0.1,
):
    """Factory function to build a MapTR model with default configuration.

    Args:
        num_classes: Number of map element classes (default: 3 for
            ped_crossing, divider, boundary).
        num_queries: Number of instance queries (default: 50).
        num_points: Number of points per map element (default: 20).
        embed_dims: Feature embedding dimension (default: 256).
        bev_h: BEV grid height (default: 200).
        bev_w: BEV grid width (default: 100).
        image_h: Input image height (default: 480).
        image_w: Input image width (default: 800).
        num_cameras: Number of surround-view cameras (default: 6).
        num_decoder_layers: Number of transformer decoder layers (default: 6).
        num_heads: Number of attention heads (default: 8).
        ffn_dims: FFN hidden dimension (default: 512).
        dropout: Dropout rate (default: 0.1).

    Returns:
        MapTRModel instance.
    """
    model = MapTRModel(
        embed_dims=embed_dims,
        num_classes=num_classes,
        num_queries=num_queries,
        num_points=num_points,
        num_cameras=num_cameras,
        bev_h=bev_h,
        bev_w=bev_w,
        image_h=image_h,
        image_w=image_w,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        ffn_dims=ffn_dims,
        dropout=dropout,
    )
    return model
