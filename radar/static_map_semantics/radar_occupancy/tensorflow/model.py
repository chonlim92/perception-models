# [IMPLEMENTED BY CLAUDE - was missing]
"""
Radar Occupancy Grid Mapping — TensorFlow 2 / Keras Implementation

Three approaches:
1. ClassicalISM: Bayesian inverse sensor model (no learning)
2. PillarOccNet: Neural single-frame occupancy prediction
3. TemporalPillarOccNet: Multi-frame temporal fusion

This mirrors the PyTorch model architecture using tf.keras throughout.
"""

import math
import numpy as np
import tensorflow as tf


class PillarFeatureNet(tf.keras.layers.Layer):
    """Encode radar points into pillar features using PointNet-style MLP.

    Input: per-pillar point features (x, y, z, rcs, vr_comp, dt + relative offsets)
    Output: max-pooled pillar encodings
    """

    def __init__(self, in_channels=9, out_channels=64, max_points_per_pillar=20,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_points = max_points_per_pillar

        # PointNet MLP: in_channels -> 32 -> 64 -> out_channels
        self.dense1 = tf.keras.layers.Dense(32, use_bias=True)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(64, use_bias=True)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(out_channels, use_bias=True)
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, pillar_features, pillar_indices, num_pillars, training=False):
        """
        Args:
            pillar_features: (B, max_pillars, max_points, C_in)
            pillar_indices: (B, max_pillars, 2) grid indices for each pillar
            num_pillars: (B,) actual number of pillars per sample
            training: bool for BatchNorm behavior

        Returns:
            pillar_encodings: (B, max_pillars, C_out)
        """
        B = tf.shape(pillar_features)[0]
        P = tf.shape(pillar_features)[1]
        N = tf.shape(pillar_features)[2]
        C = tf.shape(pillar_features)[3]

        # Flatten to apply shared MLP across all points
        # (B, P, N, C_in) -> (B*P*N, C_in)
        features_flat = tf.reshape(pillar_features, [B * P * N, self.in_channels])

        # MLP with BatchNorm and ReLU
        x = self.dense1(features_flat)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.dense3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        # Reshape back: (B*P*N, C_out) -> (B, P, N, C_out)
        encoded = tf.reshape(x, [B, P, N, self.out_channels])

        # Max-pool over points dimension -> (B, P, C_out)
        pillar_encodings = tf.reduce_max(encoded, axis=2)

        return pillar_encodings


class ScatterBEV(tf.keras.layers.Layer):
    """Scatter pillar features to BEV pseudo-image (channels-last format)."""

    def __init__(self, grid_size, channels, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size  # [H, W]
        self.channels = channels

    def call(self, pillar_features, pillar_indices, num_pillars):
        """
        Args:
            pillar_features: (B, max_pillars, C)
            pillar_indices: (B, max_pillars, 2) [grid_x, grid_y]
            num_pillars: (B,)

        Returns:
            bev: (B, H, W, C) -- TF uses channels-last
        """
        B = tf.shape(pillar_features)[0]
        H, W = self.grid_size
        C = self.channels

        # Initialize empty BEV grid
        bev = tf.zeros([B, H, W, C], dtype=pillar_features.dtype)

        # Scatter pillar features to grid using tf.tensor_scatter_nd_update
        for b in tf.range(B):
            n = num_pillars[b]
            if n == 0:
                continue

            indices = pillar_indices[b, :n]  # (n, 2)
            features = pillar_features[b, :n]  # (n, C)

            # Filter valid indices
            valid_x = tf.logical_and(indices[:, 0] >= 0, indices[:, 0] < H)
            valid_y = tf.logical_and(indices[:, 1] >= 0, indices[:, 1] < W)
            valid = tf.logical_and(valid_x, valid_y)

            valid_indices = tf.boolean_mask(indices, valid)  # (n_valid, 2)
            valid_features = tf.boolean_mask(features, valid)  # (n_valid, C)

            # Build scatter indices: prepend batch index
            batch_idx = tf.fill([tf.shape(valid_indices)[0], 1], b)
            scatter_indices = tf.concat([batch_idx, valid_indices], axis=1)  # (n_valid, 3)

            bev = tf.tensor_scatter_nd_update(bev, scatter_indices, valid_features)

        return bev

    @tf.function
    def call_vectorized(self, pillar_features, pillar_indices, num_pillars):
        """Vectorized scatter using batch indices (alternative implementation).

        This avoids the Python-level loop for better GPU utilization.
        """
        B = tf.shape(pillar_features)[0]
        H, W = self.grid_size
        C = self.channels

        bev = tf.zeros([B, H, W, C], dtype=pillar_features.dtype)

        # Create batch indices
        batch_range = tf.range(B)
        batch_indices = tf.repeat(batch_range, tf.cast(num_pillars, tf.int32))

        # Create mask for valid pillars
        max_pillars = tf.shape(pillar_features)[1]
        pillar_range = tf.range(max_pillars)
        # (B, max_pillars) mask
        mask = pillar_range[tf.newaxis, :] < num_pillars[:, tf.newaxis]

        # Flatten valid entries
        flat_features = tf.boolean_mask(pillar_features, mask)  # (total_valid, C)
        flat_indices = tf.boolean_mask(pillar_indices, mask)  # (total_valid, 2)
        flat_batch = tf.boolean_mask(
            tf.broadcast_to(tf.range(B)[:, tf.newaxis], [B, max_pillars]),
            mask
        )  # (total_valid,)

        # Filter spatially valid indices
        valid_x = tf.logical_and(flat_indices[:, 0] >= 0, flat_indices[:, 0] < H)
        valid_y = tf.logical_and(flat_indices[:, 1] >= 0, flat_indices[:, 1] < W)
        valid = tf.logical_and(valid_x, valid_y)

        flat_features = tf.boolean_mask(flat_features, valid)
        flat_indices = tf.boolean_mask(flat_indices, valid)
        flat_batch = tf.boolean_mask(flat_batch, valid)

        # Build full scatter indices: (batch, grid_x, grid_y)
        scatter_indices = tf.stack(
            [flat_batch, flat_indices[:, 0], flat_indices[:, 1]], axis=1
        )

        bev = tf.tensor_scatter_nd_update(bev, scatter_indices, flat_features)
        return bev


class ConvBlock(tf.keras.layers.Layer):
    """Conv2D -> BN -> ReLU -> Conv2D -> BN -> ReLU with optional stride."""

    def __init__(self, out_channels, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            out_channels, 3, strides=stride, padding='same', use_bias=False
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            out_channels, 3, strides=1, padding='same', use_bias=False
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        return x


class UpBlock(tf.keras.layers.Layer):
    """ConvTranspose2D -> BN -> ReLU -> Concat skip -> ConvBlock."""

    def __init__(self, out_channels, skip_channels, **kwargs):
        super().__init__(**kwargs)
        self.up_conv = tf.keras.layers.Conv2DTranspose(
            out_channels, 4, strides=2, padding='same', use_bias=False
        )
        self.bn_up = tf.keras.layers.BatchNormalization()
        self.conv_block = ConvBlock(out_channels)
        self.skip_channels = skip_channels

    def call(self, x, skip, training=False):
        x = self.up_conv(x)
        x = self.bn_up(x, training=training)
        x = tf.nn.relu(x)

        # Handle spatial size mismatch
        x_shape = tf.shape(x)
        skip_shape = tf.shape(skip)
        if x_shape[1] != skip_shape[1] or x_shape[2] != skip_shape[2]:
            x = tf.image.resize(x, [skip_shape[1], skip_shape[2]],
                                method='bilinear')

        # Concatenate along channel axis (last dim in channels-last)
        x = tf.concat([x, skip], axis=-1)
        x = self.conv_block(x, training=training)
        return x


class UNetBackbone(tf.keras.Model):
    """U-Net encoder-decoder for BEV feature processing.

    Uses channels-last format (B, H, W, C) throughout.
    """

    def __init__(self, in_channels=64, encoder_channels=None, decoder_channels=None,
                 **kwargs):
        super().__init__(**kwargs)
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]
        if decoder_channels is None:
            decoder_channels = [256, 128, 64]

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        # Encoder: first block stride=1, rest stride=2
        self.encoders = []
        for i, ch_out in enumerate(encoder_channels):
            stride = 1 if i == 0 else 2
            self.encoders.append(ConvBlock(ch_out, stride=stride,
                                           name=f'encoder_{i}'))

        # Decoder: UpBlocks with skip connections
        enc_channels_rev = list(reversed(encoder_channels[:-1]))
        self.decoders = []
        for i, ch_out in enumerate(decoder_channels):
            skip_ch = enc_channels_rev[i] if i < len(enc_channels_rev) else 0
            self.decoders.append(UpBlock(ch_out, skip_channels=skip_ch,
                                         name=f'decoder_{i}'))

        self.out_channels = decoder_channels[-1]

    def call(self, x, training=False):
        """
        Args:
            x: (B, H, W, C_in) BEV feature map

        Returns:
            (B, H, W, C_out) decoded feature map
        """
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x, training=training)
            if i < len(self.encoders) - 1:
                skips.append(x)

        skips = list(reversed(skips))
        for i, dec in enumerate(self.decoders):
            if i < len(skips):
                skip = skips[i]
            else:
                skip = tf.zeros_like(x)
            x = dec(x, skip, training=training)

        return x


class PillarOccNet(tf.keras.Model):
    """Neural radar occupancy prediction from a single frame.

    Architecture:
        PillarFeatureNet -> ScatterBEV -> UNetBackbone -> Heads
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        pillar_cfg = config["model"]["pillar"]
        backbone_cfg = config["model"]["backbone"]
        heads_cfg = config["model"]["heads"]
        grid_cfg = config["grid"]

        self.grid_size = grid_cfg["grid_size"]

        self.pillar_net = PillarFeatureNet(
            in_channels=pillar_cfg["input_features"],
            out_channels=pillar_cfg["pillar_features"],
            max_points_per_pillar=pillar_cfg["max_points_per_pillar"],
            name='pillar_feature_net',
        )

        self.scatter = ScatterBEV(
            grid_size=self.grid_size,
            channels=pillar_cfg["pillar_features"],
            name='scatter_bev',
        )

        self.backbone = UNetBackbone(
            in_channels=pillar_cfg["pillar_features"],
            encoder_channels=backbone_cfg["encoder_channels"],
            decoder_channels=backbone_cfg["decoder_channels"],
            name='unet_backbone',
        )

        out_ch = backbone_cfg["decoder_channels"][-1]

        # Occupancy head: 1x1 conv -> 1 channel (sigmoid at inference)
        self.occ_head = tf.keras.layers.Conv2D(
            1, kernel_size=1, padding='same', name='occupancy_head'
        )

        # Optional semantic head
        self.semantic_enabled = heads_cfg["semantics"]["enabled"]
        if self.semantic_enabled:
            num_classes = heads_cfg["semantics"]["num_classes"]
            self.sem_head = tf.keras.Sequential([
                tf.keras.layers.Conv2D(out_ch // 2, 3, padding='same',
                                       use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(num_classes, 1, padding='same'),
            ], name='semantic_head')
        else:
            self.sem_head = None

    def call(self, inputs, training=False):
        """
        Args:
            inputs: dict with keys:
                'pillar_features': (B, max_pillars, max_points, C_in)
                'pillar_indices': (B, max_pillars, 2)
                'num_pillars': (B,)

        Returns:
            dict with:
                'occupancy': (B, H, W, 1) logits
                'semantic': (B, H, W, K) logits (if enabled)
        """
        pillar_features = inputs['pillar_features']
        pillar_indices = inputs['pillar_indices']
        num_pillars = inputs['num_pillars']

        # Encode pillars
        pillar_enc = self.pillar_net(
            pillar_features, pillar_indices, num_pillars, training=training
        )

        # Scatter to BEV grid
        bev = self.scatter(pillar_enc, pillar_indices, num_pillars)

        # Backbone feature extraction
        features = self.backbone(bev, training=training)

        # Prediction heads
        outputs = {'occupancy': self.occ_head(features)}
        if self.sem_head is not None:
            outputs['semantic'] = self.sem_head(features, training=training)

        return outputs


class TemporalPillarOccNet(tf.keras.Model):
    """Multi-frame temporal radar occupancy prediction.

    Fuses BEV features from current + past T frames using ego-motion compensation.
    Supports three fusion methods: concat_conv, attention, gru.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        pillar_cfg = config["model"]["pillar"]
        backbone_cfg = config["model"]["backbone"]
        heads_cfg = config["model"]["heads"]
        temporal_cfg = config["model"]["temporal"]
        grid_cfg = config["grid"]

        self.grid_size = grid_cfg["grid_size"]
        self.cell_size = grid_cfg["cell_size"]
        self.x_range = grid_cfg["x_range"]
        self.y_range = grid_cfg["y_range"]
        self.num_frames = temporal_cfg["num_frames"]
        self.fusion_method = temporal_cfg["fusion_method"]

        feat_ch = pillar_cfg["pillar_features"]

        self.pillar_net = PillarFeatureNet(
            in_channels=pillar_cfg["input_features"],
            out_channels=feat_ch,
            max_points_per_pillar=pillar_cfg["max_points_per_pillar"],
            name='pillar_feature_net',
        )

        self.scatter = ScatterBEV(
            grid_size=self.grid_size,
            channels=feat_ch,
            name='scatter_bev',
        )

        # Temporal fusion layers
        if self.fusion_method == "concat_conv":
            temporal_conv_ch = temporal_cfg["temporal_conv_channels"]
            self.temporal_conv = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    temporal_conv_ch, 3, padding='same', use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ], name='temporal_conv')
            backbone_in = temporal_conv_ch
        elif self.fusion_method == "attention":
            self.temporal_attn = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=feat_ch // 4, name='temporal_attn'
            )
            self.temporal_norm = tf.keras.layers.LayerNormalization(
                name='temporal_norm'
            )
            backbone_in = feat_ch
        elif self.fusion_method == "gru":
            self.temporal_gru = tf.keras.layers.GRU(
                feat_ch, return_sequences=True, name='temporal_gru'
            )
            backbone_in = feat_ch
        else:
            backbone_in = feat_ch

        self.backbone = UNetBackbone(
            in_channels=backbone_in,
            encoder_channels=backbone_cfg["encoder_channels"],
            decoder_channels=backbone_cfg["decoder_channels"],
            name='unet_backbone',
        )

        out_ch = backbone_cfg["decoder_channels"][-1]

        # Occupancy head
        self.occ_head = tf.keras.layers.Conv2D(
            1, kernel_size=1, padding='same', name='occupancy_head'
        )

        # Optional semantic head
        self.semantic_enabled = heads_cfg["semantics"]["enabled"]
        if self.semantic_enabled:
            num_classes = heads_cfg["semantics"]["num_classes"]
            self.sem_head = tf.keras.Sequential([
                tf.keras.layers.Conv2D(out_ch // 2, 3, padding='same',
                                       use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(num_classes, 1, padding='same'),
            ], name='semantic_head')
        else:
            self.sem_head = None

    def warp_bev(self, bev_features, ego_transform):
        """Warp past BEV features to current ego frame via bilinear sampling.

        Implements manual bilinear interpolation (no dependency on tfa).

        Args:
            bev_features: (B, H, W, C) past frame BEV features (channels-last)
            ego_transform: (B, 4, 4) transform from past to current frame

        Returns:
            warped: (B, H, W, C) aligned BEV features
        """
        B = tf.shape(bev_features)[0]
        H, W = self.grid_size
        C = tf.shape(bev_features)[-1]

        # Build world coordinate grid
        # ys corresponds to rows (H dimension), xs to columns (W dimension)
        ys = tf.linspace(
            float(self.y_range[0]), float(self.y_range[1]), H
        )  # (H,)
        xs = tf.linspace(
            float(self.x_range[0]), float(self.x_range[1]), W
        )  # (W,)

        # Create meshgrid: grid_y is (H, W), grid_x is (H, W)
        grid_x, grid_y = tf.meshgrid(xs, ys)  # both (H, W)

        ones = tf.ones_like(grid_x)
        zeros = tf.zeros_like(grid_x)

        # Homogeneous coordinates: (H, W, 4)
        coords = tf.stack([grid_x, grid_y, zeros, ones], axis=-1)
        # Reshape to (4, H*W) for matrix multiply
        coords_flat = tf.reshape(coords, [H * W, 4])
        coords_flat = tf.transpose(coords_flat)  # (4, H*W)

        # Process each batch element
        warped_list = []
        for b in tf.range(B):
            T = ego_transform[b]  # (4, 4)
            T_inv = tf.linalg.inv(T)

            # Transform current coordinates to past frame
            past_coords = tf.linalg.matmul(T_inv, coords_flat)  # (4, H*W)
            past_x = past_coords[0]  # (H*W,)
            past_y = past_coords[1]  # (H*W,)

            # Normalize to pixel coordinates [0, W-1] and [0, H-1]
            pixel_x = (past_x - self.x_range[0]) / \
                      (self.x_range[1] - self.x_range[0]) * tf.cast(W - 1, tf.float32)
            pixel_y = (past_y - self.y_range[0]) / \
                      (self.y_range[1] - self.y_range[0]) * tf.cast(H - 1, tf.float32)

            # Bilinear interpolation
            warped_frame = self._bilinear_sample(
                bev_features[b], pixel_x, pixel_y, H, W
            )  # (H*W, C)
            warped_frame = tf.reshape(warped_frame, [H, W, C])
            warped_list.append(warped_frame)

        warped = tf.stack(warped_list, axis=0)  # (B, H, W, C)
        return warped

    @staticmethod
    def _bilinear_sample(image, x, y, H, W):
        """Manual bilinear sampling from a single image.

        Args:
            image: (H, W, C) source feature map
            x: (N,) x coordinates in pixel space
            y: (N,) y coordinates in pixel space
            H: image height
            W: image width

        Returns:
            sampled: (N, C) interpolated features
        """
        C = tf.shape(image)[-1]
        N = tf.shape(x)[0]

        # Clamp to valid range
        x = tf.clip_by_value(x, 0.0, tf.cast(W - 1, tf.float32))
        y = tf.clip_by_value(y, 0.0, tf.cast(H - 1, tf.float32))

        # Get corner pixel indices
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        # Clamp indices
        x0 = tf.clip_by_value(x0, 0, W - 1)
        x1 = tf.clip_by_value(x1, 0, W - 1)
        y0 = tf.clip_by_value(y0, 0, H - 1)
        y1 = tf.clip_by_value(y1, 0, H - 1)

        # Gather corner values
        # Flatten image for gathering: (H*W, C)
        image_flat = tf.reshape(image, [H * W, C])

        idx_00 = y0 * W + x0
        idx_01 = y0 * W + x1
        idx_10 = y1 * W + x0
        idx_11 = y1 * W + x1

        val_00 = tf.gather(image_flat, idx_00)  # (N, C)
        val_01 = tf.gather(image_flat, idx_01)
        val_10 = tf.gather(image_flat, idx_10)
        val_11 = tf.gather(image_flat, idx_11)

        # Compute interpolation weights
        x0_f = tf.cast(x0, tf.float32)
        x1_f = tf.cast(x1, tf.float32)
        y0_f = tf.cast(y0, tf.float32)
        y1_f = tf.cast(y1, tf.float32)

        wa = tf.expand_dims((x1_f - x) * (y1_f - y), axis=1)  # (N, 1)
        wb = tf.expand_dims((x - x0_f) * (y1_f - y), axis=1)
        wc = tf.expand_dims((x1_f - x) * (y - y0_f), axis=1)
        wd = tf.expand_dims((x - x0_f) * (y - y0_f), axis=1)

        sampled = wa * val_00 + wb * val_01 + wc * val_10 + wd * val_11
        return sampled

    def call(self, inputs, training=False):
        """
        Args:
            inputs: dict with keys:
                'pillar_features_seq': list of T tensors, each (B, max_pillars, max_points, C)
                'pillar_indices_seq': list of T tensors, each (B, max_pillars, 2)
                'num_pillars_seq': list of T tensors, each (B,)
                'ego_transforms': (B, T-1, 4, 4) transforms from past frames to current

        Returns:
            dict with:
                'occupancy': (B, H, W, 1) logits
                'semantic': (B, H, W, K) logits (if enabled)
        """
        pillar_features_seq = inputs['pillar_features_seq']
        pillar_indices_seq = inputs['pillar_indices_seq']
        num_pillars_seq = inputs['num_pillars_seq']
        ego_transforms = inputs['ego_transforms']

        bev_features_list = []

        for t in range(self.num_frames):
            # Encode pillars for frame t
            pillar_enc = self.pillar_net(
                pillar_features_seq[t], pillar_indices_seq[t],
                num_pillars_seq[t], training=training
            )
            # Scatter to BEV
            bev = self.scatter(pillar_enc, pillar_indices_seq[t],
                               num_pillars_seq[t])

            # Warp past frames to current ego frame
            if t < self.num_frames - 1:
                ego_t = ego_transforms[:, t]  # (B, 4, 4)
                bev = self.warp_bev(bev, ego_t)

            bev_features_list.append(bev)

        # Temporal fusion
        if self.fusion_method == "concat_conv":
            # Concatenate along channel axis: (B, H, W, C*T)
            fused = tf.concat(bev_features_list, axis=-1)
            fused = self.temporal_conv(fused, training=training)

        elif self.fusion_method == "attention":
            # Reshape for attention: (B*H*W, T, C)
            B = tf.shape(bev_features_list[0])[0]
            H, W = self.grid_size
            C = tf.shape(bev_features_list[0])[-1]

            # Stack: (B, H, W, T, C)
            stacked = tf.stack(bev_features_list, axis=3)
            # Reshape to (B*H*W, T, C)
            stacked = tf.reshape(stacked, [B * H * W, self.num_frames, C])

            # Query is the current frame (last timestep)
            current = stacked[:, -1:, :]  # (B*H*W, 1, C)

            # Multi-head attention: query=current, key=value=all frames
            attn_out = self.temporal_attn(
                query=current, value=stacked, key=stacked,
                training=training
            )  # (B*H*W, 1, C)

            # Residual + LayerNorm
            attn_out = self.temporal_norm(attn_out + current)

            # Reshape back: (B, H, W, C)
            fused = tf.reshape(attn_out[:, 0, :], [B, H, W, C])

        elif self.fusion_method == "gru":
            # Reshape for GRU: (B*H*W, T, C)
            B = tf.shape(bev_features_list[0])[0]
            H, W = self.grid_size
            C = tf.shape(bev_features_list[0])[-1]

            # Stack: (B, H, W, T, C)
            stacked = tf.stack(bev_features_list, axis=3)
            # Reshape to (B*H*W, T, C)
            stacked = tf.reshape(stacked, [B * H * W, self.num_frames, C])

            # GRU over temporal dimension
            output = self.temporal_gru(stacked, training=training)  # (B*H*W, T, C)

            # Take last timestep output
            fused = output[:, -1, :]  # (B*H*W, C)
            fused = tf.reshape(fused, [B, H, W, C])

        else:
            # No fusion, just use current frame
            fused = bev_features_list[-1]

        # Backbone
        features = self.backbone(fused, training=training)

        # Prediction heads
        outputs = {'occupancy': self.occ_head(features)}
        if self.sem_head is not None:
            outputs['semantic'] = self.sem_head(features, training=training)

        return outputs


def build_model(config):
    """Factory function to build the appropriate model.

    Args:
        config: dict with 'grid' and 'model' configuration.
            config['model']['type'] selects the model:
                - 'pillar_occ_net': single-frame PillarOccNet
                - 'temporal_pillar_occ_net': multi-frame TemporalPillarOccNet

    Returns:
        tf.keras.Model instance

    Example config:
        config = {
            "grid": {
                "grid_size": [200, 200],
                "cell_size": 0.5,
                "x_range": [-50, 50],
                "y_range": [-50, 50]
            },
            "model": {
                "type": "pillar_occ_net",
                "pillar": {
                    "input_features": 9,
                    "pillar_features": 64,
                    "max_points_per_pillar": 20,
                    "max_pillars": 10000
                },
                "backbone": {
                    "encoder_channels": [64, 128, 256, 512],
                    "decoder_channels": [256, 128, 64]
                },
                "heads": {"semantics": {"enabled": True, "num_classes": 5}},
                "temporal": {
                    "num_frames": 3,
                    "fusion_method": "concat_conv",
                    "temporal_conv_channels": 64
                }
            }
        }
    """
    model_type = config["model"]["type"]

    if model_type == "pillar_occ_net":
        return PillarOccNet(config, name='pillar_occ_net')
    elif model_type == "temporal_pillar_occ_net":
        return TemporalPillarOccNet(config, name='temporal_pillar_occ_net')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
