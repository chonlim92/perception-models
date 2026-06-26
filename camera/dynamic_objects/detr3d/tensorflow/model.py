"""
DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries
TensorFlow 2 Implementation

Reference: https://arxiv.org/abs/2110.06922
"""

import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment


class FPN(tf.keras.layers.Layer):
    """Feature Pyramid Network with lateral connections and top-down pathway."""

    def __init__(self, out_channels=256, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def build(self, input_shape):
        num_levels = len(input_shape)
        self.lateral_convs = []
        self.output_convs = []
        for i in range(num_levels):
            lateral = tf.keras.layers.Conv2D(
                self.out_channels, 1, padding='same',
                kernel_initializer='he_normal', name=f'lateral_{i}'
            )
            output = tf.keras.layers.Conv2D(
                self.out_channels, 3, padding='same',
                kernel_initializer='he_normal', name=f'output_{i}'
            )
            self.lateral_convs.append(lateral)
            self.output_convs.append(output)

    def call(self, features):
        """
        Args:
            features: list of feature maps from backbone [C2, C3, C4, C5]
        Returns:
            list of FPN feature maps [P2, P3, P4, P5]
        """
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 2, -1, -1):
            h, w = tf.shape(laterals[i])[1], tf.shape(laterals[i])[2]
            upsampled = tf.image.resize(laterals[i + 1], [h, w], method='bilinear')
            laterals[i] = laterals[i] + upsampled

        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs


class BilinearFeatureSampler(tf.keras.layers.Layer):
    """Sample features from multi-view feature maps at projected 2D locations
    using differentiable bilinear interpolation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, feature_maps, sample_points):
        """
        Args:
            feature_maps: (B, num_cams, H, W, C) multi-view feature maps
            sample_points: (B, num_cams, num_queries, 2) normalized [0,1] 2D coordinates
        Returns:
            sampled_features: (B, num_queries, num_cams, C)
        """
        batch_size = tf.shape(feature_maps)[0]
        num_cams = tf.shape(feature_maps)[1]
        h = tf.shape(feature_maps)[2]
        w = tf.shape(feature_maps)[3]
        channels = tf.shape(feature_maps)[4]
        num_queries = tf.shape(sample_points)[2]

        h_f = tf.cast(h, tf.float32)
        w_f = tf.cast(w, tf.float32)

        x = sample_points[..., 0] * (w_f - 1.0)
        y = sample_points[..., 1] * (h_f - 1.0)

        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, w - 1)
        x1 = tf.clip_by_value(x1, 0, w - 1)
        y0 = tf.clip_by_value(y0, 0, h - 1)
        y1 = tf.clip_by_value(y1, 0, h - 1)

        x = tf.clip_by_value(x, 0.0, w_f - 1.0)
        y = tf.clip_by_value(y, 0.0, h_f - 1.0)

        wa = (tf.cast(x1, tf.float32) - x) * (tf.cast(y1, tf.float32) - y)
        wb = (tf.cast(x1, tf.float32) - x) * (y - tf.cast(y0, tf.float32))
        wc = (x - tf.cast(x0, tf.float32)) * (tf.cast(y1, tf.float32) - y)
        wd = (x - tf.cast(x0, tf.float32)) * (y - tf.cast(y0, tf.float32))

        wa = tf.expand_dims(wa, -1)
        wb = tf.expand_dims(wb, -1)
        wc = tf.expand_dims(wc, -1)
        wd = tf.expand_dims(wd, -1)

        batch_idx = tf.range(batch_size)
        batch_idx = tf.reshape(batch_idx, [batch_size, 1, 1])
        batch_idx = tf.broadcast_to(batch_idx, [batch_size, num_cams, num_queries])

        cam_idx = tf.range(num_cams)
        cam_idx = tf.reshape(cam_idx, [1, num_cams, 1])
        cam_idx = tf.broadcast_to(cam_idx, [batch_size, num_cams, num_queries])

        def gather_pixels(yi, xi):
            indices = tf.stack([batch_idx, cam_idx, yi, xi], axis=-1)
            return tf.gather_nd(feature_maps, indices)

        Ia = gather_pixels(y0, x0)
        Ib = gather_pixels(y1, x0)
        Ic = gather_pixels(y0, x1)
        Id = gather_pixels(y1, x1)

        sampled = wa * Ia + wb * Ib + wc * Ic + wd * Id
        sampled = tf.transpose(sampled, [0, 2, 1, 3])
        return sampled


class ReferencePointProjection(tf.keras.layers.Layer):
    """Project 3D reference points to 2D image coordinates using camera parameters."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, reference_points_3d, intrinsics, extrinsics):
        """
        Args:
            reference_points_3d: (B, num_queries, 3) 3D reference points in world frame
            intrinsics: (B, num_cams, 3, 3) camera intrinsic matrices
            extrinsics: (B, num_cams, 4, 4) world-to-camera transformation matrices
        Returns:
            projected_2d: (B, num_cams, num_queries, 2) normalized 2D coords
            valid_mask: (B, num_cams, num_queries) boolean mask for valid projections
        """
        batch_size = tf.shape(reference_points_3d)[0]
        num_queries = tf.shape(reference_points_3d)[1]
        num_cams = tf.shape(intrinsics)[1]

        ones = tf.ones([batch_size, num_queries, 1], dtype=tf.float32)
        points_homo = tf.concat([reference_points_3d, ones], axis=-1)

        points_homo = tf.expand_dims(points_homo, 1)
        points_homo = tf.broadcast_to(points_homo, [batch_size, num_cams, num_queries, 4])

        rotation = extrinsics[:, :, :3, :3]
        translation = extrinsics[:, :, :3, 3:]

        points_3d_world = tf.expand_dims(reference_points_3d, 1)
        points_3d_world = tf.broadcast_to(points_3d_world, [batch_size, num_cams, num_queries, 3])

        points_cam = tf.einsum('bcij,bcqj->bcqi', rotation, points_3d_world) + \
                     tf.transpose(translation, [0, 1, 3, 2])

        depth = points_cam[..., 2:3]
        depth_clipped = tf.maximum(depth, 1e-5)

        points_2d_homo = points_cam / depth_clipped

        points_2d_homo_3 = points_2d_homo[..., :3]
        projected = tf.einsum('bcij,bcqj->bcqi', intrinsics, points_2d_homo_3)

        u = projected[..., 0:1]
        v = projected[..., 1:2]
        projected_2d = tf.concat([u, v], axis=-1)

        img_h = intrinsics[:, :, 1, 2] * 2.0
        img_w = intrinsics[:, :, 0, 2] * 2.0

        img_h = tf.reshape(img_h, [batch_size, num_cams, 1, 1])
        img_w = tf.reshape(img_w, [batch_size, num_cams, 1, 1])

        normalized_2d = tf.concat([
            projected_2d[..., 0:1] / tf.maximum(img_w, 1.0),
            projected_2d[..., 1:2] / tf.maximum(img_h, 1.0)
        ], axis=-1)

        valid_mask = tf.logical_and(
            tf.logical_and(normalized_2d[..., 0] >= 0.0, normalized_2d[..., 0] <= 1.0),
            tf.logical_and(normalized_2d[..., 1] >= 0.0, normalized_2d[..., 1] <= 1.0)
        )
        valid_mask = tf.logical_and(valid_mask, depth[..., 0] > 0.0)

        normalized_2d = tf.clip_by_value(normalized_2d, 0.0, 1.0)

        return normalized_2d, valid_mask


class DETR3DTransformerDecoderLayer(tf.keras.layers.Layer):
    """Single Transformer decoder layer with self-attention,
    feature-sampling cross-attention, and FFN."""

    def __init__(self, d_model=256, num_heads=8, dim_feedforward=2048,
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate, name='self_attn'
        )
        self.self_attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='self_attn_norm')
        self.self_attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.cross_attn_proj = tf.keras.layers.Dense(self.d_model, name='cross_attn_proj')
        self.cross_attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='cross_attn_norm')
        self.cross_attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.cam_attention_weights = tf.keras.layers.Dense(1, name='cam_attention_weights')

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dim_feedforward, activation='relu', name='ffn_linear1'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.d_model, name='ffn_linear2'),
            tf.keras.layers.Dropout(self.dropout_rate),
        ], name='ffn')
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='ffn_norm')

    def call(self, query, sampled_features, valid_mask, training=False):
        """
        Args:
            query: (B, num_queries, d_model) object queries
            sampled_features: (B, num_queries, num_cams, d_model) features from projection
            valid_mask: (B, num_cams, num_queries) validity mask
        Returns:
            query: (B, num_queries, d_model) updated queries
        """
        residual = query
        query_normed = self.self_attn_norm(query)
        query = residual + self.self_attn_dropout(
            self.self_attn(query_normed, query_normed, query_normed, training=training),
            training=training
        )

        residual = query
        query_normed = self.cross_attn_norm(query)

        valid_mask_t = tf.transpose(valid_mask, [0, 2, 1])
        mask_weight = tf.cast(tf.expand_dims(valid_mask_t, -1), tf.float32)

        cam_weights = self.cam_attention_weights(sampled_features)
        cam_weights = cam_weights * mask_weight + (1.0 - mask_weight) * (-1e9)
        cam_weights = tf.nn.softmax(cam_weights, axis=2)

        cross_attn_output = tf.reduce_sum(sampled_features * cam_weights, axis=2)
        cross_attn_output = self.cross_attn_proj(cross_attn_output)

        query = residual + self.cross_attn_dropout(cross_attn_output, training=training)

        residual = query
        query_normed = self.ffn_norm(query)
        query = residual + self.ffn(query_normed, training=training)

        return query


class DETR3DTransformerDecoder(tf.keras.layers.Layer):
    """Stack of DETR3D Transformer decoder layers."""

    def __init__(self, num_layers=6, d_model=256, num_heads=8,
                 dim_feedforward=2048, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.layers_list = [
            DETR3DTransformerDecoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout_rate=self.dropout_rate,
                name=f'decoder_layer_{i}'
            )
            for i in range(self.num_layers)
        ]

    def call(self, query, sampled_features, valid_mask, training=False):
        """
        Args:
            query: (B, num_queries, d_model)
            sampled_features: (B, num_queries, num_cams, d_model)
            valid_mask: (B, num_cams, num_queries)
        Returns:
            intermediate: list of (B, num_queries, d_model) for auxiliary losses
        """
        intermediate = []
        for layer in self.layers_list:
            query = layer(query, sampled_features, valid_mask, training=training)
            intermediate.append(query)
        return intermediate


class DetectionHead(tf.keras.layers.Layer):
    """Classification and regression heads for 3D object detection."""

    def __init__(self, num_classes=10, num_reg_params=10, d_model=256, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_reg_params = num_reg_params
        self.d_model = d_model

    def build(self, input_shape):
        self.cls_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.d_model, activation='relu', name='cls_fc1'),
            tf.keras.layers.Dense(self.d_model, activation='relu', name='cls_fc2'),
            tf.keras.layers.Dense(self.num_classes, name='cls_out'),
        ], name='cls_head')

        self.reg_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.d_model, activation='relu', name='reg_fc1'),
            tf.keras.layers.Dense(self.d_model, activation='relu', name='reg_fc2'),
            tf.keras.layers.Dense(self.num_reg_params, name='reg_out'),
        ], name='reg_head')

    def call(self, query):
        """
        Args:
            query: (B, num_queries, d_model)
        Returns:
            cls_logits: (B, num_queries, num_classes)
            reg_preds: (B, num_queries, num_reg_params)
                       [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
        """
        cls_logits = self.cls_head(query)
        reg_preds = self.reg_head(query)
        return cls_logits, reg_preds


class HungarianMatcher:
    """Hungarian matcher for bipartite matching between predictions and ground truth."""

    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def compute_matching(self, cls_logits, reg_preds, gt_labels, gt_boxes):
        """
        Compute optimal bipartite matching using Hungarian algorithm.

        Args:
            cls_logits: (num_queries, num_classes) predicted class logits
            reg_preds: (num_queries, num_reg_params) predicted regression params
            gt_labels: (num_gt,) ground truth class labels
            gt_boxes: (num_gt, num_reg_params) ground truth boxes

        Returns:
            matched_row_indices: indices into predictions
            matched_col_indices: indices into ground truth
        """
        num_queries = cls_logits.shape[0]
        num_gt = gt_labels.shape[0]

        if num_gt == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        cls_probs = tf.nn.sigmoid(cls_logits).numpy()
        gt_labels_np = gt_labels.numpy().astype(np.int64)

        cost_class = -cls_probs[:, gt_labels_np]

        pred_centers = reg_preds[:, :3].numpy()
        gt_centers = gt_boxes[:, :3].numpy()
        cost_bbox = np.sum(np.abs(pred_centers[:, None, :] - gt_centers[None, :, :]), axis=-1)

        pred_sizes = reg_preds[:, 3:6].numpy()
        gt_sizes = gt_boxes[:, 3:6].numpy()
        cost_size = np.sum(np.abs(pred_sizes[:, None, :] - gt_sizes[None, :, :]), axis=-1)

        cost_matrix = (self.cost_class * cost_class +
                       self.cost_bbox * (cost_bbox + cost_size))

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        return row_indices, col_indices


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal loss for classification.

    Args:
        logits: (B, num_queries, num_classes) predicted logits
        targets: (B, num_queries, num_classes) one-hot targets
    Returns:
        loss: scalar focal loss
    """
    probs = tf.nn.sigmoid(logits)
    ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)

    p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    focal_weight = alpha_t * tf.pow(1.0 - p_t, gamma)

    loss = focal_weight * ce_loss
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


def l1_loss(pred, target, mask=None):
    """
    L1 loss for regression.

    Args:
        pred: (B, num_queries, D)
        target: (B, num_queries, D)
        mask: (B, num_queries) optional mask
    Returns:
        loss: scalar L1 loss
    """
    loss = tf.abs(pred - target)
    if mask is not None:
        mask = tf.expand_dims(mask, -1)
        loss = loss * tf.cast(mask, tf.float32)
        num_pos = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
        return tf.reduce_sum(loss) / num_pos
    return tf.reduce_mean(loss)


class DETR3D(tf.keras.Model):
    """
    DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries.

    Architecture:
        1. ResNet101 backbone extracts multi-scale features from each camera view
        2. FPN combines multi-scale features into 256-channel feature maps
        3. Learnable 3D reference points are projected to each camera view
        4. Bilinear sampling extracts features at projected locations
        5. Transformer decoder refines object queries
        6. Detection heads output class predictions and 3D box parameters
    """

    def __init__(self, num_classes=10, num_queries=900, d_model=256,
                 num_heads=8, num_decoder_layers=6, dim_feedforward=2048,
                 dropout_rate=0.1, num_reg_params=10, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.num_reg_params = num_reg_params

        self.backbone = self._build_backbone()
        self.fpn = FPN(out_channels=d_model, name='fpn')

        self.input_proj = tf.keras.layers.Conv2D(
            d_model, 1, padding='same', name='input_proj'
        )

        self.reference_points = tf.Variable(
            tf.random.uniform([num_queries, 3], -1.0, 1.0),
            trainable=True, name='reference_points'
        )

        self.query_embedding = tf.Variable(
            tf.random.normal([num_queries, d_model], stddev=0.02),
            trainable=True, name='query_embedding'
        )

        self.projection_layer = ReferencePointProjection(name='ref_point_proj')
        self.feature_sampler = BilinearFeatureSampler(name='bilinear_sampler')

        self.feature_proj = tf.keras.layers.Dense(d_model, name='feature_proj')

        self.decoder = DETR3DTransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            name='decoder'
        )

        self.detection_head = DetectionHead(
            num_classes=num_classes,
            num_reg_params=num_reg_params,
            d_model=d_model,
            name='detection_head'
        )

        self.matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0)

    def _build_backbone(self):
        """Build ResNet101 backbone and extract multi-scale feature layers."""
        base_model = tf.keras.applications.ResNet101(
            include_top=False, weights='imagenet',
            input_shape=(None, None, 3)
        )
        layer_names = [
            'conv2_block3_out',
            'conv3_block4_out',
            'conv4_block23_out',
            'conv5_block3_out',
        ]
        outputs = [base_model.get_layer(name).output for name in layer_names]
        return tf.keras.Model(inputs=base_model.input, outputs=outputs, name='resnet101_backbone')

    def extract_features(self, images, training=False):
        """
        Extract multi-scale features from multi-view images.

        Args:
            images: (B, num_cams, H, W, 3)
        Returns:
            features: (B, num_cams, H', W', d_model) combined feature map
        """
        batch_size = tf.shape(images)[0]
        num_cams = tf.shape(images)[1]
        h = tf.shape(images)[2]
        w = tf.shape(images)[3]

        images_flat = tf.reshape(images, [-1, h, w, 3])

        images_preprocessed = tf.keras.applications.resnet.preprocess_input(images_flat)

        backbone_features = self.backbone(images_preprocessed, training=training)

        fpn_features = self.fpn(backbone_features)

        feat = self.input_proj(fpn_features[1])

        feat_h = tf.shape(feat)[1]
        feat_w = tf.shape(feat)[2]
        feat = tf.reshape(feat, [batch_size, num_cams, feat_h, feat_w, self.d_model])

        return feat

    def call(self, inputs, training=False):
        """
        Full forward pass.

        Args:
            inputs: dict with keys:
                'images': (B, num_cams, H, W, 3) multi-view images
                'intrinsics': (B, num_cams, 3, 3) camera intrinsic matrices
                'extrinsics': (B, num_cams, 4, 4) camera extrinsic matrices
        Returns:
            outputs: dict with keys:
                'cls_logits': (B, num_queries, num_classes)
                'reg_preds': (B, num_queries, num_reg_params)
                'aux_outputs': list of dicts for intermediate decoder outputs
        """
        images = inputs['images']
        intrinsics = inputs['intrinsics']
        extrinsics = inputs['extrinsics']

        batch_size = tf.shape(images)[0]

        features = self.extract_features(images, training=training)

        ref_points = tf.nn.sigmoid(self.reference_points)
        ref_points_batch = tf.expand_dims(ref_points, 0)
        ref_points_batch = tf.broadcast_to(ref_points_batch, [batch_size, self.num_queries, 3])

        ref_points_scaled = ref_points_batch * tf.constant([[100.0, 100.0, 8.0]]) - \
                           tf.constant([[50.0, 50.0, 2.0]])

        projected_2d, valid_mask = self.projection_layer(
            ref_points_scaled, intrinsics, extrinsics
        )

        sampled_features = self.feature_sampler(features, projected_2d)
        sampled_features = self.feature_proj(sampled_features)

        query = tf.expand_dims(self.query_embedding, 0)
        query = tf.broadcast_to(query, [batch_size, self.num_queries, self.d_model])
        query = tf.identity(query)

        intermediate = self.decoder(query, sampled_features, valid_mask, training=training)

        outputs_list = []
        for inter_query in intermediate:
            cls_logits, reg_preds = self.detection_head(inter_query)
            reg_preds_with_ref = reg_preds + tf.concat([
                ref_points_scaled,
                tf.zeros([batch_size, self.num_queries, self.num_reg_params - 3])
            ], axis=-1)
            outputs_list.append({
                'cls_logits': cls_logits,
                'reg_preds': reg_preds_with_ref,
            })

        final_output = outputs_list[-1]
        final_output['aux_outputs'] = outputs_list[:-1]

        return final_output

    def compute_loss(self, predictions, gt_labels_list, gt_boxes_list):
        """
        Compute total loss with Hungarian matching.

        Args:
            predictions: dict from forward pass
            gt_labels_list: list of (num_gt,) tensors per sample
            gt_boxes_list: list of (num_gt, num_reg_params) tensors per sample
        Returns:
            total_loss: scalar loss
            loss_dict: dict of individual loss components
        """
        cls_logits = predictions['cls_logits']
        reg_preds = predictions['reg_preds']
        batch_size = cls_logits.shape[0]

        total_cls_loss = 0.0
        total_reg_loss = 0.0

        for b in range(batch_size):
            cls_b = cls_logits[b]
            reg_b = reg_preds[b]
            gt_labels_b = gt_labels_list[b]
            gt_boxes_b = gt_boxes_list[b]

            row_indices, col_indices = self.matcher.compute_matching(
                cls_b, reg_b, gt_labels_b, gt_boxes_b
            )

            cls_targets = tf.zeros([self.num_queries, self.num_classes], dtype=tf.float32)
            reg_targets = tf.zeros([self.num_queries, self.num_reg_params], dtype=tf.float32)
            reg_mask = tf.zeros([self.num_queries], dtype=tf.float32)

            if len(row_indices) > 0:
                indices_2d = []
                values = []
                for r, c in zip(row_indices, col_indices):
                    label = int(gt_labels_b[c].numpy())
                    indices_2d.append([int(r), label])
                    values.append(1.0)

                if indices_2d:
                    cls_targets = tf.tensor_scatter_nd_update(
                        cls_targets,
                        tf.constant(indices_2d, dtype=tf.int32),
                        tf.constant(values, dtype=tf.float32)
                    )

                matched_gt_boxes = tf.gather(gt_boxes_b, col_indices)
                row_indices_tf = tf.constant(row_indices, dtype=tf.int32)
                row_indices_2d = tf.expand_dims(row_indices_tf, 1)
                reg_targets = tf.tensor_scatter_nd_update(
                    reg_targets, row_indices_2d, matched_gt_boxes
                )
                mask_updates = tf.ones([len(row_indices)], dtype=tf.float32)
                reg_mask = tf.tensor_scatter_nd_update(
                    reg_mask,
                    tf.expand_dims(row_indices_tf, 1),
                    mask_updates
                )

            cls_loss_b = focal_loss(
                tf.expand_dims(cls_b, 0),
                tf.expand_dims(cls_targets, 0)
            )
            reg_loss_b = l1_loss(
                tf.expand_dims(reg_b, 0),
                tf.expand_dims(reg_targets, 0),
                tf.expand_dims(reg_mask, 0)
            )

            total_cls_loss += cls_loss_b
            total_reg_loss += reg_loss_b

        total_cls_loss /= float(batch_size)
        total_reg_loss /= float(batch_size)

        aux_loss = 0.0
        if 'aux_outputs' in predictions:
            for aux_pred in predictions['aux_outputs']:
                for b in range(batch_size):
                    cls_b = aux_pred['cls_logits'][b]
                    reg_b = aux_pred['reg_preds'][b]
                    gt_labels_b = gt_labels_list[b]
                    gt_boxes_b = gt_boxes_list[b]

                    row_indices, col_indices = self.matcher.compute_matching(
                        cls_b, reg_b, gt_labels_b, gt_boxes_b
                    )

                    cls_targets = tf.zeros([self.num_queries, self.num_classes], dtype=tf.float32)
                    reg_targets = tf.zeros([self.num_queries, self.num_reg_params], dtype=tf.float32)
                    reg_mask = tf.zeros([self.num_queries], dtype=tf.float32)

                    if len(row_indices) > 0:
                        indices_2d = []
                        values = []
                        for r, c in zip(row_indices, col_indices):
                            label = int(gt_labels_b[c].numpy())
                            indices_2d.append([int(r), label])
                            values.append(1.0)
                        if indices_2d:
                            cls_targets = tf.tensor_scatter_nd_update(
                                cls_targets,
                                tf.constant(indices_2d, dtype=tf.int32),
                                tf.constant(values, dtype=tf.float32)
                            )

                        matched_gt_boxes = tf.gather(gt_boxes_b, col_indices)
                        row_indices_tf = tf.constant(row_indices, dtype=tf.int32)
                        row_indices_2d = tf.expand_dims(row_indices_tf, 1)
                        reg_targets = tf.tensor_scatter_nd_update(
                            reg_targets, row_indices_2d, matched_gt_boxes
                        )
                        mask_updates = tf.ones([len(row_indices)], dtype=tf.float32)
                        reg_mask = tf.tensor_scatter_nd_update(
                            reg_mask,
                            tf.expand_dims(row_indices_tf, 1),
                            mask_updates
                        )

                    cls_loss_b = focal_loss(
                        tf.expand_dims(cls_b, 0),
                        tf.expand_dims(cls_targets, 0)
                    )
                    reg_loss_b = l1_loss(
                        tf.expand_dims(reg_b, 0),
                        tf.expand_dims(reg_targets, 0),
                        tf.expand_dims(reg_mask, 0)
                    )
                    aux_loss += (cls_loss_b + reg_loss_b * 0.25)

            aux_loss /= float(batch_size * len(predictions['aux_outputs']))

        total_loss = total_cls_loss + total_reg_loss * 0.25 + aux_loss * 0.5

        loss_dict = {
            'cls_loss': total_cls_loss,
            'reg_loss': total_reg_loss,
            'aux_loss': aux_loss,
            'total_loss': total_loss,
        }

        return total_loss, loss_dict


def build_detr3d(num_classes=10, num_queries=900, d_model=256,
                 num_heads=8, num_decoder_layers=6):
    """Factory function to build DETR3D model."""
    model = DETR3D(
        num_classes=num_classes,
        num_queries=num_queries,
        d_model=d_model,
        num_heads=num_heads,
        num_decoder_layers=num_decoder_layers,
    )
    return model
