"""
PointNet++ (PointNet Set Abstraction) for 3D Point Cloud Processing.

Complete TensorFlow 2 / Keras implementation including:
- Set Abstraction (SA) layers with single-scale and multi-scale grouping
- Feature Propagation (FP) layers for upsampling
- Classification, Detection, and Segmentation model heads

Reference: Qi et al., "PointNet++: Deep Hierarchical Feature Learning on
Point Sets in a Metric Space", NeurIPS 2017.
"""

import tensorflow as tf
import numpy as np


# ===========================================================================
# Utility Functions
# ===========================================================================


@tf.function
def square_distance(src, dst):
    """Compute pairwise squared Euclidean distances between two point sets.

    Args:
        src: (B, N, 3) source point coordinates.
        dst: (B, M, 3) destination point coordinates.

    Returns:
        dist: (B, N, M) squared distances where dist[b, i, j] = ||src[b,i] - dst[b,j]||^2.
    """
    # ||src - dst||^2 = ||src||^2 + ||dst||^2 - 2 * src . dst^T
    B = tf.shape(src)[0]
    N = tf.shape(src)[1]
    M = tf.shape(dst)[1]

    # (B, N, 1)
    src_sq = tf.reduce_sum(src ** 2, axis=-1, keepdims=True)
    # (B, 1, M)
    dst_sq = tf.reduce_sum(dst ** 2, axis=-1, keepdims=True)
    dst_sq = tf.transpose(dst_sq, perm=[0, 2, 1])

    # (B, N, M)
    inner = tf.matmul(src, tf.transpose(dst, perm=[0, 2, 1]))

    dist = src_sq + dst_sq - 2.0 * inner
    return dist


@tf.function
def farthest_point_sampling(xyz, npoint):
    """Iteratively select the farthest points from a point cloud.

    Args:
        xyz: (B, N, 3) input point coordinates.
        npoint: int, number of points to sample.

    Returns:
        centroids_idx: (B, npoint) indices of selected points.
    """
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]

    centroids = tf.TensorArray(dtype=tf.int32, size=npoint, dynamic_size=False)
    distance = tf.fill([B, N], 1e10)

    # Start from a random point (index 0 for determinism in compiled graph)
    farthest = tf.zeros([B], dtype=tf.int32)

    batch_indices = tf.range(B, dtype=tf.int32)

    for i in tf.range(npoint):
        centroids = centroids.write(i, farthest)

        # Gather the farthest point coordinates: (B, 3)
        indices = tf.stack([batch_indices, farthest], axis=1)
        centroid = tf.gather_nd(xyz, indices)  # (B, 3)
        centroid = tf.expand_dims(centroid, axis=1)  # (B, 1, 3)

        # Compute distance from all points to this centroid: (B, N)
        dist = tf.reduce_sum((xyz - centroid) ** 2, axis=-1)

        # Update minimum distances
        distance = tf.minimum(distance, dist)

        # Select point with maximum distance
        farthest = tf.argmax(distance, axis=-1, output_type=tf.int32)

    # (npoint, B) -> (B, npoint)
    centroids_idx = centroids.stack()
    centroids_idx = tf.transpose(centroids_idx, perm=[1, 0])
    return centroids_idx


@tf.function
def index_points(points, idx):
    """Gather points by indices.

    Args:
        points: (B, N, C) input point features.
        idx: (B, S) or (B, S, K) index tensor.

    Returns:
        gathered: (B, S, C) or (B, S, K, C) gathered point features.
    """
    B = tf.shape(points)[0]

    # Build batch indices
    batch_indices = tf.range(B, dtype=tf.int32)

    idx_shape = tf.shape(idx)
    idx_rank = len(idx.shape)

    if idx_rank == 2:
        # idx: (B, S)
        S = idx_shape[1]
        batch_indices = tf.reshape(batch_indices, [B, 1])
        batch_indices = tf.tile(batch_indices, [1, S])
        full_indices = tf.stack([batch_indices, idx], axis=-1)
    else:
        # idx: (B, S, K)
        S = idx_shape[1]
        K = idx_shape[2]
        batch_indices = tf.reshape(batch_indices, [B, 1, 1])
        batch_indices = tf.tile(batch_indices, [1, S, K])
        full_indices = tf.stack([batch_indices, idx], axis=-1)

    gathered = tf.gather_nd(points, full_indices)
    return gathered


@tf.function
def ball_query(radius, nsample, xyz, new_xyz):
    """Find all points within a given radius around each query point.

    For each query point in new_xyz, finds up to nsample points in xyz
    that lie within the ball of the given radius. If fewer than nsample
    points are found, the first found point is repeated to fill.

    Args:
        radius: float, ball radius.
        nsample: int, maximum number of neighbors.
        xyz: (B, N, 3) all point coordinates.
        new_xyz: (B, S, 3) query point coordinates (centroids).

    Returns:
        group_idx: (B, S, nsample) indices of grouped points.
    """
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]
    S = tf.shape(new_xyz)[1]

    # Compute pairwise distances: (B, S, N)
    dists = square_distance(new_xyz, xyz)

    radius_sq = radius * radius

    # Mask points outside radius with a large index (N, which will be clipped)
    group_idx = tf.where(
        dists <= radius_sq,
        tf.tile(
            tf.reshape(tf.range(N, dtype=tf.int32), [1, 1, N]),
            [B, S, 1]
        ),
        tf.fill([B, S, N], N)
    )

    # Sort so valid indices come first
    group_idx = tf.sort(group_idx, axis=-1)

    # Take first nsample
    group_idx = group_idx[:, :, :nsample]

    # Replace any remaining N values with the first valid index in that group
    first_idx = group_idx[:, :, 0:1]
    first_idx = tf.tile(first_idx, [1, 1, nsample])
    group_idx = tf.where(group_idx >= N, first_idx, group_idx)

    return group_idx


@tf.function
def sample_and_group(npoint, radius, nsample, xyz, points):
    """Sample centroids via FPS and group local neighborhoods.

    Args:
        npoint: int, number of centroids to sample.
        radius: float, ball query radius.
        nsample: int, number of neighbors per group.
        xyz: (B, N, 3) point coordinates.
        points: (B, N, C) point features, or None.

    Returns:
        new_xyz: (B, npoint, 3) centroid coordinates.
        new_points: (B, npoint, nsample, 3+C) grouped and centered features.
    """
    fps_idx = farthest_point_sampling(xyz, npoint)  # (B, npoint)
    new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)

    idx = ball_query(radius, nsample, xyz, new_xyz)  # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)

    # Center the coordinates
    grouped_xyz_centered = grouped_xyz - tf.expand_dims(new_xyz, axis=2)

    if points is not None:
        grouped_points = index_points(points, idx)  # (B, npoint, nsample, C)
        new_points = tf.concat([grouped_xyz_centered, grouped_points], axis=-1)
    else:
        new_points = grouped_xyz_centered

    return new_xyz, new_points


@tf.function
def sample_and_group_all(xyz, points):
    """Group all points into a single set (for global feature extraction).

    Args:
        xyz: (B, N, 3) point coordinates.
        points: (B, N, C) point features, or None.

    Returns:
        new_xyz: (B, 1, 3) origin as the single centroid.
        new_points: (B, 1, N, 3+C) all points grouped together.
    """
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]

    new_xyz = tf.zeros([B, 1, 3], dtype=xyz.dtype)
    grouped_xyz = tf.expand_dims(xyz, axis=1)  # (B, 1, N, 3)

    if points is not None:
        grouped_points = tf.expand_dims(points, axis=1)  # (B, 1, N, C)
        new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points


# ===========================================================================
# PointNet Set Abstraction Layer (Single-Scale Grouping)
# ===========================================================================


class PointNetSetAbstraction(tf.keras.layers.Layer):
    """PointNet Set Abstraction module with single-scale grouping.

    Performs: FPS -> Ball Query -> Shared MLP -> Max Pooling.

    Args:
        npoint: int, number of centroids to sample.
        radius: float, ball query radius.
        nsample: int, maximum number of neighbors per group.
        mlp_list: list of int, output channels for each MLP layer.
        group_all: bool, if True group all points (ignore npoint/radius/nsample).
    """

    def __init__(self, npoint, radius, nsample, mlp_list, group_all=False, **kwargs):
        super(PointNetSetAbstraction, self).__init__(**kwargs)
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_list = mlp_list
        self.group_all = group_all

        self.conv_layers = []
        self.bn_layers = []

        for i, out_channels in enumerate(mlp_list):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    out_channels, kernel_size=1, strides=1,
                    padding='valid', use_bias=False,
                    name=f'sa_conv_{i}'
                )
            )
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization(name=f'sa_bn_{i}')
            )

    def call(self, xyz, points, training=False):
        """Forward pass.

        Args:
            xyz: (B, N, 3) point coordinates.
            points: (B, N, C) point features, or None.
            training: bool, training mode flag.

        Returns:
            new_xyz: (B, npoint, 3) sampled centroid coordinates.
            new_points: (B, npoint, D) abstracted point features.
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )

        # new_points: (B, npoint, nsample, 3+C) or (B, 1, N, 3+C)
        # Reshape for Conv1D: merge batch and npoint dims
        B = tf.shape(new_points)[0]
        S = tf.shape(new_points)[1]
        K = tf.shape(new_points)[2]
        C = new_points.shape[-1] if new_points.shape[-1] is not None else tf.shape(new_points)[-1]

        # (B*S, K, C_in)
        new_points = tf.reshape(new_points, [B * S, K, -1])

        # Apply shared MLPs (Conv1D + BN + ReLU)
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            new_points = conv(new_points)
            new_points = bn(new_points, training=training)
            new_points = tf.nn.relu(new_points)

        # Max pooling over neighbors: (B*S, K, D) -> (B*S, D)
        new_points = tf.reduce_max(new_points, axis=1)

        # Reshape back: (B, S, D)
        D = tf.shape(new_points)[-1]
        new_points = tf.reshape(new_points, [B, S, D])

        return new_xyz, new_points

    def get_config(self):
        config = super().get_config()
        config.update({
            'npoint': self.npoint,
            'radius': self.radius,
            'nsample': self.nsample,
            'mlp_list': self.mlp_list,
            'group_all': self.group_all,
        })
        return config


# ===========================================================================
# PointNet Set Abstraction Layer (Multi-Scale Grouping)
# ===========================================================================


class PointNetSetAbstractionMsg(tf.keras.layers.Layer):
    """PointNet Set Abstraction module with Multi-Scale Grouping (MSG).

    Groups neighborhoods at multiple radii and concatenates features.

    Args:
        npoint: int, number of centroids to sample.
        radius_list: list of float, ball query radii for each scale.
        nsample_list: list of int, number of neighbors for each scale.
        mlp_lists: list of list of int, MLP channels for each scale.
    """

    def __init__(self, npoint, radius_list, nsample_list, mlp_lists, **kwargs):
        super(PointNetSetAbstractionMsg, self).__init__(**kwargs)
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp_lists = mlp_lists

        assert len(radius_list) == len(nsample_list) == len(mlp_lists)

        self.conv_blocks = []
        self.bn_blocks = []

        for scale_idx, mlp_list in enumerate(mlp_lists):
            convs = []
            bns = []
            for i, out_channels in enumerate(mlp_list):
                convs.append(
                    tf.keras.layers.Conv1D(
                        out_channels, kernel_size=1, strides=1,
                        padding='valid', use_bias=False,
                        name=f'msg_scale{scale_idx}_conv{i}'
                    )
                )
                bns.append(
                    tf.keras.layers.BatchNormalization(
                        name=f'msg_scale{scale_idx}_bn{i}'
                    )
                )
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def call(self, xyz, points, training=False):
        """Forward pass.

        Args:
            xyz: (B, N, 3) point coordinates.
            points: (B, N, C) point features, or None.
            training: bool, training mode flag.

        Returns:
            new_xyz: (B, npoint, 3) sampled centroid coordinates.
            new_points: (B, npoint, sum(mlp[-1] for each scale)) concatenated features.
        """
        fps_idx = farthest_point_sampling(xyz, self.npoint)  # (B, npoint)
        new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)

        B = tf.shape(xyz)[0]
        S = self.npoint

        multi_scale_features = []

        for scale_idx, (radius, nsample) in enumerate(
            zip(self.radius_list, self.nsample_list)
        ):
            # Ball query at this scale
            idx = ball_query(radius, nsample, xyz, new_xyz)  # (B, S, nsample)
            grouped_xyz = index_points(xyz, idx)  # (B, S, nsample, 3)
            grouped_xyz_centered = grouped_xyz - tf.expand_dims(new_xyz, axis=2)

            if points is not None:
                grouped_points = index_points(points, idx)  # (B, S, nsample, C)
                grouped_points = tf.concat(
                    [grouped_xyz_centered, grouped_points], axis=-1
                )
            else:
                grouped_points = grouped_xyz_centered

            # (B*S, nsample, C_in)
            K = tf.shape(grouped_points)[2]
            grouped_points = tf.reshape(grouped_points, [B * S, K, -1])

            # Apply MLPs for this scale
            for conv, bn in zip(
                self.conv_blocks[scale_idx], self.bn_blocks[scale_idx]
            ):
                grouped_points = conv(grouped_points)
                grouped_points = bn(grouped_points, training=training)
                grouped_points = tf.nn.relu(grouped_points)

            # Max pool: (B*S, nsample, D) -> (B*S, D)
            grouped_points = tf.reduce_max(grouped_points, axis=1)

            # Reshape: (B, S, D)
            D = tf.shape(grouped_points)[-1]
            grouped_points = tf.reshape(grouped_points, [B, S, D])

            multi_scale_features.append(grouped_points)

        # Concatenate features from all scales: (B, S, sum(D_scales))
        new_points = tf.concat(multi_scale_features, axis=-1)

        return new_xyz, new_points

    def get_config(self):
        config = super().get_config()
        config.update({
            'npoint': self.npoint,
            'radius_list': self.radius_list,
            'nsample_list': self.nsample_list,
            'mlp_lists': self.mlp_lists,
        })
        return config


# ===========================================================================
# PointNet Feature Propagation Layer
# ===========================================================================


class PointNetFeaturePropagation(tf.keras.layers.Layer):
    """PointNet Feature Propagation module for upsampling.

    Interpolates features from sparse points (xyz2) to dense points (xyz1)
    using inverse distance weighted averaging, then applies an MLP.

    Args:
        mlp_list: list of int, output channels for each MLP layer.
    """

    def __init__(self, mlp_list, **kwargs):
        super(PointNetFeaturePropagation, self).__init__(**kwargs)
        self.mlp_list = mlp_list

        self.conv_layers = []
        self.bn_layers = []

        for i, out_channels in enumerate(mlp_list):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    out_channels, kernel_size=1, strides=1,
                    padding='valid', use_bias=False,
                    name=f'fp_conv_{i}'
                )
            )
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization(name=f'fp_bn_{i}')
            )

    def call(self, xyz1, xyz2, points1, points2, training=False):
        """Forward pass: interpolate features from xyz2 to xyz1.

        Args:
            xyz1: (B, N, 3) dense point coordinates (target).
            xyz2: (B, S, 3) sparse point coordinates (source).
            points1: (B, N, C1) features at dense points (skip connection), or None.
            points2: (B, S, C2) features at sparse points to interpolate.
            training: bool, training mode flag.

        Returns:
            new_points: (B, N, D) interpolated and refined features.
        """
        B = tf.shape(xyz1)[0]
        N = tf.shape(xyz1)[1]
        S = tf.shape(xyz2)[1]

        if S == 1:
            # Only one source point: broadcast its features to all target points
            interpolated_points = tf.tile(points2, [1, N, 1])
        else:
            # Compute distances from each point in xyz1 to all points in xyz2
            dists = square_distance(xyz1, xyz2)  # (B, N, S)

            # Find k=3 nearest neighbors
            # Negate for top_k (which returns largest values)
            neg_dists = -dists
            _, knn_idx = tf.math.top_k(neg_dists, k=3)  # (B, N, 3)
            # Gather distances of k nearest neighbors
            knn_dists = tf.gather(dists, knn_idx, batch_dims=2)  # (B, N, 3)

            # Inverse distance weighting
            # Add small epsilon to avoid division by zero
            weights = 1.0 / (knn_dists + 1e-8)  # (B, N, 3)
            weights_sum = tf.reduce_sum(weights, axis=-1, keepdims=True)  # (B, N, 1)
            weights_normalized = weights / weights_sum  # (B, N, 3)

            # Gather features from k nearest source points
            knn_points = index_points(points2, knn_idx)  # (B, N, 3, C2)

            # Weighted sum: (B, N, 3, C2) * (B, N, 3, 1) -> sum -> (B, N, C2)
            weights_expanded = tf.expand_dims(weights_normalized, axis=-1)
            interpolated_points = tf.reduce_sum(
                knn_points * weights_expanded, axis=2
            )

        # Concatenate with skip connection features
        if points1 is not None:
            new_points = tf.concat([points1, interpolated_points], axis=-1)
        else:
            new_points = interpolated_points

        # Apply MLP (Conv1D + BN + ReLU)
        # new_points: (B, N, C)
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            new_points = conv(new_points)
            new_points = bn(new_points, training=training)
            new_points = tf.nn.relu(new_points)

        return new_points

    def get_config(self):
        config = super().get_config()
        config.update({
            'mlp_list': self.mlp_list,
        })
        return config


# ===========================================================================
# PointNet++ Classification Model
# ===========================================================================


class PointNetPPClassification(tf.keras.Model):
    """PointNet++ model for 3D point cloud classification.

    Architecture:
        SA1: 512 centroids, radius=0.2, 32 neighbors, MLP [64, 64, 128]
        SA2: 128 centroids, radius=0.4, 64 neighbors, MLP [128, 128, 256]
        SA3: group_all, MLP [256, 512, 1024] (global feature)
        FC: 512 -> 256 -> num_classes

    Args:
        num_classes: int, number of output classes.
    """

    def __init__(self, num_classes=10, **kwargs):
        super(PointNetPPClassification, self).__init__(**kwargs)
        self.num_classes = num_classes

        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            mlp_list=[64, 64, 128], group_all=False,
            name='sa1'
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            mlp_list=[128, 128, 256], group_all=False,
            name='sa2'
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            mlp_list=[256, 512, 1024], group_all=True,
            name='sa3'
        )

        # Fully connected classification head
        self.fc1 = tf.keras.layers.Dense(512, use_bias=False, name='cls_fc1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='cls_bn1')
        self.drop1 = tf.keras.layers.Dropout(0.4, name='cls_drop1')

        self.fc2 = tf.keras.layers.Dense(256, use_bias=False, name='cls_fc2')
        self.bn2 = tf.keras.layers.BatchNormalization(name='cls_bn2')
        self.drop2 = tf.keras.layers.Dropout(0.4, name='cls_drop2')

        self.fc3 = tf.keras.layers.Dense(num_classes, name='cls_fc3')

    def call(self, xyz, training=False):
        """Forward pass.

        Args:
            xyz: (B, N, 3) input point cloud coordinates.
            training: bool, training mode flag.

        Returns:
            logits: (B, num_classes) classification logits.
        """
        # SA layers progressively reduce point count
        l1_xyz, l1_points = self.sa1(xyz, None, training=training)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, training=training)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, training=training)

        # Global feature: (B, 1, 1024) -> (B, 1024)
        x = tf.squeeze(l3_points, axis=1)

        # FC head
        x = self.fc1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.drop1(x, training=training)

        x = self.fc2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.drop2(x, training=training)

        logits = self.fc3(x)
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config


# ===========================================================================
# PointNet++ Detection Model
# ===========================================================================


class PointNetPPDetection(tf.keras.Model):
    """PointNet++ model for 3D object detection.

    Produces bounding box proposals with (x, y, z, w, h, l, yaw) and class scores.

    Architecture:
        SA1: 2048 centroids, radius=0.2, 64 neighbors, MLP [64, 64, 128]
        SA2: 1024 centroids, radius=0.4, 32 neighbors, MLP [128, 128, 256]
        SA3: 512 centroids, radius=0.8, 16 neighbors, MLP [256, 256, 512]
        SA4: 256 centroids, radius=1.6, 16 neighbors, MLP [512, 512, 1024]
        Vote/Proposal head: reduce to num_proposals, predict boxes + classes

    Args:
        num_classes: int, number of object classes.
        num_proposals: int, number of bounding box proposals to generate.
    """

    def __init__(self, num_classes=3, num_proposals=128, **kwargs):
        super(PointNetPPDetection, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_proposals = num_proposals

        # Encoder SA layers
        self.sa1 = PointNetSetAbstraction(
            npoint=2048, radius=0.2, nsample=64,
            mlp_list=[64, 64, 128], group_all=False,
            name='det_sa1'
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=1024, radius=0.4, nsample=32,
            mlp_list=[128, 128, 256], group_all=False,
            name='det_sa2'
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=512, radius=0.8, nsample=16,
            mlp_list=[256, 256, 512], group_all=False,
            name='det_sa3'
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=num_proposals, radius=1.6, nsample=16,
            mlp_list=[512, 512, 1024], group_all=False,
            name='det_sa4'
        )

        # Detection head: predict 3D bounding boxes and class scores
        # Box regression: (x, y, z, w, h, l, yaw) = 7 parameters
        self.box_fc1 = tf.keras.layers.Dense(512, use_bias=False, name='box_fc1')
        self.box_bn1 = tf.keras.layers.BatchNormalization(name='box_bn1')
        self.box_fc2 = tf.keras.layers.Dense(256, use_bias=False, name='box_fc2')
        self.box_bn2 = tf.keras.layers.BatchNormalization(name='box_bn2')
        self.box_fc3 = tf.keras.layers.Dense(7, name='box_regression')

        # Classification head per proposal
        self.cls_fc1 = tf.keras.layers.Dense(256, use_bias=False, name='det_cls_fc1')
        self.cls_bn1 = tf.keras.layers.BatchNormalization(name='det_cls_bn1')
        self.cls_fc2 = tf.keras.layers.Dense(128, use_bias=False, name='det_cls_fc2')
        self.cls_bn2 = tf.keras.layers.BatchNormalization(name='det_cls_bn2')
        self.cls_fc3 = tf.keras.layers.Dense(num_classes, name='det_cls_output')

    def call(self, xyz, training=False):
        """Forward pass.

        Args:
            xyz: (B, N, 3) input point cloud coordinates.
            training: bool, training mode flag.

        Returns:
            boxes: (B, num_proposals, 7) predicted 3D bounding boxes
                   (x, y, z, w, h, l, yaw).
            class_scores: (B, num_proposals, num_classes) classification logits.
        """
        # Encoder
        l1_xyz, l1_points = self.sa1(xyz, None, training=training)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, training=training)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, training=training)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points, training=training)

        # l4_points: (B, num_proposals, 1024)
        proposal_features = l4_points

        # Box regression head
        box_feat = self.box_fc1(proposal_features)
        box_feat = self.box_bn1(box_feat, training=training)
        box_feat = tf.nn.relu(box_feat)

        box_feat = self.box_fc2(box_feat)
        box_feat = self.box_bn2(box_feat, training=training)
        box_feat = tf.nn.relu(box_feat)

        boxes = self.box_fc3(box_feat)  # (B, num_proposals, 7)

        # Classification head
        cls_feat = self.cls_fc1(proposal_features)
        cls_feat = self.cls_bn1(cls_feat, training=training)
        cls_feat = tf.nn.relu(cls_feat)

        cls_feat = self.cls_fc2(cls_feat)
        cls_feat = self.cls_bn2(cls_feat, training=training)
        cls_feat = tf.nn.relu(cls_feat)

        class_scores = self.cls_fc3(cls_feat)  # (B, num_proposals, num_classes)

        return boxes, class_scores

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'num_proposals': self.num_proposals,
        })
        return config


# ===========================================================================
# PointNet++ Segmentation Model
# ===========================================================================


class PointNetPPSegmentation(tf.keras.Model):
    """PointNet++ model for 3D point cloud semantic segmentation.

    Uses Set Abstraction (encoder) and Feature Propagation (decoder) layers
    in a U-Net style architecture with skip connections.

    Architecture:
        Encoder:
            SA1: 1024 centroids, radius=0.1, 32 neighbors, MLP [32, 32, 64]
            SA2: 256 centroids, radius=0.2, 32 neighbors, MLP [64, 64, 128]
            SA3: 64 centroids, radius=0.4, 32 neighbors, MLP [128, 128, 256]
            SA4: 16 centroids, radius=0.8, 32 neighbors, MLP [256, 256, 512]
        Decoder:
            FP4: MLP [256, 256]
            FP3: MLP [256, 256]
            FP2: MLP [256, 128]
            FP1: MLP [128, 128, 128]
        Head: Conv1D(128) -> Conv1D(num_classes)

    Args:
        num_classes: int, number of segmentation classes.
    """

    def __init__(self, num_classes=13, **kwargs):
        super(PointNetPPSegmentation, self).__init__(**kwargs)
        self.num_classes = num_classes

        # Encoder (Set Abstraction layers)
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32,
            mlp_list=[32, 32, 64], group_all=False,
            name='seg_sa1'
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=32,
            mlp_list=[64, 64, 128], group_all=False,
            name='seg_sa2'
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=32,
            mlp_list=[128, 128, 256], group_all=False,
            name='seg_sa3'
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16, radius=0.8, nsample=32,
            mlp_list=[256, 256, 512], group_all=False,
            name='seg_sa4'
        )

        # Decoder (Feature Propagation layers)
        self.fp4 = PointNetFeaturePropagation(
            mlp_list=[256, 256], name='seg_fp4'
        )
        self.fp3 = PointNetFeaturePropagation(
            mlp_list=[256, 256], name='seg_fp3'
        )
        self.fp2 = PointNetFeaturePropagation(
            mlp_list=[256, 128], name='seg_fp2'
        )
        self.fp1 = PointNetFeaturePropagation(
            mlp_list=[128, 128, 128], name='seg_fp1'
        )

        # Segmentation head
        self.head_conv1 = tf.keras.layers.Conv1D(
            128, kernel_size=1, strides=1, padding='valid',
            use_bias=False, name='seg_head_conv1'
        )
        self.head_bn1 = tf.keras.layers.BatchNormalization(name='seg_head_bn1')
        self.head_drop1 = tf.keras.layers.Dropout(0.5, name='seg_head_drop1')
        self.head_conv2 = tf.keras.layers.Conv1D(
            num_classes, kernel_size=1, strides=1, padding='valid',
            name='seg_head_conv2'
        )

    def call(self, xyz, training=False):
        """Forward pass.

        Args:
            xyz: (B, N, 3) input point cloud coordinates.
            training: bool, training mode flag.

        Returns:
            logits: (B, N, num_classes) per-point classification logits.
        """
        # Store original coordinates for skip connections
        l0_xyz = xyz
        l0_points = None

        # Encoder forward pass
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points, training=training)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, training=training)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, training=training)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points, training=training)

        # Decoder forward pass (with skip connections)
        # FP4: interpolate from l4 (16 pts) to l3 (64 pts)
        l3_points_dec = self.fp4(
            l3_xyz, l4_xyz, l3_points, l4_points, training=training
        )

        # FP3: interpolate from l3_dec (64 pts) to l2 (256 pts)
        l2_points_dec = self.fp3(
            l2_xyz, l3_xyz, l2_points, l3_points_dec, training=training
        )

        # FP2: interpolate from l2_dec (256 pts) to l1 (1024 pts)
        l1_points_dec = self.fp2(
            l1_xyz, l2_xyz, l1_points, l2_points_dec, training=training
        )

        # FP1: interpolate from l1_dec (1024 pts) to l0 (N pts)
        l0_points_dec = self.fp1(
            l0_xyz, l1_xyz, l0_points, l1_points_dec, training=training
        )

        # Segmentation head: (B, N, 128) -> (B, N, num_classes)
        x = self.head_conv1(l0_points_dec)
        x = self.head_bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.head_drop1(x, training=training)

        logits = self.head_conv2(x)

        return logits

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config


# ===========================================================================
# Convenience factory functions
# ===========================================================================


def create_classification_model(num_classes=10):
    """Create a PointNet++ classification model.

    Args:
        num_classes: int, number of output classes.

    Returns:
        model: PointNetPPClassification instance.
    """
    return PointNetPPClassification(num_classes=num_classes)


def create_detection_model(num_classes=3, num_proposals=128):
    """Create a PointNet++ 3D detection model.

    Args:
        num_classes: int, number of object classes.
        num_proposals: int, number of bounding box proposals.

    Returns:
        model: PointNetPPDetection instance.
    """
    return PointNetPPDetection(
        num_classes=num_classes, num_proposals=num_proposals
    )


def create_segmentation_model(num_classes=13):
    """Create a PointNet++ segmentation model.

    Args:
        num_classes: int, number of segmentation classes.

    Returns:
        model: PointNetPPSegmentation instance.
    """
    return PointNetPPSegmentation(num_classes=num_classes)
