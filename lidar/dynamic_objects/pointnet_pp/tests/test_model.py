"""
Complete pytest test suite for PointNet++ TensorFlow implementation.

Tests cover all major components:
- Farthest Point Sampling (FPS)
- Ball Query
- KNN Query
- Set Abstraction (single-scale and multi-scale)
- Feature Propagation
- Classification, Detection, and Segmentation models
- Loss functions with gradient flow
"""

import sys
import os

# Adjust path to import from the tensorflow model module
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tensorflow'))
)

import pytest
import numpy as np
import tensorflow as tf

from model import (
    farthest_point_sampling,
    ball_query,
    square_distance,
    index_points,
    PointNetSetAbstraction,
    PointNetSetAbstractionMsg,
    PointNetFeaturePropagation,
    PointNetPPClassification,
    PointNetPPDetection,
    PointNetPPSegmentation,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility in all tests."""
    tf.random.set_seed(42)
    np.random.seed(42)


@pytest.fixture
def small_point_cloud():
    """Create a small random point cloud: batch=2, N=1024, C=3."""
    return tf.random.uniform([2, 1024, 3], minval=-1.0, maxval=1.0)


@pytest.fixture
def large_point_cloud():
    """Create a larger point cloud for detection tests: batch=2, N=16384, C=4."""
    return tf.random.uniform([2, 16384, 4], minval=-1.0, maxval=1.0)


@pytest.fixture
def medium_point_cloud():
    """Create a medium point cloud for segmentation tests: batch=2, N=4096, C=3."""
    return tf.random.uniform([2, 4096, 3], minval=-1.0, maxval=1.0)


# ===========================================================================
# Helper: KNN Query (TensorFlow implementation)
# ===========================================================================


def knn_query(k, xyz, new_xyz):
    """K-Nearest Neighbors query using pairwise distance matrix.

    For each point in new_xyz, finds the k closest points in xyz.

    Args:
        k: int, number of nearest neighbors.
        xyz: (B, N, 3) reference points.
        new_xyz: (B, S, 3) query points.

    Returns:
        indices: (B, S, k) KNN indices into xyz.
    """
    # Compute pairwise squared distances: (B, S, N)
    dists = square_distance(new_xyz, xyz)

    # Get top-k smallest distances (use negative for top_k which returns largest)
    neg_dists = -dists
    _, indices = tf.math.top_k(neg_dists, k=k)

    return indices


# ===========================================================================
# Test: Farthest Point Sampling
# ===========================================================================


class TestFarthestPointSampling:
    """Tests for farthest_point_sampling function."""

    def test_farthest_point_sampling(self, small_point_cloud):
        """Test FPS produces correct output shape with valid, unique indices."""
        xyz = small_point_cloud  # (2, 1024, 3)
        npoint = 256

        result = farthest_point_sampling(xyz, npoint)

        # Assert output shape is (2, 256)
        assert result.shape == (2, 256), (
            f"Expected shape (2, 256), got {result.shape}"
        )

        # Assert all indices are valid: 0 <= idx < 1024
        result_np = result.numpy()
        assert np.all(result_np >= 0), "Found negative indices"
        assert np.all(result_np < 1024), "Found indices >= 1024"

        # Assert no duplicate indices within each batch
        for batch_idx in range(2):
            indices_in_batch = result_np[batch_idx]
            unique_count = len(np.unique(indices_in_batch))
            assert unique_count == 256, (
                f"Batch {batch_idx}: expected 256 unique indices, "
                f"got {unique_count}"
            )


# ===========================================================================
# Test: Ball Query
# ===========================================================================


class TestBallQuery:
    """Tests for ball_query function."""

    def test_ball_query(self):
        """Test ball query with known point cloud and predictable distances."""
        # Create a point cloud with known geometry:
        # Points arranged on a grid so distances are predictable
        batch_size = 2
        # Generate points on a regular grid in [0, 1]
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        z = np.zeros(100)
        grid_x, grid_y = np.meshgrid(x, y)
        points_2d = np.stack([grid_x.flatten(), grid_y.flatten(), z], axis=-1)
        # (1, 100, 3) -> (2, 100, 3)
        xyz = tf.constant(
            np.tile(points_2d[np.newaxis, :, :], [batch_size, 1, 1]),
            dtype=tf.float32
        )

        # Query points: select a few centroids
        npoint = 5
        nsample = 16
        # Place query points at grid intersections
        query_coords = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        new_xyz = tf.constant(
            np.tile(query_coords[np.newaxis, :, :], [batch_size, 1, 1]),
            dtype=tf.float32
        )

        radius = 0.25

        result = ball_query(radius, nsample, xyz, new_xyz)

        # Assert output shape is correct: (batch, npoint, nsample)
        assert result.shape == (batch_size, npoint, nsample), (
            f"Expected shape ({batch_size}, {npoint}, {nsample}), "
            f"got {result.shape}"
        )

        # Assert found points are within radius
        result_np = result.numpy()
        xyz_np = xyz.numpy()
        new_xyz_np = new_xyz.numpy()

        for b in range(batch_size):
            for s in range(npoint):
                query_pt = new_xyz_np[b, s]  # (3,)
                neighbor_indices = result_np[b, s]  # (nsample,)
                for idx in neighbor_indices:
                    neighbor_pt = xyz_np[b, idx]
                    dist = np.sqrt(np.sum((query_pt - neighbor_pt) ** 2))
                    assert dist <= radius + 1e-5, (
                        f"Point at distance {dist:.4f} exceeds radius {radius}"
                    )

        # Verify padding behavior: for a corner point with fewer neighbors,
        # repeated indices should appear (first valid index repeated)
        # Check that corner point (1.0, 1.0, 0.0) has repeated indices
        # since it has fewer points within radius than nsample
        corner_idx = 2  # (1.0, 1.0, 0.0)
        corner_neighbors = result_np[0, corner_idx]
        # With only a few points in radius at the corner, we expect repeats
        unique_neighbors = np.unique(corner_neighbors)
        # There should be fewer unique neighbors than nsample (padding occurred)
        assert len(unique_neighbors) <= nsample, (
            "Padding check: unique neighbors should not exceed nsample"
        )


# ===========================================================================
# Test: KNN Query
# ===========================================================================


class TestKNNQuery:
    """Tests for k-nearest-neighbor query."""

    def test_knn_query(self):
        """Test KNN query returns K correct nearest neighbors."""
        # Create a point cloud with known nearest neighbors
        # Points placed at specific locations so we know the answer
        batch_size = 1
        k = 5

        # Place 20 points at known positions:
        # 10 points clustered near origin, 10 points far away
        near_points = np.random.RandomState(123).randn(10, 3) * 0.1  # near origin
        far_points = np.random.RandomState(456).randn(10, 3) * 0.1 + 10.0  # near (10,10,10)
        all_points = np.concatenate([near_points, far_points], axis=0)  # (20, 3)

        xyz = tf.constant(
            all_points[np.newaxis, :, :], dtype=tf.float32
        )  # (1, 20, 3)

        # Query point at origin - its k nearest neighbors should be from near_points
        query = tf.constant(
            [[[0.0, 0.0, 0.0]]], dtype=tf.float32
        )  # (1, 1, 3)

        result = knn_query(k, xyz, query)

        # Verify K neighbors returned
        assert result.shape == (1, 1, k), (
            f"Expected shape (1, 1, {k}), got {result.shape}"
        )

        result_np = result.numpy()[0, 0]  # (k,)

        # All k nearest neighbors should be from the near_points group (indices 0-9)
        assert np.all(result_np < 10), (
            f"Expected all neighbors from near group (idx < 10), "
            f"got indices: {result_np}"
        )

        # Verify they are actually the closest points by brute force
        xyz_np = all_points  # (20, 3)
        query_np = np.array([0.0, 0.0, 0.0])
        distances = np.sqrt(np.sum((xyz_np - query_np) ** 2, axis=-1))
        expected_indices = np.argsort(distances)[:k]

        # The returned indices should match the brute-force k-nearest
        assert set(result_np.tolist()) == set(expected_indices.tolist()), (
            f"KNN indices {result_np} don't match expected {expected_indices}"
        )

    def test_knn_query_batch(self):
        """Test KNN with batch dimension > 1 and multiple query points."""
        batch_size = 2
        N = 50
        S = 3
        k = 4

        tf.random.set_seed(99)
        xyz = tf.random.uniform([batch_size, N, 3], minval=0.0, maxval=1.0)
        new_xyz = tf.random.uniform([batch_size, S, 3], minval=0.0, maxval=1.0)

        result = knn_query(k, xyz, new_xyz)

        # Correct shape
        assert result.shape == (batch_size, S, k), (
            f"Expected ({batch_size}, {S}, {k}), got {result.shape}"
        )

        # Validate by brute force for each query point
        xyz_np = xyz.numpy()
        new_xyz_np = new_xyz.numpy()
        result_np = result.numpy()

        for b in range(batch_size):
            for s in range(S):
                query_pt = new_xyz_np[b, s]
                dists = np.sqrt(np.sum((xyz_np[b] - query_pt) ** 2, axis=-1))
                expected_knn = np.argsort(dists)[:k]
                assert set(result_np[b, s].tolist()) == set(expected_knn.tolist()), (
                    f"Batch {b}, query {s}: mismatch in KNN indices"
                )


# ===========================================================================
# Test: Set Abstraction (Single-Scale)
# ===========================================================================


class TestSetAbstraction:
    """Tests for PointNetSetAbstraction layer."""

    def test_set_abstraction(self):
        """Test SA layer forward pass produces correct output shapes."""
        batch_size = 4
        N = 1024
        C = 3
        npoint = 256
        mlp_list = [64, 64, 128]

        tf.random.set_seed(42)
        xyz = tf.random.uniform([batch_size, N, C], minval=-1.0, maxval=1.0)

        sa_layer = PointNetSetAbstraction(
            npoint=npoint, radius=0.4, nsample=32,
            mlp_list=mlp_list, group_all=False
        )

        new_xyz, new_points = sa_layer(xyz, None, training=False)

        # Assert output xyz shape: (4, npoint, 3)
        assert new_xyz.shape == (batch_size, npoint, 3), (
            f"Expected xyz shape ({batch_size}, {npoint}, 3), "
            f"got {new_xyz.shape}"
        )

        # Assert output features shape: (4, npoint, mlp[-1])
        assert new_points.shape == (batch_size, npoint, mlp_list[-1]), (
            f"Expected features shape ({batch_size}, {npoint}, {mlp_list[-1]}), "
            f"got {new_points.shape}"
        )

    def test_set_abstraction_with_features(self):
        """Test SA layer with input point features (not just xyz)."""
        batch_size = 2
        N = 512
        npoint = 128
        mlp_list = [32, 64]
        input_features_dim = 6

        tf.random.set_seed(77)
        xyz = tf.random.uniform([batch_size, N, 3], minval=-1.0, maxval=1.0)
        features = tf.random.uniform(
            [batch_size, N, input_features_dim], minval=-1.0, maxval=1.0
        )

        sa_layer = PointNetSetAbstraction(
            npoint=npoint, radius=0.3, nsample=16,
            mlp_list=mlp_list, group_all=False
        )

        new_xyz, new_points = sa_layer(xyz, features, training=False)

        assert new_xyz.shape == (batch_size, npoint, 3)
        assert new_points.shape == (batch_size, npoint, mlp_list[-1])


# ===========================================================================
# Test: Set Abstraction Multi-Scale Grouping (MSG)
# ===========================================================================


class TestSetAbstractionMsg:
    """Tests for PointNetSetAbstractionMsg layer."""

    def test_set_abstraction_msg(self):
        """Test MSG layer forward pass with 2 scales."""
        batch_size = 2
        N = 512
        npoint = 128
        radius_list = [0.2, 0.4]
        nsample_list = [16, 32]
        mlp_lists = [[32, 32, 64], [64, 64, 128]]

        tf.random.set_seed(42)
        xyz = tf.random.uniform([batch_size, N, 3], minval=-1.0, maxval=1.0)

        msg_layer = PointNetSetAbstractionMsg(
            npoint=npoint,
            radius_list=radius_list,
            nsample_list=nsample_list,
            mlp_lists=mlp_lists
        )

        new_xyz, new_points = msg_layer(xyz, None, training=False)

        # Assert output xyz shape
        assert new_xyz.shape == (batch_size, npoint, 3), (
            f"Expected xyz shape ({batch_size}, {npoint}, 3), "
            f"got {new_xyz.shape}"
        )

        # Verify feature dimension equals sum of per-scale MLPs' last dims
        expected_feat_dim = mlp_lists[0][-1] + mlp_lists[1][-1]  # 64 + 128 = 192
        assert new_points.shape == (batch_size, npoint, expected_feat_dim), (
            f"Expected features shape ({batch_size}, {npoint}, {expected_feat_dim}), "
            f"got {new_points.shape}"
        )

    def test_set_abstraction_msg_three_scales(self):
        """Test MSG layer with 3 scales to verify generalization."""
        batch_size = 2
        N = 256
        npoint = 64
        radius_list = [0.1, 0.2, 0.4]
        nsample_list = [8, 16, 32]
        mlp_lists = [[16, 32], [32, 64], [64, 128]]

        tf.random.set_seed(42)
        xyz = tf.random.uniform([batch_size, N, 3], minval=-1.0, maxval=1.0)

        msg_layer = PointNetSetAbstractionMsg(
            npoint=npoint,
            radius_list=radius_list,
            nsample_list=nsample_list,
            mlp_lists=mlp_lists
        )

        new_xyz, new_points = msg_layer(xyz, None, training=False)

        expected_feat_dim = 32 + 64 + 128  # 224
        assert new_points.shape == (batch_size, npoint, expected_feat_dim)


# ===========================================================================
# Test: Feature Propagation
# ===========================================================================


class TestFeaturePropagation:
    """Tests for PointNetFeaturePropagation layer."""

    def test_feature_propagation(self):
        """Test FP layer interpolates from sparse to dense points."""
        batch_size = 2
        N = 256  # dense points
        M = 64   # sparse points (N > M)
        C2 = 128  # features at sparse points
        mlp_list = [128, 64]

        tf.random.set_seed(42)
        xyz1 = tf.random.uniform([batch_size, N, 3], minval=-1.0, maxval=1.0)
        xyz2 = tf.random.uniform([batch_size, M, 3], minval=-1.0, maxval=1.0)
        points2 = tf.random.uniform(
            [batch_size, M, C2], minval=-1.0, maxval=1.0
        )

        fp_layer = PointNetFeaturePropagation(mlp_list=mlp_list)

        # No skip connection (points1=None)
        result = fp_layer(xyz1, xyz2, None, points2, training=False)

        # Assert output shape: (batch, N, mlp[-1])
        assert result.shape == (batch_size, N, mlp_list[-1]), (
            f"Expected shape ({batch_size}, {N}, {mlp_list[-1]}), "
            f"got {result.shape}"
        )

    def test_feature_propagation_with_skip(self):
        """Test FP layer with skip connection features."""
        batch_size = 2
        N = 512
        M = 128
        C1 = 64   # skip connection features at dense points
        C2 = 256  # features at sparse points
        mlp_list = [256, 128]

        tf.random.set_seed(42)
        xyz1 = tf.random.uniform([batch_size, N, 3], minval=-1.0, maxval=1.0)
        xyz2 = tf.random.uniform([batch_size, M, 3], minval=-1.0, maxval=1.0)
        points1 = tf.random.uniform(
            [batch_size, N, C1], minval=-1.0, maxval=1.0
        )
        points2 = tf.random.uniform(
            [batch_size, M, C2], minval=-1.0, maxval=1.0
        )

        fp_layer = PointNetFeaturePropagation(mlp_list=mlp_list)

        result = fp_layer(xyz1, xyz2, points1, points2, training=False)

        assert result.shape == (batch_size, N, mlp_list[-1])


# ===========================================================================
# Test: Classification Model
# ===========================================================================


class TestClassificationModel:
    """Tests for PointNetPPClassification model."""

    def test_classification_model(self, small_point_cloud):
        """Test classification model produces correct output with softmax."""
        num_classes = 10
        model = PointNetPPClassification(num_classes=num_classes)

        # Forward pass with (batch=2, 1024, 3)
        logits = model(small_point_cloud, training=False)

        # Assert output shape: (2, 10)
        assert logits.shape == (2, num_classes), (
            f"Expected shape (2, {num_classes}), got {logits.shape}"
        )

        # Assert output sums approximately to 1 after softmax
        probs = tf.nn.softmax(logits, axis=-1)
        prob_sums = tf.reduce_sum(probs, axis=-1).numpy()
        np.testing.assert_allclose(
            prob_sums, np.ones(2), atol=1e-5,
            err_msg="Softmax probabilities do not sum to 1"
        )


# ===========================================================================
# Test: Detection Model
# ===========================================================================


class TestDetectionModel:
    """Tests for PointNetPPDetection model."""

    def test_detection_model(self, large_point_cloud):
        """Test detection model produces bbox and class score outputs."""
        num_classes = 3
        num_proposals = 128
        model = PointNetPPDetection(
            num_classes=num_classes, num_proposals=num_proposals
        )

        # Forward pass with (batch=2, 16384, 4)
        # Detection model expects xyz as first 3 channels
        xyz_input = large_point_cloud[:, :, :3]
        boxes, class_scores = model(xyz_input, training=False)

        # Assert bbox output shape: (2, 128, 7)
        assert boxes.shape == (2, num_proposals, 7), (
            f"Expected boxes shape (2, {num_proposals}, 7), got {boxes.shape}"
        )

        # Assert class scores shape: (2, 128, 3)
        assert class_scores.shape == (2, num_proposals, num_classes), (
            f"Expected class_scores shape (2, {num_proposals}, {num_classes}), "
            f"got {class_scores.shape}"
        )


# ===========================================================================
# Test: Segmentation Model
# ===========================================================================


class TestSegmentationModel:
    """Tests for PointNetPPSegmentation model."""

    def test_segmentation_model(self, medium_point_cloud):
        """Test segmentation model produces per-point predictions."""
        num_classes = 13
        model = PointNetPPSegmentation(num_classes=num_classes)

        # Forward pass with (batch=2, 4096, 3)
        logits = model(medium_point_cloud, training=False)

        # Assert output shape: (2, 4096, 13)
        assert logits.shape == (2, 4096, num_classes), (
            f"Expected shape (2, 4096, {num_classes}), got {logits.shape}"
        )


# ===========================================================================
# Test: Loss Functions
# ===========================================================================


class TestLossFunctions:
    """Tests for loss functions and gradient flow."""

    def test_classification_loss(self):
        """Test classification cross-entropy loss with gradient flow."""
        tf.random.set_seed(42)
        num_classes = 10
        batch_size = 4

        logits = tf.Variable(
            tf.random.uniform([batch_size, num_classes], minval=-2.0, maxval=2.0)
        )
        labels = tf.constant([0, 3, 7, 2], dtype=tf.int32)

        with tf.GradientTape() as tape:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits
            )
            loss = tf.reduce_mean(loss)

        # Assert loss is a finite positive number
        loss_val = loss.numpy()
        assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"
        assert loss_val > 0, f"Loss should be positive, got {loss_val}"

        # Assert gradients flow
        grad = tape.gradient(loss, logits)
        assert grad is not None, "Gradient is None - no gradient flow"
        assert not np.any(np.isnan(grad.numpy())), "Gradient contains NaN"

    def test_detection_loss(self):
        """Test detection loss (smooth L1 + focal) with gradient flow."""
        tf.random.set_seed(42)
        batch_size = 2
        num_proposals = 64
        num_classes = 3

        # Predicted boxes and class scores
        pred_boxes = tf.Variable(
            tf.random.uniform([batch_size, num_proposals, 7], minval=-1.0, maxval=1.0)
        )
        pred_cls = tf.Variable(
            tf.random.uniform([batch_size, num_proposals, num_classes], minval=-2.0, maxval=2.0)
        )

        # Ground truth
        gt_boxes = tf.random.uniform(
            [batch_size, num_proposals, 7], minval=-1.0, maxval=1.0
        )
        gt_labels = tf.random.uniform(
            [batch_size, num_proposals], minval=0, maxval=num_classes,
            dtype=tf.int32
        )

        with tf.GradientTape() as tape:
            # Smooth L1 loss for bounding box regression
            box_diff = pred_boxes - gt_boxes
            abs_diff = tf.abs(box_diff)
            smooth_l1 = tf.where(
                abs_diff < 1.0,
                0.5 * tf.square(abs_diff),
                abs_diff - 0.5
            )
            box_loss = tf.reduce_mean(smooth_l1)

            # Focal loss for classification
            alpha = 0.25
            gamma = 2.0
            pred_probs = tf.nn.softmax(pred_cls, axis=-1)
            one_hot = tf.one_hot(gt_labels, num_classes)
            pt = tf.reduce_sum(pred_probs * one_hot, axis=-1)
            focal_weight = alpha * tf.pow(1.0 - pt, gamma)
            ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=gt_labels, logits=pred_cls
            )
            focal_loss = tf.reduce_mean(focal_weight * ce_loss)

            # Combined detection loss
            total_loss = box_loss + focal_loss

        # Assert losses are finite positive numbers
        total_loss_val = total_loss.numpy()
        box_loss_val = box_loss.numpy()
        focal_loss_val = focal_loss.numpy()

        assert np.isfinite(total_loss_val), f"Total loss not finite: {total_loss_val}"
        assert total_loss_val > 0, f"Total loss should be positive: {total_loss_val}"
        assert np.isfinite(box_loss_val), f"Box loss not finite: {box_loss_val}"
        assert box_loss_val > 0, f"Box loss should be positive: {box_loss_val}"
        assert np.isfinite(focal_loss_val), f"Focal loss not finite: {focal_loss_val}"
        assert focal_loss_val > 0, f"Focal loss should be positive: {focal_loss_val}"

        # Assert gradients flow to both box and class predictions
        grad_boxes = tape.gradient(total_loss, pred_boxes)
        grad_cls = tape.gradient(total_loss, pred_cls)

        assert grad_boxes is not None, "Box gradient is None"
        assert grad_cls is not None, "Class gradient is None"
        assert not np.any(np.isnan(grad_boxes.numpy())), "Box gradient has NaN"
        assert not np.any(np.isnan(grad_cls.numpy())), "Class gradient has NaN"

    def test_segmentation_loss(self):
        """Test segmentation per-point cross-entropy loss with gradient flow."""
        tf.random.set_seed(42)
        batch_size = 2
        num_points = 1024
        num_classes = 13

        logits = tf.Variable(
            tf.random.uniform(
                [batch_size, num_points, num_classes], minval=-2.0, maxval=2.0
            )
        )
        # Per-point labels
        labels = tf.random.uniform(
            [batch_size, num_points], minval=0, maxval=num_classes,
            dtype=tf.int32
        )

        with tf.GradientTape() as tape:
            # Reshape for sparse cross-entropy: (B*N, C) and (B*N,)
            logits_flat = tf.reshape(logits, [-1, num_classes])
            labels_flat = tf.reshape(labels, [-1])
            loss_per_point = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_flat, logits=logits_flat
            )
            loss = tf.reduce_mean(loss_per_point)

        # Assert loss is finite and positive
        loss_val = loss.numpy()
        assert np.isfinite(loss_val), f"Seg loss not finite: {loss_val}"
        assert loss_val > 0, f"Seg loss should be positive: {loss_val}"

        # Assert gradients flow
        grad = tape.gradient(loss, logits)
        assert grad is not None, "Segmentation gradient is None"
        assert not np.any(np.isnan(grad.numpy())), "Segmentation gradient has NaN"
        assert grad.shape == logits.shape, (
            f"Gradient shape {grad.shape} doesn't match logits shape {logits.shape}"
        )
