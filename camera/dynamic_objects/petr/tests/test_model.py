"""
Comprehensive tests for PETR/StreamPETR TensorFlow model.

Tests cover:
  - 3D position embedding generation and coordinate transforms
  - Position embedding sensitivity to camera parameters
  - StreamPETR temporal query propagation with ego-motion
  - Motion-Aware Layer Normalization modulation behavior
  - Transformer decoder forward pass shape verification
  - Full model end-to-end forward pass (PETR, PETRv2, StreamPETR)
  - Loss computation with Hungarian matching
  - Sequential inference temporal consistency
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tensorflow"))

import tensorflow as tf
from model import (
    PETR,
    FPN,
    PositionEmbedding3D,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
    MotionAwareLayerNorm,
    TransformerDecoder,
    TransformerDecoderLayer,
    DetectionHead,
    PETRLoss,
    HungarianMatcher,
    build_petr_model,
)


@pytest.fixture
def base_config():
    """Base PETR configuration for testing."""
    return {
        "num_classes": 10,
        "embed_dims": 64,
        "num_queries": 50,
        "num_decoder_layers": 2,
        "num_heads": 4,
        "ffn_dims": 128,
        "dropout": 0.0,
        "num_depth_bins": 8,
        "depth_range": (1.0, 61.0),
        "temporal": False,
        "num_propagated_queries": 20,
        "backbone_output_layers": [
            "conv3_block4_out",
            "conv4_block6_out",
            "conv5_block3_out",
        ],
    }


@pytest.fixture
def temporal_config(base_config):
    """StreamPETR configuration with temporal enabled."""
    config = base_config.copy()
    config["temporal"] = True
    return config


@pytest.fixture
def dummy_camera_params():
    """Generate dummy camera intrinsics and extrinsics."""
    batch_size = 2
    num_cameras = 6

    fx, fy = 1260.0, 1260.0
    cx, cy = 800.0, 450.0
    intrinsics = np.zeros((batch_size, num_cameras, 3, 3), dtype=np.float32)
    for b in range(batch_size):
        for n in range(num_cameras):
            intrinsics[b, n] = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ])

    extrinsics = np.zeros((batch_size, num_cameras, 4, 4), dtype=np.float32)
    angles = [0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3]
    for b in range(batch_size):
        for n, angle in enumerate(angles):
            rot = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ], dtype=np.float32)
            extrinsics[b, n, :3, :3] = rot
            extrinsics[b, n, :3, 3] = [
                1.5 * np.cos(angle), 1.5 * np.sin(angle), 1.5
            ]
            extrinsics[b, n, 3, 3] = 1.0

    return (
        tf.constant(intrinsics),
        tf.constant(extrinsics),
    )


class TestPositionEmbedding3D:
    """Tests for 3D Position Embedding layer."""

    def test_frustum_generation_shapes(self):
        """Verify frustum grid has correct shape (D, H, W, 3)."""
        pe = PositionEmbedding3D(embed_dims=64, num_depth_bins=8)
        pe.build(None)

        frustum = pe._create_frustum(height=15, width=25)
        assert frustum.shape == (8, 15, 25, 3), f"Expected (8, 15, 25, 3), got {frustum.shape}"

        u_vals = frustum[0, 0, :, 0].numpy()
        assert u_vals[0] > 0, "First u value should be positive"
        assert u_vals[-1] < 25, "Last u value should be less than width"
        assert np.all(np.diff(u_vals) > 0), "u values should be monotonically increasing"

        v_vals = frustum[0, :, 0, 1].numpy()
        assert np.all(np.diff(v_vals) > 0), "v values should be monotonically increasing"

        d_vals = frustum[:, 0, 0, 2].numpy()
        assert d_vals[0] == pytest.approx(1.0, abs=0.1), "First depth should be near depth_start"
        assert np.all(np.diff(d_vals) > 0), "depth values should be monotonically increasing"

    def test_coordinate_transform_shapes(self, dummy_camera_params):
        """Verify 3D coordinate transform output shape."""
        intrinsics, extrinsics = dummy_camera_params
        pe = PositionEmbedding3D(embed_dims=64, num_depth_bins=4)
        pe.build(None)

        frustum = pe._create_frustum(height=8, width=12)
        coords_3d = pe._frustum_to_3d(frustum, intrinsics, extrinsics)

        batch_size = intrinsics.shape[0]
        num_cameras = intrinsics.shape[1]
        num_points = 4 * 8 * 12

        assert coords_3d.shape == (batch_size, num_cameras, num_points, 3), \
            f"Expected ({batch_size}, {num_cameras}, {num_points}, 3), got {coords_3d.shape}"

    def test_full_pe_output_shape(self, dummy_camera_params):
        """Verify full position embedding output shape."""
        intrinsics, extrinsics = dummy_camera_params
        pe = PositionEmbedding3D(embed_dims=64, num_depth_bins=4)
        pe.build(None)

        pos_embed = pe((8, 12), intrinsics, extrinsics)
        B, N = intrinsics.shape[0], intrinsics.shape[1]
        expected_tokens = 4 * 8 * 12

        assert pos_embed.shape == (B, N, expected_tokens, 64), \
            f"Expected ({B}, {N}, {expected_tokens}, 64), got {pos_embed.shape}"

    def test_pe_values_in_range(self, dummy_camera_params):
        """Verify PE output has reasonable values (no NaN/Inf)."""
        intrinsics, extrinsics = dummy_camera_params
        pe = PositionEmbedding3D(embed_dims=64, num_depth_bins=4)
        pe.build(None)

        pos_embed = pe((8, 12), intrinsics, extrinsics)
        assert not tf.reduce_any(tf.math.is_nan(pos_embed)).numpy(), "PE contains NaN"
        assert not tf.reduce_any(tf.math.is_inf(pos_embed)).numpy(), "PE contains Inf"


class TestPositionEmbeddingWithCameraParams:
    """Test that 3D PE changes with different camera intrinsics/extrinsics."""

    def test_different_intrinsics_produce_different_pe(self):
        """PE should change when camera intrinsics change."""
        pe = PositionEmbedding3D(embed_dims=32, num_depth_bins=4)
        pe.build(None)

        intrinsics_1 = tf.constant([[[[1260, 0, 800], [0, 1260, 450], [0, 0, 1]]]], dtype=tf.float32)
        intrinsics_2 = tf.constant([[[[800, 0, 640], [0, 800, 360], [0, 0, 1]]]], dtype=tf.float32)
        extrinsics = tf.eye(4, batch_shape=[1, 1])

        pe_1 = pe((4, 6), intrinsics_1, extrinsics)
        pe_2 = pe((4, 6), intrinsics_2, extrinsics)

        diff = tf.reduce_mean(tf.abs(pe_1 - pe_2)).numpy()
        assert diff > 1e-5, f"PEs should differ with different intrinsics, but diff={diff}"

    def test_different_extrinsics_produce_different_pe(self):
        """PE should change when camera extrinsics change."""
        pe = PositionEmbedding3D(embed_dims=32, num_depth_bins=4)
        pe.build(None)

        intrinsics = tf.constant([[[[1260, 0, 800], [0, 1260, 450], [0, 0, 1]]]], dtype=tf.float32)

        ext_1 = tf.eye(4, batch_shape=[1, 1])
        ext_2 = tf.constant([[[[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]]]], dtype=tf.float32)

        pe_1 = pe((4, 6), intrinsics, ext_1)
        pe_2 = pe((4, 6), intrinsics, ext_2)

        diff = tf.reduce_mean(tf.abs(pe_1 - pe_2)).numpy()
        assert diff > 1e-5, f"PEs should differ with different extrinsics, but diff={diff}"

    def test_identical_params_produce_identical_pe(self, dummy_camera_params):
        """Same parameters should produce identical PE (deterministic)."""
        intrinsics, extrinsics = dummy_camera_params
        pe = PositionEmbedding3D(embed_dims=32, num_depth_bins=4)
        pe.build(None)

        pe_1 = pe((4, 6), intrinsics, extrinsics)
        pe_2 = pe((4, 6), intrinsics, extrinsics)

        diff = tf.reduce_max(tf.abs(pe_1 - pe_2)).numpy()
        assert diff < 1e-6, f"Identical inputs should give identical PE, max diff={diff}"


class TestTemporalPropagation:
    """Tests for StreamPETR temporal query propagation."""

    def test_propagated_query_shape(self, temporal_config):
        """Verify propagated query dimensions after ego-motion transform."""
        model = build_petr_model(temporal_config)
        B = 2
        embed_dims = temporal_config["embed_dims"]
        num_queries = temporal_config["num_queries"]
        num_propagated = temporal_config["num_propagated_queries"]

        prev_query = tf.random.normal([B, num_queries, embed_dims])
        ego_motion = tf.eye(4, batch_shape=[B])

        propagated, motion_embed = model._propagate_queries(prev_query, ego_motion)

        assert propagated.shape == (B, num_propagated, embed_dims), \
            f"Expected ({B}, {num_propagated}, {embed_dims}), got {propagated.shape}"
        assert motion_embed.shape == (B, embed_dims), \
            f"Expected ({B}, {embed_dims}), got {motion_embed.shape}"

    def test_ego_motion_affects_output(self, temporal_config):
        """Different ego-motion should produce different motion embeddings."""
        model = build_petr_model(temporal_config)
        B = 1
        embed_dims = temporal_config["embed_dims"]
        num_queries = temporal_config["num_queries"]

        prev_query = tf.random.normal([B, num_queries, embed_dims])

        ego_identity = tf.eye(4, batch_shape=[B])
        ego_translated = tf.constant([[[1, 0, 0, 5], [0, 1, 0, 3], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=tf.float32)

        _, motion_embed_1 = model._propagate_queries(prev_query, ego_identity)
        _, motion_embed_2 = model._propagate_queries(prev_query, ego_translated)

        diff = tf.reduce_mean(tf.abs(motion_embed_1 - motion_embed_2)).numpy()
        assert diff > 1e-4, f"Different ego-motion should produce different embeddings, diff={diff}"

    def test_no_prev_query_defaults_to_learned(self, temporal_config):
        """When no prev_query is provided, model uses learned queries."""
        model = build_petr_model(temporal_config)
        B = 1
        num_cameras = 6
        H, W = 224, 400

        images = tf.random.normal([B, num_cameras, H, W, 3])
        intrinsics = tf.eye(3, batch_shape=[B, num_cameras])
        intrinsics = intrinsics * tf.constant([[[1260, 1260, 1]]], dtype=tf.float32)[:, :, :, None] * intrinsics
        extrinsics = tf.eye(4, batch_shape=[B, num_cameras])

        simple_intrinsics = tf.constant(
            [[[[1260, 0, 200], [0, 1260, 112], [0, 0, 1]]]] * num_cameras,
            dtype=tf.float32,
        )
        simple_intrinsics = tf.reshape(simple_intrinsics, [1, num_cameras, 3, 3])

        outputs = model(
            images=images,
            intrinsics=simple_intrinsics,
            extrinsics=extrinsics,
            ego_motion=None,
            prev_query=None,
            training=False,
        )

        assert "cls_scores" in outputs
        assert "bbox_preds" in outputs
        assert "query_output" in outputs


class TestMotionAwareLayerNorm:
    """Tests for Motion-Aware Layer Normalization."""

    def test_output_shape(self):
        """Verify output shape matches input shape."""
        maln = MotionAwareLayerNorm(embed_dims=64)
        x = tf.random.normal([2, 50, 64])
        motion = tf.random.normal([2, 64])

        maln.build(None)
        output = maln(x, motion)
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    def test_different_motion_different_output(self):
        """Different motion inputs should produce different outputs."""
        maln = MotionAwareLayerNorm(embed_dims=64)
        maln.build(None)

        x = tf.random.normal([2, 50, 64])
        motion_1 = tf.zeros([2, 64])
        motion_2 = tf.ones([2, 64])

        out_1 = maln(x, motion_1)
        out_2 = maln(x, motion_2)

        diff = tf.reduce_mean(tf.abs(out_1 - out_2)).numpy()
        assert diff > 1e-4, f"Different motion should give different output, diff={diff}"

    def test_zero_motion_is_standard_ln(self):
        """With zero motion, MALN should behave close to standard LayerNorm + identity."""
        maln = MotionAwareLayerNorm(embed_dims=64)
        maln.build(None)

        maln.motion_proj.set_weights([
            np.zeros((64, 128), dtype=np.float32),
            np.zeros(128, dtype=np.float32),
        ])

        x = tf.random.normal([2, 50, 64])
        motion = tf.zeros([2, 64])

        output = maln(x, motion)
        expected = maln.norm(x)

        diff = tf.reduce_max(tf.abs(output - expected)).numpy()
        assert diff < 1e-5, f"Zero motion with zero weights should give standard LN, max diff={diff}"


class TestDecoderForward:
    """Tests for Transformer decoder forward pass."""

    def test_single_layer_shapes(self):
        """Verify single decoder layer output shape."""
        layer = TransformerDecoderLayer(embed_dims=64, num_heads=4)
        layer.build(None)

        B, Q, K, C = 2, 50, 200, 64
        query = tf.random.normal([B, Q, C])
        key = tf.random.normal([B, K, C])
        value = tf.random.normal([B, K, C])
        query_pos = tf.random.normal([B, Q, C])
        key_pos = tf.random.normal([B, K, C])

        output = layer(query, key, value, query_pos=query_pos, key_pos=key_pos, training=False)
        assert output.shape == (B, Q, C), f"Expected ({B}, {Q}, {C}), got {output.shape}"

    def test_decoder_returns_intermediate(self):
        """Verify decoder returns outputs from each layer."""
        decoder = TransformerDecoder(num_layers=3, embed_dims=64, num_heads=4)
        decoder.build(None)

        B, Q, K, C = 2, 50, 200, 64
        query = tf.random.normal([B, Q, C])
        key = tf.random.normal([B, K, C])
        value = tf.random.normal([B, K, C])

        outputs = decoder(query, key, value, training=False)
        assert len(outputs) == 3, f"Expected 3 intermediate outputs, got {len(outputs)}"
        for i, out in enumerate(outputs):
            assert out.shape == (B, Q, C), f"Layer {i} output shape mismatch"

    def test_decoder_with_motion_aware_ln(self):
        """Verify decoder with motion-aware LN processes correctly."""
        decoder = TransformerDecoder(
            num_layers=2, embed_dims=64, num_heads=4, use_motion_aware_ln=True
        )
        decoder.build(None)

        B, Q, K, C = 2, 50, 200, 64
        query = tf.random.normal([B, Q, C])
        key = tf.random.normal([B, K, C])
        value = tf.random.normal([B, K, C])
        motion_embed = tf.random.normal([B, C])

        outputs = decoder(query, key, value, motion_embed=motion_embed, training=False)
        assert len(outputs) == 2
        assert outputs[-1].shape == (B, Q, C)

    def test_no_nan_in_output(self):
        """Verify no NaN values in decoder output."""
        decoder = TransformerDecoder(num_layers=2, embed_dims=64, num_heads=4)
        decoder.build(None)

        B, Q, K, C = 2, 50, 200, 64
        query = tf.random.normal([B, Q, C])
        key = tf.random.normal([B, K, C])
        value = tf.random.normal([B, K, C])

        outputs = decoder(query, key, value, training=False)
        for out in outputs:
            assert not tf.reduce_any(tf.math.is_nan(out)).numpy(), "NaN in decoder output"


class TestFullModelForward:
    """Tests for end-to-end model forward pass."""

    def test_petr_forward(self, base_config):
        """Verify PETR base model forward pass shapes."""
        model = build_petr_model(base_config)

        B = 1
        N = 6
        H, W = 224, 400
        num_queries = base_config["num_queries"]
        num_classes = base_config["num_classes"]
        num_layers = base_config["num_decoder_layers"]

        images = tf.random.normal([B, N, H, W, 3])
        intrinsics = tf.constant(
            [[[[1260, 0, 200], [0, 1260, 112], [0, 0, 1]]]] * N,
            dtype=tf.float32,
        )
        intrinsics = tf.reshape(intrinsics, [1, N, 3, 3])
        extrinsics = tf.eye(4, batch_shape=[B, N])

        outputs = model(images, intrinsics, extrinsics, training=False)

        assert len(outputs["cls_scores"]) == num_layers
        assert len(outputs["bbox_preds"]) == num_layers
        assert outputs["cls_scores"][-1].shape == (B, num_queries, num_classes)
        assert outputs["bbox_preds"][-1].shape == (B, num_queries, 10)
        assert outputs["query_output"].shape == (B, num_queries, base_config["embed_dims"])

    def test_stream_petr_forward(self, temporal_config):
        """Verify StreamPETR model forward pass with temporal inputs."""
        model = build_petr_model(temporal_config)

        B = 1
        N = 6
        H, W = 224, 400
        num_queries = temporal_config["num_queries"]
        embed_dims = temporal_config["embed_dims"]

        images = tf.random.normal([B, N, H, W, 3])
        intrinsics = tf.constant(
            [[[[1260, 0, 200], [0, 1260, 112], [0, 0, 1]]]] * N,
            dtype=tf.float32,
        )
        intrinsics = tf.reshape(intrinsics, [1, N, 3, 3])
        extrinsics = tf.eye(4, batch_shape=[B, N])
        ego_motion = tf.eye(4, batch_shape=[B])
        prev_query = tf.random.normal([B, num_queries, embed_dims])

        outputs = model(
            images, intrinsics, extrinsics,
            ego_motion=ego_motion, prev_query=prev_query,
            training=False,
        )

        assert outputs["cls_scores"][-1].shape == (B, num_queries, temporal_config["num_classes"])
        assert outputs["bbox_preds"][-1].shape == (B, num_queries, 10)

    def test_stream_petr_without_prev_query(self, temporal_config):
        """StreamPETR should work without prev_query (first frame)."""
        model = build_petr_model(temporal_config)

        B = 1
        N = 6
        H, W = 224, 400

        images = tf.random.normal([B, N, H, W, 3])
        intrinsics = tf.constant(
            [[[[1260, 0, 200], [0, 1260, 112], [0, 0, 1]]]] * N,
            dtype=tf.float32,
        )
        intrinsics = tf.reshape(intrinsics, [1, N, 3, 3])
        extrinsics = tf.eye(4, batch_shape=[B, N])

        outputs = model(images, intrinsics, extrinsics, training=False)
        assert outputs["cls_scores"][-1].shape[0] == B

    def test_output_no_nan(self, base_config):
        """Verify no NaN values in full model output."""
        model = build_petr_model(base_config)

        B, N, H, W = 1, 6, 224, 400
        images = tf.random.normal([B, N, H, W, 3])
        intrinsics = tf.constant(
            [[[[1260, 0, 200], [0, 1260, 112], [0, 0, 1]]]] * N, dtype=tf.float32,
        )
        intrinsics = tf.reshape(intrinsics, [1, N, 3, 3])
        extrinsics = tf.eye(4, batch_shape=[B, N])

        outputs = model(images, intrinsics, extrinsics, training=False)

        for layer_cls in outputs["cls_scores"]:
            assert not tf.reduce_any(tf.math.is_nan(layer_cls)).numpy()
        for layer_bbox in outputs["bbox_preds"]:
            assert not tf.reduce_any(tf.math.is_nan(layer_bbox)).numpy()


class TestLossComputation:
    """Tests for Hungarian matching and loss computation."""

    def test_hungarian_matcher_basic(self):
        """Verify matcher produces valid assignment."""
        matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0)

        B, Q, C = 1, 10, 10
        cls_scores = tf.random.normal([B, Q, C])
        bbox_preds = tf.random.normal([B, Q, 10])

        gt_labels = np.full((B, 5), -1, dtype=np.int32)
        gt_labels[0, :3] = [0, 1, 2]
        gt_bboxes = np.random.randn(B, 5, 10).astype(np.float32)

        matches = matcher.match(
            cls_scores, bbox_preds,
            tf.constant(gt_labels), tf.constant(gt_bboxes),
        )

        assert len(matches) == B
        pred_idx, gt_idx = matches[0]
        assert len(pred_idx) == 3, f"Expected 3 matches, got {len(pred_idx)}"
        assert len(gt_idx) == 3
        assert len(set(pred_idx.numpy())) == 3, "Pred indices should be unique"
        assert len(set(gt_idx.numpy())) == 3, "GT indices should be unique"

    def test_hungarian_matcher_empty_gt(self):
        """Matcher should handle empty ground truth gracefully."""
        matcher = HungarianMatcher()

        B, Q, C = 1, 10, 10
        cls_scores = tf.random.normal([B, Q, C])
        bbox_preds = tf.random.normal([B, Q, 10])

        gt_labels = tf.constant([[-1, -1, -1]], dtype=tf.int32)
        gt_bboxes = tf.zeros([1, 3, 10])

        matches = matcher.match(cls_scores, bbox_preds, gt_labels, gt_bboxes)
        pred_idx, gt_idx = matches[0]
        assert len(pred_idx) == 0
        assert len(gt_idx) == 0

    def test_loss_computation_nonzero(self):
        """Verify loss computation produces non-zero values with valid inputs."""
        loss_fn = PETRLoss(num_classes=10, cls_weight=2.0, bbox_weight=5.0)

        B, Q, C = 1, 20, 10
        outputs = {
            "cls_scores": [tf.random.normal([B, Q, C])],
            "bbox_preds": [tf.random.normal([B, Q, 10])],
        }

        gt_labels = np.full((B, 5), -1, dtype=np.int32)
        gt_labels[0, :3] = [0, 3, 7]
        gt_bboxes = np.random.randn(B, 5, 10).astype(np.float32)

        losses = loss_fn(
            outputs,
            tf.constant(gt_labels, dtype=tf.int32),
            tf.constant(gt_bboxes, dtype=tf.float32),
        )

        assert losses["total_loss"].numpy() > 0, "Total loss should be positive"
        assert losses["cls_loss"].numpy() >= 0, "Classification loss should be non-negative"
        assert losses["bbox_loss"].numpy() >= 0, "Bbox loss should be non-negative"
        assert not np.isnan(losses["total_loss"].numpy()), "Loss should not be NaN"

    def test_loss_decreases_with_perfect_match(self):
        """Loss should be lower when predictions are close to ground truth."""
        loss_fn = PETRLoss(num_classes=10, cls_weight=2.0, bbox_weight=5.0)

        B, Q, C = 1, 20, 10
        gt_labels = np.full((B, 5), -1, dtype=np.int32)
        gt_labels[0, :2] = [0, 1]
        gt_bboxes = np.zeros((B, 5, 10), dtype=np.float32)
        gt_bboxes[0, 0] = [1, 2, 0, 2, 4, 1.5, 0, 1, 1, 0]
        gt_bboxes[0, 1] = [5, 3, 0, 1.8, 4.5, 1.6, 0.7, 0.7, 0.5, 0.3]

        random_cls = tf.random.normal([B, Q, C])
        random_bbox = tf.random.normal([B, Q, 10]) * 10

        outputs_random = {"cls_scores": [random_cls], "bbox_preds": [random_bbox]}

        close_bbox = tf.Variable(tf.zeros([B, Q, 10]))
        close_bbox[0, 0].assign(gt_bboxes[0, 0])
        close_bbox[0, 1].assign(gt_bboxes[0, 1])

        close_cls = tf.constant(np.full((B, Q, C), -5.0, dtype=np.float32))
        close_cls_np = close_cls.numpy()
        close_cls_np[0, 0, 0] = 5.0
        close_cls_np[0, 1, 1] = 5.0
        close_cls = tf.constant(close_cls_np)

        outputs_close = {"cls_scores": [close_cls], "bbox_preds": [tf.constant(close_bbox.numpy())]}

        loss_random = loss_fn(outputs_random, tf.constant(gt_labels), tf.constant(gt_bboxes))
        loss_close = loss_fn(outputs_close, tf.constant(gt_labels), tf.constant(gt_bboxes))

        assert loss_close["bbox_loss"].numpy() < loss_random["bbox_loss"].numpy(), \
            "Close predictions should have lower bbox loss"


class TestInferenceSequence:
    """Tests for temporal consistency in sequential inference (StreamPETR)."""

    def test_query_propagation_across_frames(self, temporal_config):
        """Verify queries can be propagated from one frame to the next."""
        model = build_petr_model(temporal_config)

        B, N, H, W = 1, 6, 224, 400
        num_queries = temporal_config["num_queries"]
        embed_dims = temporal_config["embed_dims"]

        intrinsics = tf.constant(
            [[[[1260, 0, 200], [0, 1260, 112], [0, 0, 1]]]] * N, dtype=tf.float32,
        )
        intrinsics = tf.reshape(intrinsics, [1, N, 3, 3])
        extrinsics = tf.eye(4, batch_shape=[B, N])

        images_t0 = tf.random.normal([B, N, H, W, 3])
        outputs_t0 = model(images_t0, intrinsics, extrinsics, training=False)

        prev_query = outputs_t0["query_output"]
        assert prev_query.shape == (B, num_queries, embed_dims)

        images_t1 = tf.random.normal([B, N, H, W, 3])
        ego_motion = tf.eye(4, batch_shape=[B])

        outputs_t1 = model(
            images_t1, intrinsics, extrinsics,
            ego_motion=ego_motion, prev_query=prev_query,
            training=False,
        )

        assert outputs_t1["cls_scores"][-1].shape == (B, num_queries, temporal_config["num_classes"])
        assert outputs_t1["query_output"].shape == (B, num_queries, embed_dims)

    def test_temporal_output_differs_from_static(self, temporal_config):
        """Temporal inference should produce different results than no temporal."""
        model = build_petr_model(temporal_config)

        B, N, H, W = 1, 6, 224, 400
        num_queries = temporal_config["num_queries"]
        embed_dims = temporal_config["embed_dims"]

        tf.random.set_seed(42)
        images = tf.random.normal([B, N, H, W, 3])
        intrinsics = tf.constant(
            [[[[1260, 0, 200], [0, 1260, 112], [0, 0, 1]]]] * N, dtype=tf.float32,
        )
        intrinsics = tf.reshape(intrinsics, [1, N, 3, 3])
        extrinsics = tf.eye(4, batch_shape=[B, N])

        outputs_no_temporal = model(images, intrinsics, extrinsics, training=False)

        prev_query = tf.random.normal([B, num_queries, embed_dims])
        ego_motion = tf.constant([[[1, 0, 0, 2], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=tf.float32)

        outputs_temporal = model(
            images, intrinsics, extrinsics,
            ego_motion=ego_motion, prev_query=prev_query,
            training=False,
        )

        cls_diff = tf.reduce_mean(tf.abs(
            outputs_temporal["cls_scores"][-1] - outputs_no_temporal["cls_scores"][-1]
        )).numpy()

        assert cls_diff > 1e-4, \
            f"Temporal and non-temporal outputs should differ, but diff={cls_diff}"

    def test_sequence_of_three_frames(self, temporal_config):
        """Run a 3-frame sequence and verify no errors or NaNs."""
        model = build_petr_model(temporal_config)

        B, N, H, W = 1, 6, 224, 400
        intrinsics = tf.constant(
            [[[[1260, 0, 200], [0, 1260, 112], [0, 0, 1]]]] * N, dtype=tf.float32,
        )
        intrinsics = tf.reshape(intrinsics, [1, N, 3, 3])
        extrinsics = tf.eye(4, batch_shape=[B, N])
        ego_motion = tf.eye(4, batch_shape=[B])

        prev_query = None
        for frame_idx in range(3):
            images = tf.random.normal([B, N, H, W, 3])

            kwargs = {
                "images": images,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "training": False,
            }
            if prev_query is not None:
                kwargs["ego_motion"] = ego_motion
                kwargs["prev_query"] = prev_query

            outputs = model(**kwargs)

            for layer_cls in outputs["cls_scores"]:
                assert not tf.reduce_any(tf.math.is_nan(layer_cls)).numpy(), \
                    f"NaN in cls_scores at frame {frame_idx}"
            for layer_bbox in outputs["bbox_preds"]:
                assert not tf.reduce_any(tf.math.is_nan(layer_bbox)).numpy(), \
                    f"NaN in bbox_preds at frame {frame_idx}"

            prev_query = outputs["query_output"]


class TestDetectionHead:
    """Tests for the detection head."""

    def test_output_shapes(self):
        """Verify detection head output dimensions."""
        head = DetectionHead(num_classes=10, embed_dims=64)
        head.build(None)

        B, Q, C = 2, 50, 64
        features = tf.random.normal([B, Q, C])
        cls_scores, bbox_preds = head(features)

        assert cls_scores.shape == (B, Q, 10)
        assert bbox_preds.shape == (B, Q, 10)

    def test_cls_logits_not_all_same(self):
        """Classification logits should vary across queries."""
        head = DetectionHead(num_classes=10, embed_dims=64)
        head.build(None)

        features = tf.random.normal([1, 50, 64])
        cls_scores, _ = head(features)

        score_std = tf.math.reduce_std(cls_scores).numpy()
        assert score_std > 0.01, "Classification scores should vary"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
