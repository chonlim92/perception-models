"""
Metrics Registry for Autonomous Driving Perception Models.

This module provides a unified interface for computing evaluation metrics
across all perception tasks: 3D detection, segmentation, HD map construction,
multi-object tracking, and temporal consistency.

Usage:
    from common.metrics import compute_metrics, METRIC_REGISTRY

    # Compute all detection metrics
    results = compute_metrics('detection', predictions=preds, ground_truths=gts)

    # Compute specific metric
    from common.metrics.detection_metrics import compute_map
    mAP_result = compute_map(predictions, ground_truths)
"""

from typing import Any, Callable, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------
from common.metrics.detection_metrics import (
    compute_map,
    compute_nds,
    compute_translation_error as compute_ate,
    compute_scale_error as compute_ase,
    compute_orientation_error as compute_aoe,
    compute_velocity_error as compute_ave,
    compute_attribute_error as compute_aae,
    evaluate_detection as compute_detection_metrics,
    compute_precision_recall_curve,
    compute_ap_per_class as compute_detection_ap_per_class,
    compute_tp_errors,
)

# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------
from common.metrics.segmentation_metrics import (
    mean_iou as compute_miou,
    per_class_iou as compute_per_class_iou,
    pixel_accuracy as compute_pixel_accuracy,
    frequency_weighted_iou as compute_frequency_weighted_iou,
    dice_coefficient as compute_dice_coefficient,
    mean_dice as compute_mean_dice,
    boundary_iou as compute_boundary_iou,
    mean_boundary_iou as compute_mean_boundary_iou,
    compute_all_metrics as compute_segmentation_metrics,
    compute_confusion_matrix,
    compute_iou_from_confusion_matrix,
    compute_dice_from_confusion_matrix,
)

# ---------------------------------------------------------------------------
# HD Map metrics
# ---------------------------------------------------------------------------
from common.metrics.map_metrics import (
    chamfer_distance as compute_chamfer_distance,
    frechet_distance as compute_frechet_distance,
    compute_ap_per_class as compute_map_ap,
    evaluate_frame as compute_map_frame,
    evaluate_batch as compute_map_batch,
    evaluate as compute_map_metrics,
    compute_chamfer_matrix,
    hungarian_match as compute_hungarian_match,
)

# Alias for per-class map AP (same function, different name for registry)
compute_per_class_map_ap = compute_map_ap

# ---------------------------------------------------------------------------
# Tracking metrics
# ---------------------------------------------------------------------------
from common.metrics.tracking_metrics import (
    compute_amota_amotp,
    compute_idf1 as _compute_idf1_internal,
    compute_mota as _compute_mota_internal,
    compute_id_switches as _compute_id_switches_internal,
    compute_fragmentations as _compute_fragmentations_internal,
    compute_mt_ml as _compute_mt_ml_internal,
    evaluate_sequence as compute_tracking_sequence,
    evaluate_batch as compute_tracking_batch,
)


def compute_amota(**kwargs: Any) -> Dict[str, float]:
    """Compute AMOTA (Average Multi-Object Tracking Accuracy).

    Wrapper that calls evaluate_sequence and extracts AMOTA.
    Accepts the same kwargs as evaluate_sequence.
    """
    from common.metrics.tracking_metrics import evaluate_sequence
    result = evaluate_sequence(**kwargs)
    return {"AMOTA": result["amota"]}


def compute_amotp(**kwargs: Any) -> Dict[str, float]:
    """Compute AMOTP (Average Multi-Object Tracking Precision).

    Wrapper that calls evaluate_sequence and extracts AMOTP.
    Accepts the same kwargs as evaluate_sequence.
    """
    from common.metrics.tracking_metrics import evaluate_sequence
    result = evaluate_sequence(**kwargs)
    return {"AMOTP": result["amotp"]}


def compute_idf1(**kwargs: Any) -> Dict[str, float]:
    """Compute IDF1 (ID F1 Score).

    Wrapper that calls evaluate_sequence and extracts IDF1.
    Accepts the same kwargs as evaluate_sequence.
    """
    from common.metrics.tracking_metrics import evaluate_sequence
    result = evaluate_sequence(**kwargs)
    return {"IDF1": result["idf1"]}


def compute_mota(**kwargs: Any) -> Dict[str, float]:
    """Compute MOTA (Multi-Object Tracking Accuracy).

    Wrapper that calls evaluate_sequence and extracts MOTA.
    Accepts the same kwargs as evaluate_sequence.
    """
    from common.metrics.tracking_metrics import evaluate_sequence
    result = evaluate_sequence(**kwargs)
    return {"MOTA": result["mota"]}


def compute_id_switches(**kwargs: Any) -> Dict[str, int]:
    """Compute ID switch count.

    Wrapper that calls evaluate_sequence and extracts id_switches.
    Accepts the same kwargs as evaluate_sequence.
    """
    from common.metrics.tracking_metrics import evaluate_sequence
    result = evaluate_sequence(**kwargs)
    return {"id_switches": result["id_switches"]}


def compute_track_fragmentation(**kwargs: Any) -> Dict[str, int]:
    """Compute track fragmentation count.

    Wrapper that calls evaluate_sequence and extracts fragmentations.
    Accepts the same kwargs as evaluate_sequence.
    """
    from common.metrics.tracking_metrics import evaluate_sequence
    result = evaluate_sequence(**kwargs)
    return {"fragmentations": result["fragmentations"]}


def compute_mostly_tracked_lost(**kwargs: Any) -> Dict[str, float]:
    """Compute mostly tracked / mostly lost ratios.

    Wrapper that calls evaluate_sequence and extracts MT/ML.
    Accepts the same kwargs as evaluate_sequence.
    """
    from common.metrics.tracking_metrics import evaluate_sequence
    result = evaluate_sequence(**kwargs)
    return {
        "mt_ratio": result["mt_ratio"],
        "ml_ratio": result["ml_ratio"],
        "num_mt": result["num_mt"],
        "num_ml": result["num_ml"],
        "num_gt_tracks": result["num_gt_tracks"],
    }


def compute_tracking_metrics(**kwargs: Any) -> Dict[str, Any]:
    """Compute all tracking metrics.

    Auto-dispatches to evaluate_sequence (single sequence) or
    evaluate_batch (multiple sequences) based on input structure.
    Accepts the same kwargs as evaluate_sequence or evaluate_batch.
    """
    from common.metrics.tracking_metrics import evaluate_sequence, evaluate_batch

    # Detect batch vs single sequence
    preds = kwargs.get("predictions")
    if preds is not None and len(preds) > 0:
        first = preds[0]
        # If first element is also a sequence of dicts, it's a batch
        if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], dict):
            return evaluate_batch(
                batch_predictions=kwargs["predictions"],
                batch_ground_truths=kwargs["ground_truths"],
                distance_threshold=kwargs.get("distance_threshold", 2.0),
                num_amota_thresholds=kwargs.get("num_amota_thresholds", 40),
                score_threshold=kwargs.get("score_threshold", 0.0),
            )

    return evaluate_sequence(
        predictions=kwargs.get("predictions", []),
        ground_truths=kwargs.get("ground_truths", []),
        distance_threshold=kwargs.get("distance_threshold", 2.0),
        num_amota_thresholds=kwargs.get("num_amota_thresholds", 40),
        score_threshold=kwargs.get("score_threshold", 0.0),
    )


# ---------------------------------------------------------------------------
# Temporal consistency metrics
# ---------------------------------------------------------------------------
from common.metrics.temporal_metrics import (
    compute_map_consistency,
    compute_streaming_ap,
    compute_temporal_smoothness,
    compute_velocity_consistency,
    compute_all_temporal_metrics as compute_temporal_metrics,
    compute_all_temporal_metrics_batched as compute_temporal_metrics_batched,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Type alias for metric computation functions
MetricFunction = Callable[..., Dict[str, Any]]

# Global metric registry mapping task names to their compute functions
METRIC_REGISTRY: Dict[str, MetricFunction] = {
    "detection": compute_detection_metrics,
    "segmentation": compute_segmentation_metrics,
    "map": compute_map_metrics,
    "tracking": compute_tracking_metrics,
    "temporal": compute_temporal_metrics,
}

# Mapping of individual metric names to their compute functions
INDIVIDUAL_METRICS: Dict[str, MetricFunction] = {
    # Detection metrics
    "mAP": compute_map,
    "NDS": compute_nds,
    "ATE": compute_ate,
    "ASE": compute_ase,
    "AOE": compute_aoe,
    "AVE": compute_ave,
    "AAE": compute_aae,
    # Segmentation metrics
    "mIoU": compute_miou,
    "per_class_iou": compute_per_class_iou,
    "pixel_accuracy": compute_pixel_accuracy,
    "freq_weighted_iou": compute_frequency_weighted_iou,
    "dice": compute_dice_coefficient,
    "mean_dice": compute_mean_dice,
    "boundary_iou": compute_boundary_iou,
    "mean_boundary_iou": compute_mean_boundary_iou,
    # Map metrics
    "chamfer_distance": compute_chamfer_distance,
    "frechet_distance": compute_frechet_distance,
    "map_ap": compute_map_ap,
    "per_class_map_ap": compute_per_class_map_ap,
    # Tracking metrics
    "AMOTA": compute_amota,
    "AMOTP": compute_amotp,
    "IDF1": compute_idf1,
    "MOTA": compute_mota,
    "id_switches": compute_id_switches,
    "track_fragmentation": compute_track_fragmentation,
    "mostly_tracked_lost": compute_mostly_tracked_lost,
    # Temporal metrics
    "map_consistency": compute_map_consistency,
    "streaming_ap": compute_streaming_ap,
    "temporal_smoothness": compute_temporal_smoothness,
    "velocity_consistency": compute_velocity_consistency,
}


def compute_metrics(
    task: str,
    *,
    metrics: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Unified metric computation dispatcher.

    Computes evaluation metrics for a given perception task. Can compute all
    metrics for a task or a specific subset.

    Args:
        task: The perception task type. One of:
            - 'detection': 3D object detection (nuScenes-style)
            - 'segmentation': Semantic/panoptic segmentation
            - 'map': Vectorized HD map construction
            - 'tracking': Multi-object tracking
            - 'temporal': Temporal consistency metrics
        metrics: Optional list of specific metric names to compute.
            If None, computes all metrics for the given task.
        **kwargs: Task-specific keyword arguments passed to the metric
            computation function. See individual metric modules for details.

    Returns:
        Dictionary mapping metric names to their computed values.
        Values may be scalars (float), arrays (numpy), or nested dicts
        depending on the metric.

    Raises:
        ValueError: If task is not recognized or metrics list contains
            unknown metric names.

    Examples:
        >>> # Compute all detection metrics
        >>> results = compute_metrics(
        ...     'detection',
        ...     predictions=pred_list,
        ...     ground_truths=gt_list,
        ... )
        >>> print(results['NDS'])
        0.65

        >>> # Compute only mIoU for segmentation
        >>> results = compute_metrics(
        ...     'segmentation',
        ...     metrics=['mIoU'],
        ...     predictions=pred_masks,
        ...     ground_truths=gt_masks,
        ...     num_classes=16,
        ... )
    """
    if task not in METRIC_REGISTRY:
        available = ", ".join(sorted(METRIC_REGISTRY.keys()))
        raise ValueError(
            f"Unknown task '{task}'. Available tasks: {available}"
        )

    if metrics is not None:
        # Compute only the requested subset of metrics
        results: Dict[str, Any] = {}
        for metric_name in metrics:
            if metric_name in INDIVIDUAL_METRICS:
                results[metric_name] = INDIVIDUAL_METRICS[metric_name](**kwargs)
            else:
                available_metrics = ", ".join(sorted(INDIVIDUAL_METRICS.keys()))
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    f"Available metrics: {available_metrics}"
                )
        return results

    # Compute all metrics for the task
    compute_fn = METRIC_REGISTRY[task]
    return compute_fn(**kwargs)


def list_available_tasks() -> List[str]:
    """Return a list of all available perception task names."""
    return sorted(METRIC_REGISTRY.keys())


def list_available_metrics(task: Optional[str] = None) -> List[str]:
    """
    Return a list of available metric names.

    Args:
        task: If provided, returns only metrics relevant to this task.
            If None, returns all available metric names.

    Returns:
        Sorted list of metric name strings.
    """
    if task is None:
        return sorted(INDIVIDUAL_METRICS.keys())

    # Map tasks to their metric subsets
    task_metrics = {
        "detection": ["mAP", "NDS", "ATE", "ASE", "AOE", "AVE", "AAE"],
        "segmentation": [
            "mIoU", "per_class_iou", "pixel_accuracy",
            "freq_weighted_iou", "dice", "mean_dice",
            "boundary_iou", "mean_boundary_iou",
        ],
        "map": [
            "chamfer_distance", "frechet_distance",
            "map_ap", "per_class_map_ap",
        ],
        "tracking": [
            "AMOTA", "AMOTP", "IDF1", "MOTA",
            "id_switches", "track_fragmentation", "mostly_tracked_lost",
        ],
        "temporal": [
            "map_consistency", "streaming_ap",
            "temporal_smoothness", "velocity_consistency",
        ],
    }

    if task not in task_metrics:
        available = ", ".join(sorted(task_metrics.keys()))
        raise ValueError(
            f"Unknown task '{task}'. Available tasks: {available}"
        )

    return sorted(task_metrics[task])


__all__ = [
    # Registry and dispatcher
    "compute_metrics",
    "METRIC_REGISTRY",
    "INDIVIDUAL_METRICS",
    "list_available_tasks",
    "list_available_metrics",
    # Detection metrics
    "compute_map",
    "compute_nds",
    "compute_ate",
    "compute_ase",
    "compute_aoe",
    "compute_ave",
    "compute_aae",
    "compute_detection_metrics",
    "compute_precision_recall_curve",
    "compute_detection_ap_per_class",
    "compute_tp_errors",
    # Segmentation metrics
    "compute_miou",
    "compute_per_class_iou",
    "compute_pixel_accuracy",
    "compute_frequency_weighted_iou",
    "compute_dice_coefficient",
    "compute_mean_dice",
    "compute_boundary_iou",
    "compute_mean_boundary_iou",
    "compute_segmentation_metrics",
    "compute_confusion_matrix",
    "compute_iou_from_confusion_matrix",
    "compute_dice_from_confusion_matrix",
    # Map metrics
    "compute_chamfer_distance",
    "compute_frechet_distance",
    "compute_map_ap",
    "compute_per_class_map_ap",
    "compute_map_frame",
    "compute_map_batch",
    "compute_map_metrics",
    "compute_chamfer_matrix",
    "compute_hungarian_match",
    # Tracking metrics
    "compute_amota",
    "compute_amotp",
    "compute_idf1",
    "compute_mota",
    "compute_id_switches",
    "compute_track_fragmentation",
    "compute_mostly_tracked_lost",
    "compute_tracking_metrics",
    "compute_tracking_sequence",
    "compute_tracking_batch",
    "compute_amota_amotp",
    # Temporal metrics
    "compute_map_consistency",
    "compute_streaming_ap",
    "compute_temporal_smoothness",
    "compute_velocity_consistency",
    "compute_temporal_metrics",
    "compute_temporal_metrics_batched",
]
