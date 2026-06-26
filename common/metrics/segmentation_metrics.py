"""Segmentation metrics for autonomous driving perception models.

This module provides efficient, confusion-matrix-based computation of standard
segmentation metrics for both 2D (BEV/image) and 3D (voxel/occupancy) tasks.

Supported metrics:
    - Mean Intersection over Union (mIoU)
    - Per-class IoU
    - Pixel/voxel accuracy
    - Frequency-weighted IoU (FWIoU)
    - Dice coefficient (per-class and mean)
    - Boundary IoU (IoU restricted to boundary pixels/voxels)

All functions accept integer label arrays (predictions and ground truth) and
support configurable ignore indices. Inputs may be single samples or batched.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage


# Type aliases for clarity
LabelArray = np.ndarray  # integer array of shape (H, W) or (D, H, W)
BatchedInput = Union[np.ndarray, List[np.ndarray]]


def _normalize_inputs(
    predictions: BatchedInput,
    ground_truth: BatchedInput,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Normalize inputs to lists of individual samples.

    Handles:
        - Single arrays of shape (H, W) or (D, H, W): treated as one sample.
        - Batched arrays of shape (N, H, W) or (N, D, H, W): split along axis 0.
        - Lists of arrays: used directly.

    Parameters
    ----------
    predictions : BatchedInput
        Predicted label array(s).
    ground_truth : BatchedInput
        Ground truth label array(s).

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        Lists of per-sample prediction and ground-truth arrays.

    Raises
    ------
    ValueError
        If the number of prediction and ground-truth samples do not match.
    """
    def _to_list(arr: BatchedInput) -> List[np.ndarray]:
        if isinstance(arr, list):
            return arr
        arr = np.asarray(arr)
        # 2D single sample: (H, W)
        if arr.ndim == 2:
            return [arr]
        # 3D: could be (D, H, W) single volumetric sample or (N, H, W) batch
        # Heuristic: if accompanying array is also 3D, treat as single sample.
        # We defer this decision to the caller pair logic below.
        if arr.ndim == 3:
            return [arr]  # default: single sample; batch logic handled below
        # 4D: (N, D, H, W) batch of volumetric samples
        if arr.ndim == 4:
            return [arr[i] for i in range(arr.shape[0])]
        return [arr]

    # Handle list inputs directly
    if isinstance(predictions, list) and isinstance(ground_truth, list):
        preds_list = predictions
        gts_list = ground_truth
    else:
        pred_arr = np.asarray(predictions)
        gt_arr = np.asarray(ground_truth)

        # Detect batch dimension: if both are 3D and shapes match, check if
        # it's a batch of 2D samples (N, H, W) by comparing with each other.
        # If both are 3D with same shape, treat as single volumetric sample
        # UNLESS the first dimension is clearly a batch (mismatching spatial dims
        # would indicate something else, but we keep it simple).
        # Convention: use explicit list input for ambiguous 3D batches.
        if pred_arr.ndim == 3 and gt_arr.ndim == 3:
            # Treat as single 3D volumetric sample
            preds_list = [pred_arr]
            gts_list = [gt_arr]
        elif pred_arr.ndim == 4 and gt_arr.ndim == 4:
            preds_list = [pred_arr[i] for i in range(pred_arr.shape[0])]
            gts_list = [gt_arr[i] for i in range(gt_arr.shape[0])]
        else:
            preds_list = _to_list(pred_arr)
            gts_list = _to_list(gt_arr)

    if len(preds_list) != len(gts_list):
        raise ValueError(
            f"Number of prediction samples ({len(preds_list)}) does not match "
            f"ground truth samples ({len(gts_list)})."
        )
    return preds_list, gts_list


def compute_confusion_matrix(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> np.ndarray:
    """Compute a confusion matrix from flat label arrays.

    Parameters
    ----------
    predictions : np.ndarray
        1-D array of predicted labels.
    ground_truth : np.ndarray
        1-D array of ground-truth labels.
    num_classes : int
        Total number of valid classes (labels 0..num_classes-1).
    ignore_index : int, optional
        Label value to ignore (e.g., 255). Pixels with this label in the
        ground truth are excluded from computation.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (num_classes, num_classes) where entry
        [i, j] counts the number of pixels with true label i predicted as j.
    """
    pred_flat = predictions.ravel().astype(np.int64)
    gt_flat = ground_truth.ravel().astype(np.int64)

    # Mask out ignored labels
    if ignore_index is not None:
        valid_mask = gt_flat != ignore_index
        pred_flat = pred_flat[valid_mask]
        gt_flat = gt_flat[valid_mask]

    # Also mask predictions that might be ignore_index (optional safety)
    # and any out-of-range labels
    valid_mask = (
        (gt_flat >= 0) & (gt_flat < num_classes) &
        (pred_flat >= 0) & (pred_flat < num_classes)
    )
    pred_flat = pred_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]

    # Efficient confusion matrix via linear indexing
    indices = gt_flat * num_classes + pred_flat
    cm = np.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def compute_iou_from_confusion_matrix(
    confusion_matrix: np.ndarray,
) -> np.ndarray:
    """Compute per-class IoU from a confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Square confusion matrix of shape (num_classes, num_classes).

    Returns
    -------
    np.ndarray
        Per-class IoU values. Classes with zero union get IoU = 0.
    """
    tp = np.diag(confusion_matrix).astype(np.float64)
    # Union = sum of row (all GT for class) + sum of col (all pred for class) - TP
    row_sum = confusion_matrix.sum(axis=1).astype(np.float64)
    col_sum = confusion_matrix.sum(axis=0).astype(np.float64)
    union = row_sum + col_sum - tp

    # Avoid division by zero: classes not present get IoU = 0
    iou = np.where(union > 0, tp / union, 0.0)
    return iou


def compute_dice_from_confusion_matrix(
    confusion_matrix: np.ndarray,
) -> np.ndarray:
    """Compute per-class Dice coefficient from a confusion matrix.

    Dice = 2 * TP / (2 * TP + FP + FN) = 2 * TP / (row_sum + col_sum)

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Square confusion matrix of shape (num_classes, num_classes).

    Returns
    -------
    np.ndarray
        Per-class Dice coefficients. Classes absent from both prediction and
        ground truth get Dice = 0.
    """
    tp = np.diag(confusion_matrix).astype(np.float64)
    row_sum = confusion_matrix.sum(axis=1).astype(np.float64)
    col_sum = confusion_matrix.sum(axis=0).astype(np.float64)
    denom = row_sum + col_sum

    dice = np.where(denom > 0, 2.0 * tp / denom, 0.0)
    return dice


def per_class_iou(
    predictions: BatchedInput,
    ground_truth: BatchedInput,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> np.ndarray:
    """Compute per-class IoU aggregated over all samples.

    Parameters
    ----------
    predictions : BatchedInput
        Predicted label array(s). Shape (H, W), (D, H, W), or batched.
    ground_truth : BatchedInput
        Ground truth label array(s). Same shape constraints as predictions.
    num_classes : int
        Number of valid classes.
    ignore_index : int, optional
        Label to ignore in ground truth.

    Returns
    -------
    np.ndarray
        Array of shape (num_classes,) with per-class IoU.
    """
    preds_list, gts_list = _normalize_inputs(predictions, ground_truth)

    # Accumulate a global confusion matrix across all samples
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(preds_list, gts_list):
        cm += compute_confusion_matrix(pred, gt, num_classes, ignore_index)

    return compute_iou_from_confusion_matrix(cm)


def mean_iou(
    predictions: BatchedInput,
    ground_truth: BatchedInput,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> float:
    """Compute mean Intersection over Union (mIoU) across all classes.

    Only classes present in the ground truth (with nonzero union) contribute
    to the mean.

    Parameters
    ----------
    predictions : BatchedInput
        Predicted label array(s).
    ground_truth : BatchedInput
        Ground truth label array(s).
    num_classes : int
        Number of valid classes.
    ignore_index : int, optional
        Label to ignore.

    Returns
    -------
    float
        Mean IoU across classes present in the data.
    """
    iou = per_class_iou(predictions, ground_truth, num_classes, ignore_index)

    # Only average over classes that are present (nonzero union)
    preds_list, gts_list = _normalize_inputs(predictions, ground_truth)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(preds_list, gts_list):
        cm += compute_confusion_matrix(pred, gt, num_classes, ignore_index)

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    union = row_sum + col_sum - np.diag(cm)
    present_classes = union > 0

    if not np.any(present_classes):
        return 0.0
    return float(np.mean(iou[present_classes]))


def pixel_accuracy(
    predictions: BatchedInput,
    ground_truth: BatchedInput,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> float:
    """Compute overall pixel (or voxel) accuracy.

    Accuracy = total correct predictions / total valid pixels.

    Parameters
    ----------
    predictions : BatchedInput
        Predicted label array(s).
    ground_truth : BatchedInput
        Ground truth label array(s).
    num_classes : int
        Number of valid classes.
    ignore_index : int, optional
        Label to ignore.

    Returns
    -------
    float
        Overall accuracy in [0, 1].
    """
    preds_list, gts_list = _normalize_inputs(predictions, ground_truth)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(preds_list, gts_list):
        cm += compute_confusion_matrix(pred, gt, num_classes, ignore_index)

    total_correct = np.diag(cm).sum()
    total_pixels = cm.sum()

    if total_pixels == 0:
        return 0.0
    return float(total_correct / total_pixels)


def frequency_weighted_iou(
    predictions: BatchedInput,
    ground_truth: BatchedInput,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> float:
    """Compute frequency-weighted IoU (FWIoU).

    Each class's IoU is weighted by the fraction of pixels belonging to that
    class in the ground truth.

    FWIoU = sum_k (freq_k * IoU_k)  where freq_k = n_k / sum(n_k)

    Parameters
    ----------
    predictions : BatchedInput
        Predicted label array(s).
    ground_truth : BatchedInput
        Ground truth label array(s).
    num_classes : int
        Number of valid classes.
    ignore_index : int, optional
        Label to ignore.

    Returns
    -------
    float
        Frequency-weighted IoU.
    """
    preds_list, gts_list = _normalize_inputs(predictions, ground_truth)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(preds_list, gts_list):
        cm += compute_confusion_matrix(pred, gt, num_classes, ignore_index)

    iou = compute_iou_from_confusion_matrix(cm)
    freq = cm.sum(axis=1).astype(np.float64)  # per-class pixel count in GT
    total = freq.sum()

    if total == 0:
        return 0.0

    freq_normalized = freq / total
    # Only weight classes with nonzero presence
    fwiou = float(np.sum(freq_normalized * iou))
    return fwiou


def dice_coefficient(
    predictions: BatchedInput,
    ground_truth: BatchedInput,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> np.ndarray:
    """Compute per-class Dice coefficient (F1 score for segmentation).

    Parameters
    ----------
    predictions : BatchedInput
        Predicted label array(s).
    ground_truth : BatchedInput
        Ground truth label array(s).
    num_classes : int
        Number of valid classes.
    ignore_index : int, optional
        Label to ignore.

    Returns
    -------
    np.ndarray
        Array of shape (num_classes,) with per-class Dice scores.
    """
    preds_list, gts_list = _normalize_inputs(predictions, ground_truth)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(preds_list, gts_list):
        cm += compute_confusion_matrix(pred, gt, num_classes, ignore_index)

    return compute_dice_from_confusion_matrix(cm)


def mean_dice(
    predictions: BatchedInput,
    ground_truth: BatchedInput,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> float:
    """Compute mean Dice coefficient across present classes.

    Parameters
    ----------
    predictions : BatchedInput
        Predicted label array(s).
    ground_truth : BatchedInput
        Ground truth label array(s).
    num_classes : int
        Number of valid classes.
    ignore_index : int, optional
        Label to ignore.

    Returns
    -------
    float
        Mean Dice coefficient across classes with nonzero presence.
    """
    preds_list, gts_list = _normalize_inputs(predictions, ground_truth)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(preds_list, gts_list):
        cm += compute_confusion_matrix(pred, gt, num_classes, ignore_index)

    dice = compute_dice_from_confusion_matrix(cm)

    # Determine present classes
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    present = (row_sum + col_sum) > 0

    if not np.any(present):
        return 0.0
    return float(np.mean(dice[present]))


def _extract_boundaries(
    label_map: np.ndarray,
    num_classes: int,
    boundary_width: int = 1,
    ignore_index: Optional[int] = None,
) -> np.ndarray:
    """Extract boundary mask from a label map using morphological operations.

    For each class, the boundary is defined as the set of pixels that are within
    `boundary_width` pixels of the class edge: boundary = dilation - erosion.

    Parameters
    ----------
    label_map : np.ndarray
        Integer label array of shape (H, W) or (D, H, W).
    num_classes : int
        Number of valid classes.
    boundary_width : int
        Width (in pixels/voxels) of the boundary region. The structuring element
        has radius = boundary_width (i.e., size = 2 * boundary_width + 1).
    ignore_index : int, optional
        Label to ignore.

    Returns
    -------
    np.ndarray
        Boolean mask of the same shape as label_map, True at boundary pixels.
    """
    ndim = label_map.ndim
    # Create structuring element (ball/disk of given radius)
    if ndim == 2:
        struct = ndimage.generate_binary_structure(2, 1)  # cross connectivity
        struct = ndimage.iterate_structure(struct, boundary_width)
    elif ndim == 3:
        struct = ndimage.generate_binary_structure(3, 1)
        struct = ndimage.iterate_structure(struct, boundary_width)
    else:
        raise ValueError(
            f"Label map must be 2D or 3D, got {ndim}D array."
        )

    boundary_mask = np.zeros(label_map.shape, dtype=bool)

    for cls_id in range(num_classes):
        cls_mask = (label_map == cls_id)
        if not np.any(cls_mask):
            continue

        # Dilate and erode
        dilated = ndimage.binary_dilation(cls_mask, structure=struct)
        eroded = ndimage.binary_erosion(cls_mask, structure=struct)

        # Boundary = dilated XOR eroded region intersected with original class
        # More precisely: boundary = dilation(mask) & ~erosion(mask)
        # This gives the "thick boundary" of width boundary_width
        cls_boundary = dilated & ~eroded
        boundary_mask |= cls_boundary

    # Exclude ignored pixels from the boundary mask
    if ignore_index is not None:
        boundary_mask &= (label_map != ignore_index)

    return boundary_mask


def boundary_iou(
    predictions: BatchedInput,
    ground_truth: BatchedInput,
    num_classes: int,
    ignore_index: Optional[int] = None,
    boundary_width: int = 1,
) -> np.ndarray:
    """Compute Boundary IoU: IoU restricted to boundary pixels/voxels.

    This metric evaluates how well the model predicts edges/boundaries between
    semantic regions, which is critical for autonomous driving (e.g., lane
    boundaries, pedestrian silhouettes).

    The boundary region is extracted from the ground truth using morphological
    dilation minus erosion. IoU is then computed only within those boundary
    pixels.

    Parameters
    ----------
    predictions : BatchedInput
        Predicted label array(s).
    ground_truth : BatchedInput
        Ground truth label array(s).
    num_classes : int
        Number of valid classes.
    ignore_index : int, optional
        Label to ignore.
    boundary_width : int
        Width of the boundary band in pixels/voxels (default: 1).

    Returns
    -------
    np.ndarray
        Array of shape (num_classes,) with per-class Boundary IoU.
    """
    preds_list, gts_list = _normalize_inputs(predictions, ground_truth)

    cm_boundary = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred, gt in zip(preds_list, gts_list):
        # Extract boundary regions from ground truth
        gt_boundary_mask = _extract_boundaries(
            gt, num_classes, boundary_width, ignore_index
        )
        # Also extract boundary from predictions for symmetric evaluation
        pred_boundary_mask = _extract_boundaries(
            pred, num_classes, boundary_width, ignore_index
        )
        # Combined boundary region: union of GT and pred boundaries
        combined_boundary = gt_boundary_mask | pred_boundary_mask

        if not np.any(combined_boundary):
            continue

        # Restrict evaluation to boundary pixels only
        pred_boundary = pred[combined_boundary]
        gt_boundary = gt[combined_boundary]

        cm_boundary += compute_confusion_matrix(
            pred_boundary, gt_boundary, num_classes, ignore_index
        )

    return compute_iou_from_confusion_matrix(cm_boundary)


def mean_boundary_iou(
    predictions: BatchedInput,
    ground_truth: BatchedInput,
    num_classes: int,
    ignore_index: Optional[int] = None,
    boundary_width: int = 1,
) -> float:
    """Compute mean Boundary IoU across present classes.

    Parameters
    ----------
    predictions : BatchedInput
        Predicted label array(s).
    ground_truth : BatchedInput
        Ground truth label array(s).
    num_classes : int
        Number of valid classes.
    ignore_index : int, optional
        Label to ignore.
    boundary_width : int
        Width of the boundary band in pixels/voxels (default: 1).

    Returns
    -------
    float
        Mean Boundary IoU across classes with nonzero boundary presence.
    """
    biou = boundary_iou(
        predictions, ground_truth, num_classes, ignore_index, boundary_width
    )
    # Determine which classes have boundary pixels
    preds_list, gts_list = _normalize_inputs(predictions, ground_truth)
    cm_boundary = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(preds_list, gts_list):
        gt_boundary_mask = _extract_boundaries(
            gt, num_classes, boundary_width, ignore_index
        )
        pred_boundary_mask = _extract_boundaries(
            pred, num_classes, boundary_width, ignore_index
        )
        combined_boundary = gt_boundary_mask | pred_boundary_mask
        if not np.any(combined_boundary):
            continue
        pred_boundary = pred[combined_boundary]
        gt_boundary = gt[combined_boundary]
        cm_boundary += compute_confusion_matrix(
            pred_boundary, gt_boundary, num_classes, ignore_index
        )

    row_sum = cm_boundary.sum(axis=1)
    col_sum = cm_boundary.sum(axis=0)
    union = row_sum + col_sum - np.diag(cm_boundary)
    present = union > 0

    if not np.any(present):
        return 0.0
    return float(np.mean(biou[present]))


def compute_all_metrics(
    predictions: BatchedInput,
    ground_truth: BatchedInput,
    num_classes: int,
    ignore_index: Optional[int] = None,
    boundary_width: int = 1,
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute all segmentation metrics in one pass.

    This is the recommended entry point for evaluation. It computes a global
    confusion matrix once and derives all metrics from it, plus a separate
    boundary confusion matrix for Boundary IoU.

    Parameters
    ----------
    predictions : BatchedInput
        Predicted label array(s). Supports:
        - Single 2D array of shape (H, W)
        - Single 3D array of shape (D, H, W) for volumetric/occupancy
        - Batched 4D array of shape (N, D, H, W)
        - List of arrays
    ground_truth : BatchedInput
        Ground truth label array(s). Same shape constraints as predictions.
    num_classes : int
        Number of valid semantic classes (labels 0 to num_classes - 1).
    ignore_index : int, optional
        Label value to exclude from evaluation (e.g., 255 for unlabeled).
    boundary_width : int
        Pixel/voxel width of the boundary band for Boundary IoU (default: 1).

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary containing:
        - ``"per_class_iou"``: np.ndarray of shape (num_classes,)
        - ``"mean_iou"``: float
        - ``"pixel_accuracy"``: float
        - ``"frequency_weighted_iou"``: float
        - ``"per_class_dice"``: np.ndarray of shape (num_classes,)
        - ``"mean_dice"``: float
        - ``"per_class_boundary_iou"``: np.ndarray of shape (num_classes,)
        - ``"mean_boundary_iou"``: float

    Examples
    --------
    >>> import numpy as np
    >>> pred = np.array([[0, 0, 1], [1, 2, 2], [2, 2, 1]], dtype=np.int32)
    >>> gt = np.array([[0, 0, 0], [1, 1, 2], [2, 2, 2]], dtype=np.int32)
    >>> metrics = compute_all_metrics(pred, gt, num_classes=3)
    >>> print(f"mIoU: {metrics['mean_iou']:.4f}")
    >>> print(f"Accuracy: {metrics['pixel_accuracy']:.4f}")
    """
    preds_list, gts_list = _normalize_inputs(predictions, ground_truth)

    # --- Global confusion matrix ---
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(preds_list, gts_list):
        cm += compute_confusion_matrix(pred, gt, num_classes, ignore_index)

    # Per-class IoU
    iou = compute_iou_from_confusion_matrix(cm)

    # Determine present classes (nonzero union)
    tp = np.diag(cm).astype(np.float64)
    row_sum = cm.sum(axis=1).astype(np.float64)
    col_sum = cm.sum(axis=0).astype(np.float64)
    union = row_sum + col_sum - tp
    present_classes = union > 0

    # Mean IoU (over present classes only)
    if np.any(present_classes):
        miou = float(np.mean(iou[present_classes]))
    else:
        miou = 0.0

    # Pixel accuracy
    total_correct = tp.sum()
    total_pixels = cm.sum()
    acc = float(total_correct / total_pixels) if total_pixels > 0 else 0.0

    # Frequency-weighted IoU
    freq = row_sum / total_pixels if total_pixels > 0 else np.zeros(num_classes)
    fwiou = float(np.sum(freq * iou))

    # Dice coefficient
    dice = compute_dice_from_confusion_matrix(cm)
    present_dice = (row_sum + col_sum) > 0
    if np.any(present_dice):
        m_dice = float(np.mean(dice[present_dice]))
    else:
        m_dice = 0.0

    # --- Boundary IoU ---
    cm_boundary = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(preds_list, gts_list):
        gt_boundary_mask = _extract_boundaries(
            gt, num_classes, boundary_width, ignore_index
        )
        pred_boundary_mask = _extract_boundaries(
            pred, num_classes, boundary_width, ignore_index
        )
        combined_boundary = gt_boundary_mask | pred_boundary_mask

        if not np.any(combined_boundary):
            continue

        pred_boundary = pred[combined_boundary]
        gt_boundary = gt[combined_boundary]
        cm_boundary += compute_confusion_matrix(
            pred_boundary, gt_boundary, num_classes, ignore_index
        )

    biou = compute_iou_from_confusion_matrix(cm_boundary)
    b_row_sum = cm_boundary.sum(axis=1)
    b_col_sum = cm_boundary.sum(axis=0)
    b_union = b_row_sum + b_col_sum - np.diag(cm_boundary)
    b_present = b_union > 0

    if np.any(b_present):
        m_biou = float(np.mean(biou[b_present]))
    else:
        m_biou = 0.0

    return {
        "per_class_iou": iou,
        "mean_iou": miou,
        "pixel_accuracy": acc,
        "frequency_weighted_iou": fwiou,
        "per_class_dice": dice,
        "mean_dice": m_dice,
        "per_class_boundary_iou": biou,
        "mean_boundary_iou": m_biou,
    }
