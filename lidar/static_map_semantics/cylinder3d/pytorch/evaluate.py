"""
Evaluation script for Cylinder3D semantic segmentation.

Computes per-class IoU and mIoU on the SemanticKITTI validation or test set.
Optionally saves predictions in SemanticKITTI submission format (.label files).

Usage:
    python -m lidar.static_map_semantics.cylinder3d.pytorch.evaluate \
        --config config.yaml \
        --checkpoint checkpoints/cylinder3d_best.pth \
        --split val

    # Save test predictions for submission:
    python -m lidar.static_map_semantics.cylinder3d.pytorch.evaluate \
        --config config.yaml \
        --checkpoint checkpoints/cylinder3d_best.pth \
        --split test \
        --save_predictions \
        --output_dir predictions/
"""

import os
import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:
    yaml = None

from .model import Cylinder3D
from .dataset import (
    SemanticKITTIDataset,
    SEMANTICKITTI_CLASSES,
    SEMANTICKITTI_LEARNING_MAP,
    build_semantickitti_splits,
    collate_fn,
)


# ==============================================================================
# Inverse Learning Map (learning class -> raw SemanticKITTI label)
# ==============================================================================

def build_inverse_learning_map() -> np.ndarray:
    """
    Build inverse mapping from learning class IDs [0..19] back to original
    SemanticKITTI label IDs for submission.

    The inverse map picks the canonical raw label for each learning class.
    For classes that have multiple raw labels mapping to the same learning ID,
    the first encountered (lowest raw label) is used.

    Returns:
        np.ndarray of shape (20,) with dtype uint32, where index = learning class,
        value = original SemanticKITTI label ID.
    """
    # Canonical mapping: learning class -> raw label
    # We pick the "primary" raw label for each learning class.
    inverse_map = np.zeros(20, dtype=np.uint32)
    inverse_map[0] = 0       # unlabeled
    inverse_map[1] = 10      # car
    inverse_map[2] = 11      # bicycle
    inverse_map[3] = 15      # motorcycle
    inverse_map[4] = 18      # truck
    inverse_map[5] = 20      # other-vehicle
    inverse_map[6] = 30      # person
    inverse_map[7] = 31      # bicyclist
    inverse_map[8] = 32      # motorcyclist
    inverse_map[9] = 40      # road
    inverse_map[10] = 44     # parking
    inverse_map[11] = 48     # sidewalk
    inverse_map[12] = 49     # other-ground
    inverse_map[13] = 50     # building
    inverse_map[14] = 51     # fence
    inverse_map[15] = 70     # vegetation
    inverse_map[16] = 71     # trunk
    inverse_map[17] = 72     # terrain
    inverse_map[18] = 80     # pole
    inverse_map[19] = 81     # traffic-sign
    return inverse_map


# ==============================================================================
# Confusion Matrix and IoU Computation
# ==============================================================================

class ConfusionMatrix:
    """
    Accumulates a confusion matrix for semantic segmentation evaluation.

    The confusion matrix C has shape (num_classes, num_classes) where:
        C[i, j] = number of points with ground truth class i predicted as class j.

    From this matrix, per-class IoU is computed as:
        IoU_c = TP_c / (TP_c + FP_c + FN_c)
    where:
        TP_c = C[c, c]
        FP_c = sum(C[:, c]) - C[c, c]  (predicted as c but not actually c)
        FN_c = sum(C[c, :]) - C[c, c]  (actually c but predicted differently)
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Update confusion matrix with a batch of predictions and targets.

        Args:
            predictions: (N,) int array of predicted class IDs.
            targets: (N,) int array of ground truth class IDs.
        """
        # Filter out invalid indices
        valid_mask = (targets >= 0) & (targets < self.num_classes) & \
                     (predictions >= 0) & (predictions < self.num_classes)
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        # Compute linear indices into the flattened confusion matrix
        indices = targets * self.num_classes + predictions
        # Use bincount to accumulate
        counts = np.bincount(indices, minlength=self.num_classes * self.num_classes)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute_iou(self, ignore_class: int = 0) -> Tuple[np.ndarray, float]:
        """
        Compute per-class IoU and mean IoU (excluding the ignore class).

        Args:
            ignore_class: Class index to exclude from mIoU calculation.
                          Typically 0 = 'unlabeled'.

        Returns:
            per_class_iou: (num_classes,) array of per-class IoU values.
                           NaN for classes with no ground truth or predictions.
            miou: Mean IoU over all valid classes excluding the ignore class.
        """
        per_class_iou = np.zeros(self.num_classes, dtype=np.float64)

        for c in range(self.num_classes):
            tp = self.matrix[c, c]
            fp = self.matrix[:, c].sum() - tp
            fn = self.matrix[c, :].sum() - tp
            denominator = tp + fp + fn

            if denominator == 0:
                per_class_iou[c] = np.nan
            else:
                per_class_iou[c] = tp / denominator

        # Compute mIoU excluding ignore class
        valid_classes = [
            c for c in range(self.num_classes)
            if c != ignore_class and not np.isnan(per_class_iou[c])
        ]
        if len(valid_classes) > 0:
            miou = np.mean(per_class_iou[valid_classes])
        else:
            miou = 0.0

        return per_class_iou, miou

    def reset(self):
        """Reset the confusion matrix to zeros."""
        self.matrix.fill(0)


# ==============================================================================
# Results Formatting
# ==============================================================================

def print_results_table(
    per_class_iou: np.ndarray,
    class_names: List[str],
    miou: float,
    ignore_class: int = 0,
):
    """
    Print a formatted results table with per-class IoU and overall mIoU.

    Args:
        per_class_iou: (num_classes,) array of IoU values.
        class_names: List of class name strings.
        miou: Mean IoU value.
        ignore_class: Class index that was excluded from mIoU.
    """
    print("\n" + "=" * 60)
    print("  Cylinder3D Evaluation Results")
    print("=" * 60)
    print(f"  {'Class':<20} {'IoU (%)':<12} {'Status'}")
    print("-" * 60)

    for c in range(len(class_names)):
        name = class_names[c]
        iou_val = per_class_iou[c]

        if c == ignore_class:
            status = "(ignored)"
            iou_str = "---"
        elif np.isnan(iou_val):
            status = "(no samples)"
            iou_str = "N/A"
        else:
            status = ""
            iou_str = f"{iou_val * 100:.2f}"

        print(f"  {name:<20} {iou_str:<12} {status}")

    print("-" * 60)
    print(f"  {'mIoU':<20} {miou * 100:.2f}%")
    print(f"  (excluding class {ignore_class}: '{class_names[ignore_class]}')")
    print("=" * 60 + "\n")


# ==============================================================================
# Prediction Saving (SemanticKITTI Format)
# ==============================================================================

def save_predictions_semantickitti(
    predictions: np.ndarray,
    output_path: str,
    inverse_map: np.ndarray,
):
    """
    Save predictions as a .label file in SemanticKITTI format.

    SemanticKITTI .label format: N x uint32 where lower 16 bits = semantic label,
    upper 16 bits = instance ID (set to 0 for semantic-only predictions).

    Args:
        predictions: (N,) int array of learning class predictions [0..19].
        output_path: Path to save the .label file.
        inverse_map: (20,) uint32 array mapping learning class -> raw label.
    """
    # Map learning classes back to original SemanticKITTI labels
    raw_labels = inverse_map[predictions].astype(np.uint32)

    # SemanticKITTI format: lower 16 bits = semantic, upper 16 bits = instance (0)
    label_data = raw_labels.astype(np.uint32)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write binary file
    label_data.tofile(output_path)


# ==============================================================================
# Configuration Loading
# ==============================================================================

def load_config(config_path: str) -> Dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required for loading config files. "
            "Install with: pip install pyyaml"
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_default_config() -> Dict:
    """
    Return default configuration when no config file is provided.

    Returns:
        Default configuration dictionary.
    """
    return {
        'model': {
            'num_classes': 20,
            'grid_size': [480, 360, 32],
            'base_channels': 32,
            'use_point_refinement': True,
        },
        'dataset': {
            'root': './data/semantickitti',
            'max_points': None,
        },
        'evaluation': {
            'batch_size': 1,
            'num_workers': 4,
        },
    }


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model(
    config: Dict,
    checkpoint_path: str,
    device: torch.device,
) -> Cylinder3D:
    """
    Instantiate and load a Cylinder3D model from a checkpoint.

    Args:
        config: Model configuration dictionary.
        checkpoint_path: Path to the model checkpoint file.
        device: Device to load the model onto.

    Returns:
        Loaded model in eval mode.
    """
    model_config = config.get('model', {})

    model = Cylinder3D(
        num_classes=model_config.get('num_classes', 20),
        grid_size=model_config.get('grid_size', [480, 360, 32]),
        base_channels=model_config.get('base_channels', 32),
        use_point_refinement=model_config.get('use_point_refinement', True),
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if model was saved with DataParallel
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            cleaned_state_dict[key[7:]] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model


# ==============================================================================
# Evaluation Loop
# ==============================================================================

def evaluate(
    model: Cylinder3D,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    save_predictions: bool = False,
    output_dir: Optional[str] = None,
    scan_files: Optional[List[str]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Run evaluation over the entire dataset.

    Args:
        model: Cylinder3D model in eval mode.
        dataloader: DataLoader for the evaluation dataset.
        num_classes: Number of semantic classes.
        device: Computation device.
        save_predictions: Whether to save prediction .label files.
        output_dir: Directory to save predictions (required if save_predictions=True).
        scan_files: List of scan file paths for deriving output filenames.

    Returns:
        per_class_iou: (num_classes,) array of per-class IoU.
        miou: Mean IoU (excluding class 0).
    """
    confusion = ConfusionMatrix(num_classes)
    inverse_map = build_inverse_learning_map()

    total_points = 0
    total_correct = 0
    num_batches = len(dataloader)
    global_scan_idx = 0

    print(f"Evaluating on {num_batches} batches...")
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            points = batch['points'].to(device)
            labels = batch['labels'].numpy()
            point_counts = batch['point_counts']

            # Forward pass
            output = model(points, num_points=point_counts.to(device))

            # Get per-point predictions
            if 'point_logits' in output:
                logits = output['point_logits']
            else:
                # Fall back to voxel logits with point-to-voxel mapping
                # Use voxel_logits and point_to_voxel for point-level predictions
                voxel_logits = output['voxel_logits']  # (B, C, D, H, W)
                point_to_voxel = output['point_to_voxel']
                # Flatten voxel logits and gather per-point predictions
                B, C = voxel_logits.shape[0], voxel_logits.shape[1]
                voxel_logits_flat = voxel_logits.reshape(B, C, -1)
                # For batch_size=1 shortcut
                logits = voxel_logits_flat[0, :, point_to_voxel].t()

            predictions = logits.argmax(dim=1).cpu().numpy()

            # Process each sample in the batch
            point_offset = 0
            for sample_idx in range(point_counts.shape[0]):
                n_pts = point_counts[sample_idx].item()
                sample_preds = predictions[point_offset:point_offset + n_pts]
                sample_labels = labels[point_offset:point_offset + n_pts]

                # Update confusion matrix
                confusion.update(sample_preds, sample_labels)

                # Accumulate accuracy stats
                valid_mask = sample_labels != 0  # Exclude unlabeled for accuracy
                if valid_mask.sum() > 0:
                    total_points += valid_mask.sum()
                    total_correct += (
                        (sample_preds[valid_mask] == sample_labels[valid_mask]).sum()
                    )

                # Save predictions if requested
                if save_predictions and output_dir is not None:
                    if scan_files is not None and global_scan_idx < len(scan_files):
                        # Derive output path from scan file path
                        scan_path = scan_files[global_scan_idx]
                        # Extract sequence and frame info
                        # e.g. .../sequences/08/velodyne/000000.bin
                        parts = scan_path.replace('\\', '/').split('/')
                        try:
                            seq_idx = parts.index('sequences')
                            seq_id = parts[seq_idx + 1]
                            frame_name = os.path.splitext(parts[-1])[0]
                        except (ValueError, IndexError):
                            seq_id = "unknown"
                            frame_name = f"{global_scan_idx:06d}"

                        label_output_path = os.path.join(
                            output_dir, 'sequences', seq_id, 'predictions',
                            frame_name + '.label'
                        )
                    else:
                        label_output_path = os.path.join(
                            output_dir, f"{global_scan_idx:06d}.label"
                        )

                    save_predictions_semantickitti(
                        sample_preds, label_output_path, inverse_map
                    )

                global_scan_idx += 1
                point_offset += n_pts

            # Progress reporting
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - start_time
                speed = (batch_idx + 1) / elapsed
                eta = (num_batches - batch_idx - 1) / speed if speed > 0 else 0
                print(
                    f"  [{batch_idx + 1}/{num_batches}] "
                    f"Speed: {speed:.1f} batches/s | "
                    f"ETA: {eta:.0f}s"
                )

    elapsed_total = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed_total:.1f}s")

    # Compute IoU
    per_class_iou, miou = confusion.compute_iou(ignore_class=0)

    # Print accuracy
    if total_points > 0:
        overall_accuracy = total_correct / total_points
        print(f"Overall point accuracy (excl. unlabeled): {overall_accuracy * 100:.2f}%")

    return per_class_iou, miou


# ==============================================================================
# Main Entry Point
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Cylinder3D on SemanticKITTI val/test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file. If not provided, uses defaults.',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file (.pth or .pt).',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['val', 'test'],
        help='Dataset split to evaluate on.',
    )
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='Save predictions as .label files in SemanticKITTI format.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./predictions',
        help='Directory to save prediction .label files.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (e.g., cuda:0, cpu). Default: auto-detect.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size for evaluation. Overrides config value.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of dataloader workers. Overrides config value.',
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Path to SemanticKITTI dataset root. Overrides config value.',
    )

    return parser.parse_args()


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Load configuration
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = get_default_config()

    # Override config with command-line arguments
    if args.data_root is not None:
        config.setdefault('dataset', {})['root'] = args.data_root
    if args.batch_size is not None:
        config.setdefault('evaluation', {})['batch_size'] = args.batch_size
    if args.num_workers is not None:
        config.setdefault('evaluation', {})['num_workers'] = args.num_workers

    # Determine device
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(config, args.checkpoint, device)
    num_classes = config.get('model', {}).get('num_classes', 20)
    print(f"Model loaded successfully. num_classes={num_classes}")

    # Prepare dataset
    dataset_config = config.get('dataset', {})
    data_root = dataset_config.get('root', './data/semantickitti')
    splits = build_semantickitti_splits()

    if args.split not in splits:
        raise ValueError(f"Unknown split '{args.split}'. Available: {list(splits.keys())}")

    sequences = splits[args.split]
    print(f"Evaluating on {args.split} split: sequences {sequences}")

    # Create dataset (no augmentation for evaluation)
    dataset = SemanticKITTIDataset(
        root=data_root,
        sequences=sequences,
        config={
            'learning_map': SEMANTICKITTI_LEARNING_MAP,
            'max_points': dataset_config.get('max_points', None),
        },
        augment=False,
    )

    print(f"Dataset size: {len(dataset)} scans")

    # Create dataloader
    eval_config = config.get('evaluation', {})
    batch_size = eval_config.get('batch_size', 1)
    num_workers = eval_config.get('num_workers', 4)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )

    # Run evaluation
    scan_files = dataset.scan_files if args.save_predictions else None

    per_class_iou, miou = evaluate(
        model=model,
        dataloader=dataloader,
        num_classes=num_classes,
        device=device,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
        scan_files=scan_files,
    )

    # Print results
    print_results_table(
        per_class_iou=per_class_iou,
        class_names=SEMANTICKITTI_CLASSES,
        miou=miou,
        ignore_class=0,
    )

    # Save predictions info
    if args.save_predictions:
        print(f"Predictions saved to: {os.path.abspath(args.output_dir)}")

    return miou


if __name__ == '__main__':
    main()
