"""
Inference script for Cylinder3D semantic segmentation.

Runs trained Cylinder3D model on individual .bin point cloud files or a directory
of scans, producing per-point semantic labels and optional colored PLY output.

Usage:
    # Single file inference:
    python -m lidar.static_map_semantics.cylinder3d.pytorch.inference \
        --checkpoint checkpoints/cylinder3d_best.pth \
        --input data/scan.bin \
        --output_dir results/ \
        --save_labels --save_ply

    # Directory of .bin files:
    python -m lidar.static_map_semantics.cylinder3d.pytorch.inference \
        --checkpoint checkpoints/cylinder3d_best.pth \
        --input data/velodyne/ \
        --output_dir results/ \
        --save_labels --save_ply
"""

import os
import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import yaml
except ImportError:
    yaml = None

from .model import Cylinder3D
from .dataset import SEMANTICKITTI_CLASSES


# ==============================================================================
# SemanticKITTI 19-class Color Map (RGB, uint8)
# ==============================================================================

SEMANTICKITTI_COLOR_MAP_RGB = {
    0:  (0, 0, 0),           # unlabeled - black
    1:  (245, 150, 100),     # car - orange
    2:  (245, 230, 100),     # bicycle - yellow
    3:  (150, 60, 30),       # motorcycle - dark orange
    4:  (180, 30, 80),       # truck - dark magenta
    5:  (255, 0, 0),         # other-vehicle - red
    6:  (30, 30, 255),       # person - blue
    7:  (200, 40, 255),      # bicyclist - purple
    8:  (90, 30, 150),       # motorcyclist - dark purple
    9:  (255, 0, 255),       # road - magenta
    10: (255, 150, 255),     # parking - light magenta
    11: (75, 0, 75),         # sidewalk - dark purple
    12: (75, 0, 175),        # other-ground - indigo
    13: (0, 200, 255),       # building - cyan
    14: (50, 120, 255),      # fence - light blue
    15: (0, 175, 0),         # vegetation - green
    16: (0, 60, 135),        # trunk - dark brown
    17: (80, 240, 150),      # terrain - light green
    18: (150, 240, 255),     # pole - light cyan
    19: (0, 0, 255),         # traffic-sign - blue
}


def get_color_for_label(label: int) -> Tuple[int, int, int]:
    """
    Get RGB color tuple for a semantic label.

    Args:
        label: Integer class label [0..19].

    Returns:
        (R, G, B) tuple with values in [0, 255].
    """
    return SEMANTICKITTI_COLOR_MAP_RGB.get(label, (0, 0, 0))


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    """
    Convert integer label array to RGB color array.

    Args:
        labels: (N,) int array of class labels.

    Returns:
        (N, 3) uint8 array of RGB colors.
    """
    num_classes = max(SEMANTICKITTI_COLOR_MAP_RGB.keys()) + 1
    color_lut = np.zeros((num_classes, 3), dtype=np.uint8)
    for class_id, (r, g, b) in SEMANTICKITTI_COLOR_MAP_RGB.items():
        color_lut[class_id] = [r, g, b]

    # Clip labels to valid range
    labels_clipped = np.clip(labels, 0, num_classes - 1)
    return color_lut[labels_clipped]


# ==============================================================================
# PLY Writing
# ==============================================================================

def write_ply(filename: str, points: np.ndarray, colors: np.ndarray):
    """
    Write a colored point cloud to an ASCII PLY file.

    Args:
        filename: Output file path (.ply).
        points: (N, 3) float array of x, y, z coordinates.
        colors: (N, 3) uint8 array of R, G, B colors.
    """
    num_points = points.shape[0]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

    with open(filename, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write points with colors
        for i in range(num_points):
            f.write(
                f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n"
            )


# ==============================================================================
# Configuration and Model Loading
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
    Return default model configuration for inference.

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
    }


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
# Point Cloud Loading
# ==============================================================================

def load_point_cloud(bin_path: str) -> np.ndarray:
    """
    Load a point cloud from a binary .bin file.

    Expects N x 4 float32 values: x, y, z, intensity (remission).

    Args:
        bin_path: Path to the .bin file.

    Returns:
        (N, 4) float32 numpy array.
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


# ==============================================================================
# Single Scan Inference
# ==============================================================================

def infer_single_scan(
    model: Cylinder3D,
    points: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Run inference on a single point cloud scan.

    Args:
        model: Cylinder3D model in eval mode.
        points: (N, 4) float32 array of x, y, z, intensity.
        device: Computation device.

    Returns:
        predictions: (N,) int array of predicted class labels.
        timings: Dict with 'preprocess_ms', 'inference_ms', 'postprocess_ms'.
    """
    timings = {}

    # Preprocessing
    t0 = time.perf_counter()
    points_tensor = torch.from_numpy(points).float().to(device)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.perf_counter()
    timings['preprocess_ms'] = (t1 - t0) * 1000.0

    # Inference
    t2 = time.perf_counter()
    with torch.no_grad():
        output = model(points_tensor, num_points=None)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    timings['inference_ms'] = (t3 - t2) * 1000.0

    # Postprocessing
    t4 = time.perf_counter()
    if 'point_logits' in output:
        logits = output['point_logits']
    else:
        # Fall back to voxel-level predictions mapped to points
        voxel_logits = output['voxel_logits']  # (1, C, D, H, W)
        point_to_voxel = output['point_to_voxel']
        C = voxel_logits.shape[1]
        voxel_logits_flat = voxel_logits.reshape(1, C, -1)
        logits = voxel_logits_flat[0, :, point_to_voxel].t()

    predictions = logits.argmax(dim=1).cpu().numpy()
    t5 = time.perf_counter()
    timings['postprocess_ms'] = (t5 - t4) * 1000.0

    return predictions, timings


# ==============================================================================
# Results Summary
# ==============================================================================

def print_prediction_summary(
    predictions: np.ndarray,
    class_names: List[str],
    scan_name: str,
):
    """
    Print per-class point counts for a single scan prediction.

    Args:
        predictions: (N,) int array of predicted labels.
        class_names: List of class name strings.
        scan_name: Name of the scan file (for display).
    """
    num_classes = len(class_names)
    counts = np.bincount(predictions, minlength=num_classes)

    print(f"\n  Prediction summary for: {scan_name}")
    print(f"  Total points: {predictions.shape[0]}")
    print(f"  {'Class':<20} {'Count':<10} {'Percentage'}")
    print("  " + "-" * 50)

    for c in range(num_classes):
        if counts[c] > 0:
            pct = counts[c] / predictions.shape[0] * 100
            print(f"  {class_names[c]:<20} {counts[c]:<10} {pct:.1f}%")


def print_timing(timings: Dict[str, float], scan_name: str):
    """
    Print timing breakdown for a single scan inference.

    Args:
        timings: Dict with timing keys in milliseconds.
        scan_name: Name of the scan file.
    """
    total_ms = sum(timings.values())
    print(f"\n  Timing for: {scan_name}")
    print(f"    Preprocessing:  {timings['preprocess_ms']:>8.1f} ms")
    print(f"    Inference:      {timings['inference_ms']:>8.1f} ms")
    print(f"    Postprocessing: {timings['postprocess_ms']:>8.1f} ms")
    print(f"    Total:          {total_ms:>8.1f} ms")
    print(f"    Throughput:     {1000.0 / total_ms:.1f} scans/s" if total_ms > 0 else "")


# ==============================================================================
# Batch Processing
# ==============================================================================

def process_scan(
    model: Cylinder3D,
    bin_path: str,
    device: torch.device,
    output_dir: str,
    save_labels: bool = True,
    save_ply: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Process a single .bin scan file through the model.

    Args:
        model: Cylinder3D model in eval mode.
        bin_path: Path to input .bin file.
        device: Computation device.
        output_dir: Directory for output files.
        save_labels: Whether to save .label prediction file.
        save_ply: Whether to save colored .ply visualization.
        verbose: Whether to print per-scan info.

    Returns:
        predictions: (N,) int array of predicted labels.
        timings: Dict with timing breakdown.
    """
    scan_name = os.path.splitext(os.path.basename(bin_path))[0]

    # Load point cloud
    points = load_point_cloud(bin_path)

    # Run inference
    predictions, timings = infer_single_scan(model, points, device)

    # Save .label file
    if save_labels:
        label_path = os.path.join(output_dir, scan_name + '.label')
        os.makedirs(output_dir, exist_ok=True)
        predictions.astype(np.uint32).tofile(label_path)
        if verbose:
            print(f"  Saved labels: {label_path}")

    # Save colored .ply file
    if save_ply:
        colors = labels_to_colors(predictions)
        ply_path = os.path.join(output_dir, scan_name + '.ply')
        os.makedirs(output_dir, exist_ok=True)
        write_ply(ply_path, points[:, :3], colors)
        if verbose:
            print(f"  Saved PLY:    {ply_path}")

    # Print summary
    if verbose:
        print_timing(timings, scan_name)
        print_prediction_summary(predictions, SEMANTICKITTI_CLASSES, scan_name)

    return predictions, timings


def process_directory(
    model: Cylinder3D,
    input_dir: str,
    device: torch.device,
    output_dir: str,
    save_labels: bool = True,
    save_ply: bool = False,
):
    """
    Process all .bin files in a directory.

    Args:
        model: Cylinder3D model in eval mode.
        input_dir: Directory containing .bin files.
        device: Computation device.
        output_dir: Directory for output files.
        save_labels: Whether to save .label prediction files.
        save_ply: Whether to save colored .ply visualizations.
    """
    # Find all .bin files
    bin_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.bin')
    ])

    if len(bin_files) == 0:
        print(f"No .bin files found in: {input_dir}")
        return

    print(f"Found {len(bin_files)} .bin files in: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    total_timings = {
        'preprocess_ms': 0.0,
        'inference_ms': 0.0,
        'postprocess_ms': 0.0,
    }
    total_points = 0

    for idx, bin_path in enumerate(bin_files):
        print(f"\n[{idx + 1}/{len(bin_files)}] Processing: {os.path.basename(bin_path)}")

        predictions, timings = process_scan(
            model=model,
            bin_path=bin_path,
            device=device,
            output_dir=output_dir,
            save_labels=save_labels,
            save_ply=save_ply,
            verbose=True,
        )

        total_points += predictions.shape[0]
        for key in total_timings:
            total_timings[key] += timings[key]

    # Print overall statistics
    total_time_ms = sum(total_timings.values())
    print("\n" + "=" * 60)
    print("  Overall Statistics")
    print("=" * 60)
    print(f"  Total scans processed:  {len(bin_files)}")
    print(f"  Total points processed: {total_points:,}")
    print(f"  Total time:             {total_time_ms / 1000.0:.2f} s")
    print(f"  Average per scan:")
    print(f"    Preprocessing:  {total_timings['preprocess_ms'] / len(bin_files):.1f} ms")
    print(f"    Inference:      {total_timings['inference_ms'] / len(bin_files):.1f} ms")
    print(f"    Postprocessing: {total_timings['postprocess_ms'] / len(bin_files):.1f} ms")
    avg_total = total_time_ms / len(bin_files)
    print(f"    Total:          {avg_total:.1f} ms")
    if avg_total > 0:
        print(f"  Average throughput: {1000.0 / avg_total:.1f} scans/s")
    print("=" * 60)


# ==============================================================================
# Main Entry Point
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Cylinder3D inference on LiDAR point cloud .bin files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file (.pth or .pt).',
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file. If not provided, uses defaults.',
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to a single .bin file or a directory containing .bin files.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./inference_output',
        help='Directory to save output files (.label and/or .ply).',
    )
    parser.add_argument(
        '--save_labels',
        action='store_true',
        help='Save per-point predictions as .label files (uint32).',
    )
    parser.add_argument(
        '--save_ply',
        action='store_true',
        help='Save colored point cloud as .ply files for visualization.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (e.g., cuda:0, cpu). Default: auto-detect.',
    )

    return parser.parse_args()


def main():
    """Main inference entry point."""
    args = parse_args()

    # Load configuration
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = get_default_config()

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

    # Validate input path
    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Ensure at least one output format is requested
    if not args.save_labels and not args.save_ply:
        print("Warning: Neither --save_labels nor --save_ply specified.")
        print("         Running inference anyway (results printed to console).")
        # Default to saving labels
        args.save_labels = True

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process input
    if os.path.isfile(input_path):
        # Single file
        if not input_path.endswith('.bin'):
            raise ValueError(
                f"Input file must be a .bin point cloud file. Got: {input_path}"
            )

        print(f"\nProcessing single scan: {input_path}")
        print("=" * 60)

        process_scan(
            model=model,
            bin_path=input_path,
            device=device,
            output_dir=args.output_dir,
            save_labels=args.save_labels,
            save_ply=args.save_ply,
            verbose=True,
        )

    elif os.path.isdir(input_path):
        # Directory of .bin files
        process_directory(
            model=model,
            input_dir=input_path,
            device=device,
            output_dir=args.output_dir,
            save_labels=args.save_labels,
            save_ply=args.save_ply,
        )
    else:
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")

    print("\nInference complete.")


if __name__ == '__main__':
    main()
