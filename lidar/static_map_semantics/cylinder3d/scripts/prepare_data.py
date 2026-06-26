#!/usr/bin/env python3
"""
prepare_data.py - Prepare SemanticKITTI dataset for Cylinder3D training.

This script:
  - Parses SemanticKITTI sequences into train/val/test splits
  - Creates file lists (train.txt, val.txt, test.txt)
  - Computes class frequency statistics from training labels
  - Computes class weights (inverse frequency / log-smoothed)
  - Computes dataset-wide point statistics (mean, std of xyz + intensity)
  - Verifies data integrity (matching .bin and .label files)
  - Saves all statistics to a JSON file

Usage:
  python prepare_data.py --dataset_root ./dataset --output_dir ./data_info
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# =============================================================================
# SemanticKITTI Configuration
# =============================================================================

# Standard SemanticKITTI train/val/test split
TRAIN_SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
VAL_SEQUENCES = ["08"]
TEST_SEQUENCES = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]

# SemanticKITTI label mapping: raw label -> training label (19 classes + ignore)
# Original labels (0-259) are mapped to 20 classes (0=unlabeled, 1-19=valid)
LEARNING_MAP = {
    0: 0,       # "unlabeled"
    1: 0,       # "outlier" -> unlabeled
    10: 1,      # "car"
    11: 2,      # "bicycle"
    13: 5,      # "bus"
    15: 3,      # "motorcycle"
    16: 5,      # "on-rails" -> bus
    18: 4,      # "truck"
    20: 5,      # "other-vehicle"
    30: 6,      # "person"
    31: 7,      # "bicyclist"
    32: 8,      # "motorcyclist"
    40: 9,      # "road"
    44: 10,     # "parking"
    48: 11,     # "sidewalk"
    49: 12,     # "other-ground"
    50: 13,     # "building"
    51: 14,     # "fence"
    52: 0,      # "other-structure" -> unlabeled
    60: 9,      # "lane-marking" -> road
    70: 15,     # "vegetation"
    71: 16,     # "trunk"
    72: 17,     # "terrain"
    80: 18,     # "pole"
    81: 19,     # "traffic-sign"
    99: 0,      # "other-object" -> unlabeled
    252: 1,     # "moving-car" -> car
    253: 7,     # "moving-bicyclist" -> bicyclist
    254: 6,     # "moving-person" -> person
    255: 8,     # "moving-motorcyclist" -> motorcyclist
    256: 5,     # "moving-on-rails" -> bus
    257: 5,     # "moving-bus" -> bus
    258: 4,     # "moving-truck" -> truck
    259: 5,     # "moving-other-vehicle" -> other-vehicle
}

# Class names for the 20 mapped classes (0=ignore, 1-19=valid)
CLASS_NAMES = {
    0: "unlabeled",
    1: "car",
    2: "bicycle",
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "bicyclist",
    8: "motorcyclist",
    9: "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign",
}

NUM_CLASSES = 20  # 0 (ignore) + 19 valid classes


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_velodyne_scan(bin_path: str) -> np.ndarray:
    """Load a velodyne point cloud from a .bin file.

    Args:
        bin_path: Path to the .bin file.

    Returns:
        Point cloud as (N, 4) array [x, y, z, intensity].
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def load_label(label_path: str) -> np.ndarray:
    """Load semantic labels from a .label file.

    Args:
        label_path: Path to the .label file.

    Returns:
        Labels as (N,) array with semantic class IDs.
    """
    # Labels are stored as uint32: lower 16 bits = semantic, upper 16 bits = instance
    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    semantic_labels = raw_labels & 0xFFFF  # Extract semantic label
    return semantic_labels


def map_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Map raw SemanticKITTI labels to training labels using LEARNING_MAP.

    Args:
        raw_labels: Raw semantic labels.

    Returns:
        Mapped labels in range [0, 19].
    """
    mapped = np.zeros_like(raw_labels)
    for raw_id, mapped_id in LEARNING_MAP.items():
        mapped[raw_labels == raw_id] = mapped_id
    return mapped


# =============================================================================
# File List Generation
# =============================================================================

def get_scan_paths(dataset_root: str, sequence: str) -> List[str]:
    """Get sorted list of scan file paths for a sequence.

    Args:
        dataset_root: Root directory of the dataset.
        sequence: Sequence identifier (e.g., "00").

    Returns:
        Sorted list of .bin file paths (relative to dataset_root).
    """
    vel_dir = Path(dataset_root) / "sequences" / sequence / "velodyne"
    if not vel_dir.exists():
        return []

    scans = sorted(vel_dir.glob("*.bin"))
    # Return paths relative to dataset_root
    return [str(scan.relative_to(dataset_root)) for scan in scans]


def get_label_paths(dataset_root: str, sequence: str) -> List[str]:
    """Get sorted list of label file paths for a sequence.

    Args:
        dataset_root: Root directory of the dataset.
        sequence: Sequence identifier (e.g., "00").

    Returns:
        Sorted list of .label file paths (relative to dataset_root).
    """
    lab_dir = Path(dataset_root) / "sequences" / sequence / "labels"
    if not lab_dir.exists():
        return []

    labels = sorted(lab_dir.glob("*.label"))
    return [str(label.relative_to(dataset_root)) for label in labels]


def generate_file_lists(
    dataset_root: str, output_dir: str
) -> Tuple[List[str], List[str], List[str]]:
    """Generate train.txt, val.txt, test.txt file lists.

    Args:
        dataset_root: Root directory of the dataset.
        output_dir: Directory to save the file lists.

    Returns:
        Tuple of (train_files, val_files, test_files).
    """
    os.makedirs(output_dir, exist_ok=True)

    train_files = []
    val_files = []
    test_files = []

    # Training sequences
    for seq in TRAIN_SEQUENCES:
        scans = get_scan_paths(dataset_root, seq)
        train_files.extend(scans)
        print(f"  Train seq {seq}: {len(scans)} scans")

    # Validation sequences
    for seq in VAL_SEQUENCES:
        scans = get_scan_paths(dataset_root, seq)
        val_files.extend(scans)
        print(f"  Val   seq {seq}: {len(scans)} scans")

    # Test sequences
    for seq in TEST_SEQUENCES:
        scans = get_scan_paths(dataset_root, seq)
        test_files.extend(scans)
        print(f"  Test  seq {seq}: {len(scans)} scans")

    # Write file lists
    for name, files in [
        ("train.txt", train_files),
        ("val.txt", val_files),
        ("test.txt", test_files),
    ]:
        filepath = os.path.join(output_dir, name)
        with open(filepath, "w") as f:
            f.write("\n".join(files))
            if files:
                f.write("\n")
        print(f"  Written {filepath} ({len(files)} entries)")

    return train_files, val_files, test_files


# =============================================================================
# Statistics Computation
# =============================================================================

def compute_class_frequencies(
    dataset_root: str, train_files: List[str], num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Compute class frequencies from training set labels.

    Args:
        dataset_root: Root directory of the dataset.
        train_files: List of training scan paths (relative to dataset_root).
        num_classes: Number of classes.

    Returns:
        Array of shape (num_classes,) with point counts per class.
    """
    print("\nComputing class frequencies from training labels...")
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for scan_path in tqdm(train_files, desc="Counting labels"):
        # Derive label path from scan path
        label_path = scan_path.replace("velodyne", "labels").replace(".bin", ".label")
        label_full_path = os.path.join(dataset_root, label_path)

        if not os.path.exists(label_full_path):
            continue

        raw_labels = load_label(label_full_path)
        mapped_labels = map_labels(raw_labels)

        for cls in range(num_classes):
            class_counts[cls] += np.sum(mapped_labels == cls)

    return class_counts


def compute_class_weights(
    class_counts: np.ndarray, method: str = "log_smoothed"
) -> np.ndarray:
    """Compute class weights from class frequencies.

    Args:
        class_counts: Point counts per class.
        method: Weighting method - 'inverse', 'sqrt_inverse', or 'log_smoothed'.

    Returns:
        Array of class weights (num_classes,). Class 0 (unlabeled) gets weight 0.
    """
    # Avoid division by zero
    counts = class_counts.copy().astype(np.float64)
    counts[counts == 0] = 1

    if method == "inverse":
        # Simple inverse frequency
        weights = 1.0 / counts
    elif method == "sqrt_inverse":
        # Square root inverse frequency
        weights = 1.0 / np.sqrt(counts)
    elif method == "log_smoothed":
        # Log-smoothed inverse frequency (common in segmentation)
        total = counts.sum()
        frequency = counts / total
        weights = 1.0 / np.log(1.02 + frequency)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Normalize so that valid class weights sum to num_valid_classes
    valid_mask = np.arange(len(weights)) > 0  # Ignore class 0
    if valid_mask.sum() > 0:
        weights[valid_mask] = weights[valid_mask] / weights[valid_mask].mean()

    # Set weight of unlabeled class to 0
    weights[0] = 0.0

    return weights


def compute_point_statistics(
    dataset_root: str,
    train_files: List[str],
    max_samples: int = 5000,
) -> Dict[str, List[float]]:
    """Compute dataset-wide point statistics (mean, std) for normalization.

    Computes mean and std for each channel: x, y, z, intensity.

    Args:
        dataset_root: Root directory of the dataset.
        train_files: List of training scan paths.
        max_samples: Maximum number of scans to sample for statistics.

    Returns:
        Dict with 'mean' and 'std' keys, each containing [x, y, z, intensity].
    """
    print("\nComputing point cloud statistics...")

    # Sample a subset if dataset is large
    if len(train_files) > max_samples:
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(len(train_files), max_samples, replace=False)
        sample_files = [train_files[i] for i in sample_indices]
    else:
        sample_files = train_files

    # Use Welford's online algorithm for numerically stable mean/std
    n = 0
    mean = np.zeros(4, dtype=np.float64)
    M2 = np.zeros(4, dtype=np.float64)

    for scan_path in tqdm(sample_files, desc="Computing statistics"):
        full_path = os.path.join(dataset_root, scan_path)
        if not os.path.exists(full_path):
            continue

        points = load_velodyne_scan(full_path)

        for point in points:
            n += 1
            delta = point - mean
            mean += delta / n
            delta2 = point - mean
            M2 += delta * delta2

    if n < 2:
        print("  WARNING: Not enough data points for statistics computation")
        return {"mean": [0.0, 0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0, 1.0]}

    variance = M2 / (n - 1)
    std = np.sqrt(variance)

    # Alternative: batch computation for speed (large memory usage)
    # This is more practical for actual use
    print(f"  Computed from {n:,} points across {len(sample_files)} scans")
    print(f"  Mean (x,y,z,i): {mean.tolist()}")
    print(f"  Std  (x,y,z,i): {std.tolist()}")

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "num_points_sampled": int(n),
        "num_scans_sampled": len(sample_files),
    }


def compute_point_statistics_batch(
    dataset_root: str,
    train_files: List[str],
    max_samples: int = 500,
) -> Dict[str, List[float]]:
    """Compute point statistics using batch method (faster, more memory).

    Args:
        dataset_root: Root directory of the dataset.
        train_files: List of training scan paths.
        max_samples: Maximum scans to use.

    Returns:
        Dict with 'mean' and 'std' for [x, y, z, intensity].
    """
    print("\nComputing point cloud statistics (batch method)...")

    if len(train_files) > max_samples:
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(len(train_files), max_samples, replace=False)
        sample_files = [train_files[i] for i in sample_indices]
    else:
        sample_files = train_files

    all_points = []
    for scan_path in tqdm(sample_files, desc="Loading scans"):
        full_path = os.path.join(dataset_root, scan_path)
        if not os.path.exists(full_path):
            continue
        points = load_velodyne_scan(full_path)
        all_points.append(points)

    if not all_points:
        return {"mean": [0.0, 0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0, 1.0]}

    all_points = np.concatenate(all_points, axis=0)
    mean = all_points.mean(axis=0).tolist()
    std = all_points.std(axis=0).tolist()

    print(f"  Computed from {len(all_points):,} points across {len(sample_files)} scans")
    print(f"  Mean (x,y,z,i): {mean}")
    print(f"  Std  (x,y,z,i): {std}")

    return {
        "mean": mean,
        "std": std,
        "num_points_sampled": int(len(all_points)),
        "num_scans_sampled": len(sample_files),
    }


# =============================================================================
# Data Integrity Verification
# =============================================================================

def verify_data_integrity(
    dataset_root: str, sequences: List[str], require_labels: bool = True
) -> Tuple[bool, List[str]]:
    """Verify that all .bin files have corresponding .label files.

    Args:
        dataset_root: Root directory of the dataset.
        sequences: List of sequence identifiers to check.
        require_labels: Whether to require labels (False for test sequences).

    Returns:
        Tuple of (all_valid, list_of_issues).
    """
    issues = []

    for seq in sequences:
        vel_dir = Path(dataset_root) / "sequences" / seq / "velodyne"
        lab_dir = Path(dataset_root) / "sequences" / seq / "labels"

        if not vel_dir.exists():
            issues.append(f"Sequence {seq}: velodyne directory missing")
            continue

        scans = sorted(vel_dir.glob("*.bin"))

        if len(scans) == 0:
            issues.append(f"Sequence {seq}: no .bin files found")
            continue

        if require_labels:
            if not lab_dir.exists():
                issues.append(f"Sequence {seq}: labels directory missing")
                continue

            for scan in scans:
                label_file = lab_dir / scan.name.replace(".bin", ".label")
                if not label_file.exists():
                    issues.append(
                        f"Sequence {seq}: missing label for {scan.name}"
                    )

            # Check for orphan labels
            labels = sorted(lab_dir.glob("*.label"))
            for label in labels:
                scan_file = vel_dir / label.name.replace(".label", ".bin")
                if not scan_file.exists():
                    issues.append(
                        f"Sequence {seq}: orphan label {label.name} (no scan)"
                    )

    all_valid = len(issues) == 0
    return all_valid, issues


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare SemanticKITTI dataset for Cylinder3D training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python prepare_data.py --dataset_root ./dataset

  # Custom output directory and weight method
  python prepare_data.py --dataset_root ./dataset --output_dir ./info --weight_method sqrt_inverse

  # Skip statistics computation (faster)
  python prepare_data.py --dataset_root ./dataset --skip_stats

  # Only verify data integrity
  python prepare_data.py --dataset_root ./dataset --verify_only
        """,
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory of SemanticKITTI dataset (containing sequences/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for file lists and stats (default: dataset_root/data_info)",
    )
    parser.add_argument(
        "--weight_method",
        type=str,
        default="log_smoothed",
        choices=["inverse", "sqrt_inverse", "log_smoothed"],
        help="Class weight computation method (default: log_smoothed)",
    )
    parser.add_argument(
        "--max_stat_samples",
        type=int,
        default=500,
        help="Maximum scans to sample for point statistics (default: 500)",
    )
    parser.add_argument(
        "--skip_stats",
        action="store_true",
        help="Skip computing point statistics (faster)",
    )
    parser.add_argument(
        "--skip_weights",
        action="store_true",
        help="Skip computing class weights (faster)",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify data integrity, don't generate file lists",
    )
    parser.add_argument(
        "--batch_stats",
        action="store_true",
        default=True,
        help="Use batch method for point statistics (faster, more memory)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    output_dir = args.output_dir or os.path.join(dataset_root, "data_info")

    print("=" * 60)
    print("SemanticKITTI Data Preparation for Cylinder3D")
    print("=" * 60)
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir:   {output_dir}")
    print()

    # Check dataset root exists
    sequences_dir = os.path.join(dataset_root, "sequences")
    if not os.path.isdir(sequences_dir):
        print(f"ERROR: sequences/ directory not found at {sequences_dir}")
        print("Please run download_data.sh first or check the dataset path.")
        sys.exit(1)

    # =========================================================================
    # Step 1: Verify data integrity
    # =========================================================================
    print("-" * 60)
    print("Step 1: Verifying data integrity")
    print("-" * 60)

    # Check train/val sequences (require labels)
    train_val_seqs = TRAIN_SEQUENCES + VAL_SEQUENCES
    valid, issues = verify_data_integrity(
        dataset_root, train_val_seqs, require_labels=True
    )

    if issues:
        print(f"\n  Found {len(issues)} issue(s):")
        for issue in issues[:20]:  # Show first 20
            print(f"    - {issue}")
        if len(issues) > 20:
            print(f"    ... and {len(issues) - 20} more")

        if not args.verify_only:
            print("\n  WARNING: Proceeding despite integrity issues.")
    else:
        print("  All train/val sequences verified OK!")

    # Check test sequences (no labels required)
    valid_test, test_issues = verify_data_integrity(
        dataset_root, TEST_SEQUENCES, require_labels=False
    )
    if test_issues:
        print(f"\n  Test sequence issues: {len(test_issues)}")
        for issue in test_issues[:10]:
            print(f"    - {issue}")

    if args.verify_only:
        print("\n  Verify-only mode. Exiting.")
        sys.exit(0 if valid else 1)

    # =========================================================================
    # Step 2: Generate file lists
    # =========================================================================
    print("\n" + "-" * 60)
    print("Step 2: Generating file lists")
    print("-" * 60)

    train_files, val_files, test_files = generate_file_lists(
        dataset_root, output_dir
    )

    # =========================================================================
    # Step 3: Compute class statistics
    # =========================================================================
    statistics = {
        "dataset": "SemanticKITTI",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "learning_map": {str(k): v for k, v in LEARNING_MAP.items()},
        "splits": {
            "train_sequences": TRAIN_SEQUENCES,
            "val_sequences": VAL_SEQUENCES,
            "test_sequences": TEST_SEQUENCES,
            "num_train_scans": len(train_files),
            "num_val_scans": len(val_files),
            "num_test_scans": len(test_files),
        },
    }

    if not args.skip_weights and train_files:
        print("\n" + "-" * 60)
        print("Step 3: Computing class frequencies and weights")
        print("-" * 60)

        class_counts = compute_class_frequencies(dataset_root, train_files)
        class_weights = compute_class_weights(class_counts, method=args.weight_method)

        # Print class statistics
        print("\n  Class statistics:")
        print(f"  {'ID':<4} {'Name':<16} {'Count':<14} {'Freq %':<10} {'Weight':<8}")
        print("  " + "-" * 56)
        total_points = class_counts.sum()
        for i in range(NUM_CLASSES):
            freq = class_counts[i] / total_points * 100 if total_points > 0 else 0
            print(
                f"  {i:<4} {CLASS_NAMES[i]:<16} {class_counts[i]:<14,} "
                f"{freq:<10.3f} {class_weights[i]:<8.4f}"
            )

        statistics["class_counts"] = class_counts.tolist()
        statistics["class_weights"] = class_weights.tolist()
        statistics["weight_method"] = args.weight_method

    # =========================================================================
    # Step 4: Compute point statistics
    # =========================================================================
    if not args.skip_stats and train_files:
        print("\n" + "-" * 60)
        print("Step 4: Computing point cloud statistics")
        print("-" * 60)

        if args.batch_stats:
            point_stats = compute_point_statistics_batch(
                dataset_root, train_files, max_samples=args.max_stat_samples
            )
        else:
            point_stats = compute_point_statistics(
                dataset_root, train_files, max_samples=args.max_stat_samples
            )

        statistics["point_statistics"] = point_stats

    # =========================================================================
    # Step 5: Save statistics
    # =========================================================================
    print("\n" + "-" * 60)
    print("Step 5: Saving statistics")
    print("-" * 60)

    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "dataset_statistics.json")

    with open(stats_path, "w") as f:
        json.dump(statistics, f, indent=2)

    print(f"  Statistics saved to: {stats_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Preparation complete!")
    print("=" * 60)
    print(f"  File lists: {output_dir}/{{train,val,test}}.txt")
    print(f"  Statistics: {stats_path}")
    print(f"  Train scans: {len(train_files)}")
    print(f"  Val scans:   {len(val_files)}")
    print(f"  Test scans:  {len(test_files)}")
    print()


if __name__ == "__main__":
    main()
