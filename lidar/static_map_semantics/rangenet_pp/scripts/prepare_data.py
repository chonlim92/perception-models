#!/usr/bin/env python3
"""
Precompute range images from SemanticKITTI point clouds for RangeNet++ training.

Reads raw .bin point cloud files and .label annotation files from SemanticKITTI,
applies spherical projection to generate range images, remaps semantic labels to
training IDs, and saves the results as .npy files.

Usage:
    python prepare_data.py \
        --data_dir /path/to/semantickitti/dataset/sequences \
        --output_dir /path/to/output \
        --sequences 00 01 02 03 04 05 06 07 08 09 10 \
        --num_workers 8
"""

import argparse
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ============================================================================
# SemanticKITTI learning map: raw label -> training ID (0-19)
# Labels not in this map are mapped to 0 (unlabeled/ignored).
# ============================================================================
LEARNING_MAP = {
    0: 0,       # "unlabeled"
    1: 0,       # "outlier" mapped to "unlabeled"
    10: 1,      # "car"
    11: 2,      # "bicycle"
    13: 5,      # "bus"
    15: 3,      # "motorcycle"
    16: 5,      # "on-rails" mapped to "bus" (other-vehicle)
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
    52: 0,      # "other-structure" mapped to "unlabeled"
    60: 9,      # "lane-marking" mapped to "road"
    70: 15,     # "vegetation"
    71: 16,     # "trunk"
    72: 17,     # "terrain"
    80: 18,     # "pole"
    81: 19,     # "traffic-sign"
    99: 0,      # "other-object" mapped to "unlabeled"
    252: 1,     # "moving-car" mapped to "car"
    253: 7,     # "moving-bicyclist" mapped to "bicyclist"
    254: 6,     # "moving-person" mapped to "person"
    255: 8,     # "moving-motorcyclist" mapped to "motorcyclist"
    256: 5,     # "moving-on-rails" mapped to "other-vehicle"
    257: 5,     # "moving-bus" mapped to "other-vehicle"
    258: 4,     # "moving-truck" mapped to "truck"
    259: 5,     # "moving-other-vehicle" mapped to "other-vehicle"
}

# Training class names for statistics reporting
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

# Projection parameters
PROJ_H = 64
PROJ_W = 2048
FOV_UP = 2.0          # degrees
FOV_DOWN = -24.8      # degrees
FOV_HORIZONTAL = 360  # degrees


def build_label_lut():
    """Build a lookup table for fast label remapping.

    Returns a numpy array of size 260 (max raw label + 1 for the base range)
    plus extended entries for moving-object labels (252-259).
    We use a dict-based approach to handle sparse labels safely.
    """
    # Max possible raw label value we need to handle
    max_label = max(LEARNING_MAP.keys()) + 1
    lut = np.zeros(max_label, dtype=np.int32)
    for raw_label, train_id in LEARNING_MAP.items():
        if raw_label < max_label:
            lut[raw_label] = train_id
    return lut


# Pre-build the LUT at module level for efficiency
LABEL_LUT = build_label_lut()


def remap_labels(raw_labels):
    """Remap raw SemanticKITTI labels to training IDs using the learning map.

    Args:
        raw_labels: numpy array of uint16 raw semantic labels.

    Returns:
        numpy array of int32 training IDs (0-19).
    """
    # Use vectorized lookup for labels within LUT range
    max_lut = len(LABEL_LUT)
    result = np.zeros_like(raw_labels, dtype=np.int32)

    # Mask for labels within the LUT range
    in_range = raw_labels < max_lut
    result[in_range] = LABEL_LUT[raw_labels[in_range]]

    # For any labels outside LUT range (shouldn't happen with valid data),
    # they remain 0 (unlabeled)
    return result


def spherical_projection(points, remissions):
    """Project 3D point cloud onto a 2D spherical range image.

    Args:
        points: (N, 3) float32 array of x, y, z coordinates.
        remissions: (N,) float32 array of remission/intensity values.

    Returns:
        range_image: (H, W, 5) float32 - channels: range, x, y, z, remission
        proj_indices: (N,) int array - linear index into (H, W) for each point,
                      or -1 if outside FOV.
        pixel_assignments: (H, W) int array - index of the point assigned to
                          each pixel (-1 if no point).
    """
    # Compute range for each point
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    depth = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Filter out points with zero range (at origin)
    valid = depth > 0
    x = x[valid]
    y = y[valid]
    z = z[valid]
    depth = depth[valid]
    remissions_valid = remissions[valid]

    # Compute yaw and pitch angles
    # yaw: -atan2(y, x) -> maps to [0, 2*pi]
    yaw = -np.arctan2(y, x)
    # pitch: asin(z / range)
    pitch = np.arcsin(np.clip(z / depth, -1.0, 1.0))

    # Convert FOV bounds to radians
    fov_up_rad = np.deg2rad(FOV_UP)
    fov_down_rad = np.deg2rad(FOV_DOWN)
    fov_total = fov_up_rad - fov_down_rad  # total vertical FOV

    # Normalize pitch to [0, 1] within the FOV
    # pitch_norm = 1.0 - (pitch - fov_down_rad) / fov_total  # inverted so top=0
    pitch_norm = 1.0 - (pitch - fov_down_rad) / fov_total

    # Normalize yaw to [0, 1]
    yaw_norm = 0.5 * (yaw / np.pi + 1.0)  # yaw in [-pi, pi] -> [0, 1]

    # Compute pixel coordinates
    proj_x = np.floor(yaw_norm * PROJ_W).astype(np.int32)
    proj_y = np.floor(pitch_norm * PROJ_H).astype(np.int32)

    # Clamp to valid range
    proj_x = np.clip(proj_x, 0, PROJ_W - 1)
    proj_y = np.clip(proj_y, 0, PROJ_H - 1)

    # Filter points outside the vertical FOV
    fov_mask = (pitch >= fov_down_rad) & (pitch <= fov_up_rad)
    proj_x = proj_x[fov_mask]
    proj_y = proj_y[fov_mask]
    depth_fov = depth[fov_mask]
    x_fov = x[fov_mask]
    y_fov = y[fov_mask]
    z_fov = z[fov_mask]
    rem_fov = remissions_valid[fov_mask]

    # Initialize output arrays
    range_image = np.full((PROJ_H, PROJ_W, 5), -1.0, dtype=np.float32)
    pixel_assignments = np.full((PROJ_H, PROJ_W), -1, dtype=np.int64)

    # For overlapping points, keep the closest one (smallest range)
    # Sort by depth descending so that closer points overwrite farther ones
    order = np.argsort(-depth_fov)
    proj_x = proj_x[order]
    proj_y = proj_y[order]
    depth_fov = depth_fov[order]
    x_fov = x_fov[order]
    y_fov = y_fov[order]
    z_fov = z_fov[order]
    rem_fov = rem_fov[order]

    # We need original indices (before valid/fov filtering) for label assignment
    # Rebuild the mapping from filtered indices to original point indices
    valid_indices = np.where(valid)[0]
    fov_indices = np.where(fov_mask)[0]
    original_indices = valid_indices[fov_indices]
    original_indices = original_indices[order]

    # Assign points to pixels (last write wins = closest point due to sort order)
    range_image[proj_y, proj_x, 0] = depth_fov
    range_image[proj_y, proj_x, 1] = x_fov
    range_image[proj_y, proj_x, 2] = y_fov
    range_image[proj_y, proj_x, 3] = z_fov
    range_image[proj_y, proj_x, 4] = rem_fov
    pixel_assignments[proj_y, proj_x] = original_indices

    return range_image, pixel_assignments


def process_scan(scan_info, data_dir, output_dir):
    """Process a single scan: load point cloud and labels, project, and save.

    Args:
        scan_info: tuple of (sequence_str, scan_filename)
        data_dir: path to SemanticKITTI sequences directory
        output_dir: path to output directory

    Returns:
        dict with statistics for this scan, or None if processing failed.
    """
    sequence_str, scan_filename = scan_info
    scan_name = os.path.splitext(scan_filename)[0]

    # Construct file paths
    bin_path = os.path.join(data_dir, sequence_str, "velodyne", scan_filename)
    label_path = os.path.join(data_dir, sequence_str, "labels", scan_name + ".label")

    # Check files exist
    if not os.path.isfile(bin_path):
        return None

    # Load point cloud: (N, 4) float32 -> x, y, z, remission
    try:
        scan_data = np.fromfile(bin_path, dtype=np.float32)
    except Exception as e:
        print(f"  [WARN] Failed to read {bin_path}: {e}")
        return None

    # Handle empty scans
    if scan_data.size == 0:
        # Save empty arrays
        seq_output_dir = os.path.join(output_dir, sequence_str)
        os.makedirs(seq_output_dir, exist_ok=True)
        range_image = np.full((PROJ_H, PROJ_W, 5), -1.0, dtype=np.float32)
        label_image = np.zeros((PROJ_H, PROJ_W), dtype=np.int32)
        valid_mask = np.zeros((PROJ_H, PROJ_W), dtype=bool)
        np.save(os.path.join(seq_output_dir, f"{scan_name}_range.npy"), range_image)
        np.save(os.path.join(seq_output_dir, f"{scan_name}_label.npy"), label_image)
        np.save(os.path.join(seq_output_dir, f"{scan_name}_mask.npy"), valid_mask)
        return {
            "num_points": 0,
            "coverage": 0.0,
            "class_counts": np.zeros(20, dtype=np.int64),
        }

    # Reshape to (N, 4)
    scan_data = scan_data.reshape(-1, 4)
    points = scan_data[:, :3]
    remissions = scan_data[:, 3]
    num_points = points.shape[0]

    # Load labels if available
    has_labels = os.path.isfile(label_path)
    if has_labels:
        try:
            raw_labels = np.fromfile(label_path, dtype=np.uint32)
        except Exception as e:
            print(f"  [WARN] Failed to read {label_path}: {e}")
            has_labels = False

    if has_labels:
        # Lower 16 bits = semantic label, upper 16 bits = instance ID
        semantic_labels = (raw_labels & 0xFFFF).astype(np.uint16)
        # Remap to training IDs
        train_labels = remap_labels(semantic_labels)
    else:
        train_labels = np.zeros(num_points, dtype=np.int32)

    # Perform spherical projection
    range_image, pixel_assignments = spherical_projection(points, remissions)

    # Build label image from pixel assignments
    label_image = np.zeros((PROJ_H, PROJ_W), dtype=np.int32)
    valid_mask = pixel_assignments >= 0

    # Assign labels to valid pixels
    valid_pixel_indices = pixel_assignments[valid_mask].astype(np.int64)
    label_image[valid_mask] = train_labels[valid_pixel_indices]

    # Set range image invalid pixels to -1 (already done in initialization)
    # But ensure the mask is boolean
    valid_mask_bool = valid_mask.astype(bool)

    # Calculate statistics
    total_pixels = PROJ_H * PROJ_W
    filled_pixels = np.sum(valid_mask_bool)
    coverage = filled_pixels / total_pixels

    # Class distribution (only for valid pixels)
    class_counts = np.zeros(20, dtype=np.int64)
    if filled_pixels > 0:
        valid_labels = label_image[valid_mask_bool]
        for cls_id in range(20):
            class_counts[cls_id] = np.sum(valid_labels == cls_id)

    # Save output files
    seq_output_dir = os.path.join(output_dir, sequence_str)
    os.makedirs(seq_output_dir, exist_ok=True)

    np.save(os.path.join(seq_output_dir, f"{scan_name}_range.npy"), range_image)
    np.save(os.path.join(seq_output_dir, f"{scan_name}_label.npy"), label_image)
    np.save(os.path.join(seq_output_dir, f"{scan_name}_mask.npy"), valid_mask_bool)

    return {
        "num_points": num_points,
        "coverage": coverage,
        "class_counts": class_counts,
    }


def gather_scan_list(data_dir, sequences):
    """Gather all scan files across the requested sequences.

    Args:
        data_dir: path to SemanticKITTI sequences directory.
        sequences: list of sequence strings (e.g., ['00', '01', ...]).

    Returns:
        List of (sequence_str, scan_filename) tuples.
    """
    scan_list = []
    for seq in sequences:
        velodyne_dir = os.path.join(data_dir, seq, "velodyne")
        if not os.path.isdir(velodyne_dir):
            print(f"  [WARN] Velodyne directory not found: {velodyne_dir}, skipping.")
            continue
        bin_files = sorted([
            f for f in os.listdir(velodyne_dir)
            if f.endswith(".bin")
        ])
        for bf in bin_files:
            scan_list.append((seq, bf))
    return scan_list


def print_statistics(results_by_sequence):
    """Print comprehensive statistics after processing.

    Args:
        results_by_sequence: dict mapping sequence -> list of result dicts.
    """
    print("\n" + "=" * 70)
    print("PROCESSING STATISTICS")
    print("=" * 70)

    total_scans = 0
    total_coverage = 0.0
    total_class_counts = np.zeros(20, dtype=np.int64)
    total_valid_scans = 0

    for seq in sorted(results_by_sequence.keys()):
        results = results_by_sequence[seq]
        num_scans = len(results)
        valid_results = [r for r in results if r is not None]
        num_valid = len(valid_results)

        if num_valid > 0:
            avg_coverage = np.mean([r["coverage"] for r in valid_results])
            seq_class_counts = np.sum(
                [r["class_counts"] for r in valid_results], axis=0
            )
        else:
            avg_coverage = 0.0
            seq_class_counts = np.zeros(20, dtype=np.int64)

        print(f"\n  Sequence {seq}:")
        print(f"    Total scans processed: {num_scans}")
        print(f"    Successfully projected: {num_valid}")
        print(f"    Average point coverage: {avg_coverage * 100:.2f}%")
        print(f"    Average empty pixel ratio: {(1.0 - avg_coverage) * 100:.2f}%")

        total_scans += num_scans
        total_valid_scans += num_valid
        total_coverage += avg_coverage * num_valid
        total_class_counts += seq_class_counts

    # Overall statistics
    print(f"\n{'─' * 70}")
    print("  OVERALL:")
    print(f"    Total scans: {total_scans}")
    print(f"    Successfully processed: {total_valid_scans}")
    if total_valid_scans > 0:
        overall_coverage = total_coverage / total_valid_scans
        print(f"    Average point coverage: {overall_coverage * 100:.2f}%")
        print(f"    Average empty pixel ratio: {(1.0 - overall_coverage) * 100:.2f}%")

    # Class distribution
    total_labeled_pixels = total_class_counts.sum()
    if total_labeled_pixels > 0:
        print(f"\n{'─' * 70}")
        print("  CLASS DISTRIBUTION (across all valid pixels):")
        print(f"    {'Class ID':<10}{'Name':<20}{'Count':<15}{'Percentage':<10}")
        print(f"    {'─' * 55}")
        for cls_id in range(20):
            count = total_class_counts[cls_id]
            pct = count / total_labeled_pixels * 100
            name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            print(f"    {cls_id:<10}{name:<20}{count:<15}{pct:.2f}%")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute range images from SemanticKITTI point clouds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data.py --data_dir /data/semantickitti/sequences --output_dir /data/rangenet_precomputed --sequences 00 01 02

  python prepare_data.py --data_dir ./dataset/sequences --output_dir ./output --sequences 00 --num_workers 4
        """,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to SemanticKITTI sequences directory (contains 00/, 01/, etc.)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for .npy files.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        type=str,
        default=["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
        help="List of sequence IDs to process (default: 00-10).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers for multiprocessing (default: 4).",
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Data directory does not exist: {args.data_dir}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("RangeNet++ Data Preparation: SemanticKITTI -> Range Images")
    print("=" * 70)
    print(f"  Data directory:   {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Sequences:        {args.sequences}")
    print(f"  Num workers:      {args.num_workers}")
    print(f"  Projection:       H={PROJ_H}, W={PROJ_W}")
    print(f"  Vertical FOV:     [{FOV_DOWN}, {FOV_UP}] degrees")
    print(f"  Horizontal FOV:   {FOV_HORIZONTAL} degrees")
    print(f"  Channels:         range, x, y, z, remission (5)")
    print("=" * 70)

    # Gather all scan files
    print("\nScanning for .bin files...")
    scan_list = gather_scan_list(args.data_dir, args.sequences)

    if not scan_list:
        print("[ERROR] No .bin files found. Check --data_dir and --sequences.")
        sys.exit(1)

    print(f"  Found {len(scan_list)} scans across {len(args.sequences)} sequence(s).")

    # Process scans using multiprocessing
    print(f"\nProcessing with {args.num_workers} worker(s)...")
    start_time = time.time()

    worker_fn = partial(process_scan, data_dir=args.data_dir, output_dir=args.output_dir)

    results_by_sequence = {seq: [] for seq in args.sequences}

    if args.num_workers <= 1:
        # Single-process mode (useful for debugging)
        for scan_info in tqdm(scan_list, desc="Processing scans", unit="scan"):
            result = worker_fn(scan_info)
            seq_str = scan_info[0]
            results_by_sequence[seq_str].append(result)
    else:
        # Multiprocessing mode
        with Pool(processes=args.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(worker_fn, scan_list),
                    total=len(scan_list),
                    desc="Processing scans",
                    unit="scan",
                )
            )
        # Organize results by sequence
        for scan_info, result in zip(scan_list, results):
            seq_str = scan_info[0]
            results_by_sequence[seq_str].append(result)

    elapsed = time.time() - start_time
    print(f"\nProcessing completed in {elapsed:.1f} seconds.")
    print(f"  Average: {elapsed / len(scan_list) * 1000:.1f} ms/scan")

    # Print statistics
    print_statistics(results_by_sequence)

    print(f"\nOutput saved to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
