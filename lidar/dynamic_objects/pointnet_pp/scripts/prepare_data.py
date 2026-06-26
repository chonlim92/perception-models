#!/usr/bin/env python3
"""
Prepare KITTI point cloud data for PointNet++ training.

Reads raw KITTI .bin point cloud files, performs ground plane removal via RANSAC,
applies range filtering, normalizes/subsamples point clouds, parses labels and
calibration files, and generates info pickle files for training.

Usage:
    python prepare_data.py --data_dir ./data/kitti --output_dir ./data/processed --split all
"""

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np


# =============================================================================
# Calibration Parsing
# =============================================================================

class KITTICalibration:
    """Parse and store KITTI calibration data with coordinate transforms."""

    def __init__(self, calib_filepath):
        calib = self._load_calib_file(calib_filepath)
        self.P0 = calib["P0"].reshape(3, 4)
        self.P1 = calib["P1"].reshape(3, 4)
        self.P2 = calib["P2"].reshape(3, 4)
        self.P3 = calib["P3"].reshape(3, 4)
        self.R0_rect = calib["R0_rect"].reshape(3, 3)
        self.Tr_velo_to_cam = calib["Tr_velo_to_cam"].reshape(3, 4)
        self.Tr_imu_to_velo = calib["Tr_imu_to_velo"].reshape(3, 4)

    @staticmethod
    def _load_calib_file(filepath):
        """Read calibration file and return dict of numpy arrays."""
        data = {}
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                values = np.array([float(x) for x in value.strip().split()])
                data[key] = values
        return data

    def velo_to_cam(self, points):
        """Transform points from velodyne to camera coordinates.

        Args:
            points: (N, 3) array in velodyne frame.

        Returns:
            (N, 3) array in rectified camera coordinates.
        """
        n = points.shape[0]
        points_hom = np.hstack([points, np.ones((n, 1))])
        cam_points = points_hom @ self.Tr_velo_to_cam.T
        rect_points = cam_points @ self.R0_rect.T
        return rect_points

    def cam_to_velo(self, points):
        """Transform points from rectified camera to velodyne coordinates.

        Args:
            points: (N, 3) array in rectified camera coordinates.

        Returns:
            (N, 3) array in velodyne frame.
        """
        R0_inv = np.linalg.inv(self.R0_rect)
        cam_points = points @ R0_inv.T
        Tr = self.Tr_velo_to_cam
        R = Tr[:, :3]
        t = Tr[:, 3]
        R_inv = np.linalg.inv(R)
        velo_points = (cam_points - t) @ R_inv.T
        return velo_points

    def velo_to_image(self, points, proj_matrix=None):
        """Project velodyne points onto the image plane.

        Args:
            points: (N, 3) array in velodyne frame.
            proj_matrix: (3, 4) projection matrix. Defaults to P2.

        Returns:
            (N, 2) array of image coordinates (u, v).
        """
        if proj_matrix is None:
            proj_matrix = self.P2
        rect_points = self.velo_to_cam(points)
        n = rect_points.shape[0]
        rect_hom = np.hstack([rect_points, np.ones((n, 1))])
        img_points = rect_hom @ proj_matrix.T
        img_points[:, 0] /= img_points[:, 2]
        img_points[:, 1] /= img_points[:, 2]
        return img_points[:, :2]

    def to_dict(self):
        """Serialize calibration to a dictionary."""
        return {
            "P0": self.P0.copy(),
            "P1": self.P1.copy(),
            "P2": self.P2.copy(),
            "P3": self.P3.copy(),
            "R0_rect": self.R0_rect.copy(),
            "Tr_velo_to_cam": self.Tr_velo_to_cam.copy(),
            "Tr_imu_to_velo": self.Tr_imu_to_velo.copy(),
        }


# =============================================================================
# Label Parsing
# =============================================================================

RELEVANT_CLASSES = {"Car", "Pedestrian", "Cyclist"}

CLASS_TO_ID = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


def parse_kitti_label(label_filepath, calib=None):
    """Parse a KITTI label file.

    Each line: type truncated occluded alpha bbox2d(4) dimensions(3) location(3) rotation_y [score]

    Args:
        label_filepath: Path to label .txt file.
        calib: Optional KITTICalibration object for coordinate transforms.

    Returns:
        List of dicts with parsed object annotations.
    """
    objects = []
    if not os.path.exists(label_filepath):
        return objects

    with open(label_filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            obj_type = parts[0]
            if obj_type not in RELEVANT_CLASSES:
                continue

            truncated = float(parts[1])
            occluded = int(parts[2])
            alpha = float(parts[3])
            bbox2d = np.array([float(parts[i]) for i in range(4, 8)])
            height = float(parts[8])
            width = float(parts[9])
            length = float(parts[10])
            location = np.array([float(parts[11]), float(parts[12]), float(parts[13])])
            rotation_y = float(parts[14])
            score = float(parts[15]) if len(parts) > 15 else -1.0

            obj = {
                "type": obj_type,
                "class_id": CLASS_TO_ID[obj_type],
                "truncated": truncated,
                "occluded": occluded,
                "alpha": alpha,
                "bbox2d": bbox2d,
                "dimensions": np.array([height, width, length]),
                "location_cam": location,
                "rotation_y": rotation_y,
                "score": score,
            }

            # Convert location from camera to velodyne frame
            if calib is not None:
                loc_velo = calib.cam_to_velo(location.reshape(1, 3))
                obj["location_velo"] = loc_velo.flatten()
                # 3D bounding box in velodyne: center, dimensions, heading
                obj["bbox3d_velo"] = np.array([
                    loc_velo[0, 0], loc_velo[0, 1], loc_velo[0, 2],
                    length, width, height,
                    rotation_y
                ])

            objects.append(obj)

    return objects


# =============================================================================
# RANSAC Ground Plane Removal
# =============================================================================

def fit_plane_ransac(points, max_iterations=200, distance_threshold=0.2,
                     min_inlier_ratio=0.3):
    """Fit a ground plane using RANSAC.

    Args:
        points: (N, 3) point cloud array.
        max_iterations: Maximum RANSAC iterations.
        distance_threshold: Distance to plane to be considered inlier (meters).
        min_inlier_ratio: Minimum fraction of inliers for a valid plane.

    Returns:
        Tuple of (plane_params, inlier_mask):
            plane_params: (a, b, c, d) such that ax + by + cz + d = 0
            inlier_mask: Boolean array, True for ground points.
        Returns (None, None) if no valid plane is found.
    """
    n_points = points.shape[0]
    if n_points < 3:
        return None, None

    best_inlier_count = 0
    best_plane = None
    best_mask = None

    rng = np.random.default_rng(42)

    for _ in range(max_iterations):
        # Sample 3 random points
        indices = rng.choice(n_points, size=3, replace=False)
        p1, p2, p3 = points[indices[0]], points[indices[1]], points[indices[2]]

        # Compute plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal = normal / norm_len

        # Ensure normal points upward (positive z in velodyne frame)
        if normal[2] < 0:
            normal = -normal

        d = -np.dot(normal, p1)

        # Compute distances
        distances = np.abs(points @ normal + d)
        inlier_mask = distances < distance_threshold
        inlier_count = np.sum(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = np.array([normal[0], normal[1], normal[2], d])
            best_mask = inlier_mask

    if best_plane is None:
        return None, None

    if best_inlier_count / n_points < min_inlier_ratio:
        # Plane found but too few inliers; still return it with warning
        pass

    return best_plane, best_mask


def remove_ground_plane(points, max_iterations=200, distance_threshold=0.2):
    """Remove ground plane points from a point cloud.

    Args:
        points: (N, 4) array (x, y, z, reflectance).

    Returns:
        Tuple of (non_ground_points, ground_params):
            non_ground_points: (M, 4) points with ground removed.
            ground_params: (4,) plane parameters or None.
    """
    xyz = points[:, :3]
    plane_params, inlier_mask = fit_plane_ransac(
        xyz, max_iterations=max_iterations, distance_threshold=distance_threshold
    )

    if plane_params is None:
        return points, None

    non_ground = points[~inlier_mask]
    return non_ground, plane_params


# =============================================================================
# Range Filtering
# =============================================================================

def filter_point_cloud(points, max_range=70.0, min_x=0.0, max_height=3.0):
    """Apply range and spatial filters to a point cloud.

    Args:
        points: (N, 4) array (x, y, z, reflectance).
        max_range: Maximum distance from sensor in meters.
        min_x: Minimum x value (removes points behind vehicle if > 0).
        max_height: Maximum height above ground level in meters.

    Returns:
        Filtered (M, 4) point cloud.
    """
    xyz = points[:, :3]

    # Range filter: Euclidean distance from sensor origin
    distances = np.linalg.norm(xyz, axis=1)
    range_mask = distances <= max_range

    # Forward filter: remove points behind the vehicle
    forward_mask = points[:, 0] >= min_x

    # Height filter: remove points too high
    height_mask = points[:, 2] <= max_height

    combined_mask = range_mask & forward_mask & height_mask
    return points[combined_mask]


# =============================================================================
# Subsampling
# =============================================================================

def farthest_point_sampling(points, num_samples):
    """Subsample points using farthest point sampling (FPS).

    Args:
        points: (N, 3+) array.
        num_samples: Target number of points.

    Returns:
        (num_samples, C) subsampled point cloud.
    """
    n = points.shape[0]
    if n <= num_samples:
        # Pad by repeating random points
        pad_indices = np.random.choice(n, size=num_samples - n, replace=True)
        return np.vstack([points, points[pad_indices]])

    selected_indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(n, np.inf)

    # Start from a random point
    current_idx = np.random.randint(0, n)
    selected_indices[0] = current_idx

    xyz = points[:, :3]

    for i in range(1, num_samples):
        current_point = xyz[current_idx]
        dist_to_current = np.sum((xyz - current_point) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_current)
        current_idx = np.argmax(distances)
        selected_indices[i] = current_idx

    return points[selected_indices]


def random_subsample(points, num_samples):
    """Subsample points randomly.

    Args:
        points: (N, C) array.
        num_samples: Target number of points.

    Returns:
        (num_samples, C) subsampled point cloud.
    """
    n = points.shape[0]
    if n <= num_samples:
        pad_indices = np.random.choice(n, size=num_samples - n, replace=True)
        return np.vstack([points, points[pad_indices]])

    indices = np.random.choice(n, size=num_samples, replace=False)
    return points[indices]


def normalize_point_cloud(points):
    """Center a point cloud at its centroid.

    Args:
        points: (N, C) array where first 3 columns are x, y, z.

    Returns:
        Tuple of (centered_points, centroid):
            centered_points: (N, C) centered array.
            centroid: (3,) original centroid.
    """
    centroid = np.mean(points[:, :3], axis=0)
    centered = points.copy()
    centered[:, :3] -= centroid
    return centered, centroid


# =============================================================================
# Data Loading
# =============================================================================

def load_point_cloud(bin_filepath):
    """Load a KITTI .bin point cloud file.

    Args:
        bin_filepath: Path to .bin file (N x 4 float32: x, y, z, reflectance).

    Returns:
        (N, 4) numpy array.
    """
    points = np.fromfile(bin_filepath, dtype=np.float32).reshape(-1, 4)
    return points


def load_split_indices(data_dir, split):
    """Load sample indices from KITTI ImageSets files.

    Args:
        data_dir: Root KITTI data directory.
        split: One of 'train', 'val', 'test'.

    Returns:
        List of sample index strings (e.g., ['000001', '000002', ...]).
    """
    imagesets_dir = os.path.join(data_dir, "ImageSets")
    split_file = os.path.join(imagesets_dir, f"{split}.txt")

    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"Split file not found: {split_file}. "
            f"Expected ImageSets directory at {imagesets_dir} with "
            f"train.txt, val.txt, test.txt files."
        )

    with open(split_file, "r") as f:
        indices = [line.strip() for line in f if line.strip()]

    return indices


# =============================================================================
# Processing Pipeline
# =============================================================================

def process_sample(sample_idx, data_dir, num_points, use_fps=True):
    """Process a single KITTI sample.

    Args:
        sample_idx: Sample index string (e.g., '000001').
        data_dir: Root KITTI data directory.
        num_points: Target number of points after subsampling.
        use_fps: If True, use farthest point sampling; otherwise random.

    Returns:
        Dict with processed data and metadata, or None on failure.
    """
    # Construct file paths
    bin_path = os.path.join(data_dir, "training", "velodyne", f"{sample_idx}.bin")
    calib_path = os.path.join(data_dir, "training", "calib", f"{sample_idx}.txt")
    label_path = os.path.join(data_dir, "training", "label_2", f"{sample_idx}.txt")

    # Check if velodyne file exists
    if not os.path.exists(bin_path):
        print(f"  [WARNING] Velodyne file not found: {bin_path}")
        return None

    # Load point cloud
    raw_points = load_point_cloud(bin_path)
    num_original = raw_points.shape[0]

    # Load calibration
    calib = None
    calib_dict = None
    if os.path.exists(calib_path):
        calib = KITTICalibration(calib_path)
        calib_dict = calib.to_dict()

    # Parse labels
    objects = parse_kitti_label(label_path, calib=calib)

    # Ground plane removal
    filtered_points, ground_params = remove_ground_plane(raw_points)

    # Range filtering
    filtered_points = filter_point_cloud(
        filtered_points, max_range=70.0, min_x=0.0, max_height=3.0
    )

    num_after_filter = filtered_points.shape[0]

    if num_after_filter == 0:
        print(f"  [WARNING] No points remaining after filtering for {sample_idx}")
        return None

    # Normalize (center)
    normalized_points, centroid = normalize_point_cloud(filtered_points)

    # Subsample to fixed size
    if use_fps and num_after_filter > num_points:
        final_points = farthest_point_sampling(normalized_points, num_points)
    else:
        final_points = random_subsample(normalized_points, num_points)

    # Extract bounding box info
    bboxes = []
    class_labels = []
    for obj in objects:
        if "bbox3d_velo" in obj:
            bboxes.append(obj["bbox3d_velo"])
        class_labels.append(obj["class_id"])

    bboxes = np.array(bboxes) if bboxes else np.zeros((0, 7))
    class_labels = np.array(class_labels) if class_labels else np.zeros((0,), dtype=np.int32)

    sample_info = {
        "filename": sample_idx,
        "num_points_original": num_original,
        "num_points_processed": final_points.shape[0],
        "bounding_boxes": bboxes,
        "class_labels": class_labels,
        "calibration_info": calib_dict,
        "ground_plane": ground_params,
        "centroid": centroid,
        "objects": objects,
    }

    return {
        "points": final_points,
        "info": sample_info,
    }


def process_split(split, data_dir, output_dir, num_points, use_fps=True):
    """Process all samples in a given split.

    Args:
        split: Split name ('train', 'val', 'test').
        data_dir: Root KITTI data directory.
        output_dir: Output directory for processed data.
        num_points: Target number of points per sample.
        use_fps: Use farthest point sampling if True.

    Returns:
        Dict with processing statistics.
    """
    print(f"\n{'='*60}")
    print(f"Processing split: {split}")
    print(f"{'='*60}")

    start_time = time.time()

    # Load split indices
    indices = load_split_indices(data_dir, split)
    print(f"Found {len(indices)} samples in {split} split")

    # Create output directories
    split_output_dir = os.path.join(output_dir, split)
    points_dir = os.path.join(split_output_dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    infos = []
    point_counts_original = []
    point_counts_processed = []
    class_counter = {"Car": 0, "Pedestrian": 0, "Cyclist": 0}
    failed_samples = []

    for i, sample_idx in enumerate(indices):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(indices)}: {sample_idx}")

        result = process_sample(sample_idx, data_dir, num_points, use_fps=use_fps)

        if result is None:
            failed_samples.append(sample_idx)
            continue

        # Save processed point cloud
        output_path = os.path.join(points_dir, f"{sample_idx}.npy")
        np.save(output_path, result["points"])

        # Collect info
        infos.append(result["info"])
        point_counts_original.append(result["info"]["num_points_original"])
        point_counts_processed.append(result["info"]["num_points_processed"])

        # Count classes
        for obj in result["info"]["objects"]:
            if obj["type"] in class_counter:
                class_counter[obj["type"]] += 1

    # Save info pickle
    info_path = os.path.join(output_dir, f"{split}_infos.pkl")
    with open(info_path, "wb") as f:
        pickle.dump(infos, f)
    print(f"  Saved info file: {info_path}")

    elapsed = time.time() - start_time

    stats = {
        "split": split,
        "total_samples": len(indices),
        "processed_samples": len(infos),
        "failed_samples": len(failed_samples),
        "failed_indices": failed_samples,
        "point_counts_original": point_counts_original,
        "point_counts_processed": point_counts_processed,
        "class_distribution": class_counter,
        "processing_time_seconds": elapsed,
    }

    return stats


def print_statistics(all_stats):
    """Print summary statistics for all processed splits.

    Args:
        all_stats: List of stats dicts from process_split.
    """
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")

    total_samples = 0
    total_processed = 0
    total_failed = 0
    total_time = 0.0
    combined_class_dist = {"Car": 0, "Pedestrian": 0, "Cyclist": 0}
    all_original_counts = []
    all_processed_counts = []

    for stats in all_stats:
        split = stats["split"]
        print(f"\n--- Split: {split} ---")
        print(f"  Total samples:     {stats['total_samples']}")
        print(f"  Processed:         {stats['processed_samples']}")
        print(f"  Failed:            {stats['failed_samples']}")
        print(f"  Processing time:   {stats['processing_time_seconds']:.1f}s")

        if stats["point_counts_original"]:
            orig = np.array(stats["point_counts_original"])
            proc = np.array(stats["point_counts_processed"])
            print(f"  Original points:   min={orig.min()}, max={orig.max()}, "
                  f"mean={orig.mean():.0f}, std={orig.std():.0f}")
            print(f"  Processed points:  min={proc.min()}, max={proc.max()}, "
                  f"mean={proc.mean():.0f}, std={proc.std():.0f}")
            all_original_counts.extend(stats["point_counts_original"])
            all_processed_counts.extend(stats["point_counts_processed"])

        print(f"  Class distribution:")
        for cls, count in stats["class_distribution"].items():
            print(f"    {cls}: {count}")
            combined_class_dist[cls] += count

        total_samples += stats["total_samples"]
        total_processed += stats["processed_samples"]
        total_failed += stats["failed_samples"]
        total_time += stats["processing_time_seconds"]

    print(f"\n--- Overall ---")
    print(f"  Total samples:     {total_samples}")
    print(f"  Total processed:   {total_processed}")
    print(f"  Total failed:      {total_failed}")
    print(f"  Total time:        {total_time:.1f}s")

    if all_original_counts:
        orig_all = np.array(all_original_counts)
        proc_all = np.array(all_processed_counts)
        print(f"  Overall original points:  min={orig_all.min()}, max={orig_all.max()}, "
              f"mean={orig_all.mean():.0f}")
        print(f"  Overall processed points: min={proc_all.min()}, max={proc_all.max()}, "
              f"mean={proc_all.mean():.0f}")

    print(f"  Combined class distribution:")
    for cls, count in combined_class_dist.items():
        print(f"    {cls}: {count}")

    print(f"\n{'='*60}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare KITTI point cloud data for PointNet++ training."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/kitti",
        help="Path to raw KITTI data directory (default: ./data/kitti)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Path for processed output (default: ./data/processed)",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=16384,
        help="Number of points per sample after subsampling (default: 16384)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split to process: train, val, test, or all (default: all)",
    )
    parser.add_argument(
        "--use_fps",
        action="store_true",
        default=True,
        help="Use farthest point sampling (default: True). Use --no_fps for random.",
    )
    parser.add_argument(
        "--no_fps",
        action="store_true",
        default=False,
        help="Use random subsampling instead of FPS.",
    )
    parser.add_argument(
        "--max_range",
        type=float,
        default=70.0,
        help="Maximum range from sensor in meters (default: 70.0)",
    )
    parser.add_argument(
        "--max_height",
        type=float,
        default=3.0,
        help="Maximum height above ground in meters (default: 3.0)",
    )
    parser.add_argument(
        "--ransac_iterations",
        type=int,
        default=200,
        help="RANSAC iterations for ground plane fitting (default: 200)",
    )
    parser.add_argument(
        "--ground_threshold",
        type=float,
        default=0.2,
        help="RANSAC distance threshold for ground plane (default: 0.2m)",
    )

    args = parser.parse_args()

    use_fps = args.use_fps and not args.no_fps

    print("KITTI Data Preparation for PointNet++")
    print(f"{'='*60}")
    print(f"Data directory:      {os.path.abspath(args.data_dir)}")
    print(f"Output directory:    {os.path.abspath(args.output_dir)}")
    print(f"Points per sample:   {args.num_points}")
    print(f"Split:               {args.split}")
    print(f"Sampling method:     {'FPS' if use_fps else 'Random'}")
    print(f"Max range:           {args.max_range}m")
    print(f"Max height:          {args.max_height}m")
    print(f"RANSAC iterations:   {args.ransac_iterations}")
    print(f"Ground threshold:    {args.ground_threshold}m")

    # Validate data directory
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine splits to process
    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]

    # Filter to only splits that have ImageSets files available
    available_splits = []
    for split in splits:
        split_file = os.path.join(args.data_dir, "ImageSets", f"{split}.txt")
        if os.path.exists(split_file):
            available_splits.append(split)
        else:
            print(f"\n[INFO] Skipping split '{split}': "
                  f"ImageSets/{split}.txt not found")

    if not available_splits:
        raise FileNotFoundError(
            f"No valid split files found in {os.path.join(args.data_dir, 'ImageSets')}. "
            f"Expected train.txt, val.txt, or test.txt."
        )

    # Process each split
    all_stats = []
    for split in available_splits:
        stats = process_split(
            split=split,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_points=args.num_points,
            use_fps=use_fps,
        )
        all_stats.append(stats)

    # Print summary statistics
    print_statistics(all_stats)

    # Save combined statistics
    stats_path = os.path.join(args.output_dir, "processing_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(all_stats, f)
    print(f"\nStatistics saved to: {stats_path}")
    print("Done.")


if __name__ == "__main__":
    main()
