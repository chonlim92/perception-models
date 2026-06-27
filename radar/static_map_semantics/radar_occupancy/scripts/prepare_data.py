# [IMPLEMENTED BY CLAUDE - was missing]
"""
Data preparation script for radar occupancy prediction.
Converts raw radar point clouds to pillar format and generates
occupancy ground truth from LiDAR using raytrace-based free space estimation.
"""

import argparse
import os
import numpy as np
import json
import glob

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from multiprocessing import Pool
from functools import partial


class PillarGenerator:
    """Converts raw radar point clouds into pillar representation."""

    def __init__(self, config):
        """
        Args:
            config: dict with grid parameters:
                - grid_size: [H, W] grid dimensions
                - cell_size: [dx, dy] size of each cell in meters
                - x_range: [x_min, x_max] range in x direction
                - y_range: [y_min, y_max] range in y direction
                - max_pillars: maximum number of non-empty pillars
                - max_points_per_pillar: maximum points kept per pillar
                - feature_dim: feature dimension (default 9)
        """
        self.grid_size = config.get("grid_size", [256, 256])
        self.cell_size = config.get("cell_size", [0.4, 0.4])
        self.x_range = config.get("x_range", [-51.2, 51.2])
        self.y_range = config.get("y_range", [-51.2, 51.2])
        self.max_pillars = config.get("max_pillars", 10000)
        self.max_points_per_pillar = config.get("max_points_per_pillar", 20)
        self.feature_dim = config.get("feature_dim", 9)

    def points_to_pillars(self, points):
        """
        Convert radar points to pillar representation.

        Args:
            points: (N, 6) array with columns [x, y, z, rcs, vr_comp, dt]

        Returns:
            pillar_features: (max_pillars, max_points_per_pillar, 9) augmented features
            pillar_indices: (max_pillars, 2) grid indices [row, col] per pillar
            num_pillars: int, number of non-empty pillars
        """
        if points.shape[0] == 0:
            return (
                np.zeros((self.max_pillars, self.max_points_per_pillar, self.feature_dim), dtype=np.float32),
                np.zeros((self.max_pillars, 2), dtype=np.int32),
                0,
            )

        # Compute grid indices for each point
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        dx, dy = self.cell_size

        # Grid column (x-axis) and row (y-axis) indices
        col_indices = np.floor((points[:, 0] - x_min) / dx).astype(np.int32)
        row_indices = np.floor((points[:, 1] - y_min) / dy).astype(np.int32)

        # Filter points outside grid boundaries
        H, W = self.grid_size
        valid_mask = (
            (col_indices >= 0) & (col_indices < W) &
            (row_indices >= 0) & (row_indices < H)
        )
        points = points[valid_mask]
        col_indices = col_indices[valid_mask]
        row_indices = row_indices[valid_mask]

        if points.shape[0] == 0:
            return (
                np.zeros((self.max_pillars, self.max_points_per_pillar, self.feature_dim), dtype=np.float32),
                np.zeros((self.max_pillars, 2), dtype=np.int32),
                0,
            )

        # Group points into pillars by grid cell
        # Create unique pillar IDs from (row, col)
        pillar_ids = row_indices * W + col_indices
        unique_pillars, inverse_indices = np.unique(pillar_ids, return_inverse=True)

        num_pillars = min(len(unique_pillars), self.max_pillars)

        # If more unique pillars than max_pillars, keep the ones with most points
        if len(unique_pillars) > self.max_pillars:
            # Count points per pillar and keep top-k
            pillar_counts = np.bincount(inverse_indices, minlength=len(unique_pillars))
            top_k_indices = np.argsort(pillar_counts)[::-1][:self.max_pillars]
            selected_pillars = unique_pillars[top_k_indices]
            # Create mask for points belonging to selected pillars
            selected_set = set(selected_pillars.tolist())
            point_mask = np.array([pid in selected_set for pid in pillar_ids], dtype=bool)
            points = points[point_mask]
            col_indices = col_indices[point_mask]
            row_indices = row_indices[point_mask]
            pillar_ids = pillar_ids[point_mask]
            unique_pillars = selected_pillars
            _, inverse_indices = np.unique(pillar_ids, return_inverse=True)

        # Initialize output arrays
        pillar_features = np.zeros(
            (self.max_pillars, self.max_points_per_pillar, self.feature_dim), dtype=np.float32
        )
        pillar_indices = np.zeros((self.max_pillars, 2), dtype=np.int32)

        # Fill pillars
        for p_idx in range(num_pillars):
            # Get points belonging to this pillar
            point_mask = inverse_indices == p_idx
            pillar_points = points[point_mask]

            # Limit to max_points_per_pillar (random subsample if needed)
            n_pts = pillar_points.shape[0]
            if n_pts > self.max_points_per_pillar:
                choice = np.random.choice(n_pts, self.max_points_per_pillar, replace=False)
                pillar_points = pillar_points[choice]
                n_pts = self.max_points_per_pillar

            # Compute pillar center (mean x, y, z of points in pillar)
            center_x = pillar_points[:, 0].mean()
            center_y = pillar_points[:, 1].mean()
            center_z = pillar_points[:, 2].mean()

            # Augment features: original 6 + offsets to pillar center
            x_offset = pillar_points[:, 0] - center_x
            y_offset = pillar_points[:, 1] - center_y
            z_offset = pillar_points[:, 2] - center_z

            augmented = np.zeros((n_pts, self.feature_dim), dtype=np.float32)
            augmented[:, :6] = pillar_points[:, :6]  # original features
            augmented[:, 6] = x_offset
            augmented[:, 7] = y_offset
            augmented[:, 8] = z_offset

            pillar_features[p_idx, :n_pts, :] = augmented

            # Store grid index for this pillar
            pillar_row = unique_pillars[p_idx] // W
            pillar_col = unique_pillars[p_idx] % W
            pillar_indices[p_idx] = [pillar_row, pillar_col]

        return pillar_features, pillar_indices, num_pillars


class OccupancyGTGenerator:
    """Generates occupancy ground truth from LiDAR point clouds using raycasting."""

    def __init__(self, config):
        """
        Args:
            config: dict with parameters:
                - grid_size: [H, W]
                - cell_size: [dx, dy]
                - x_range: [x_min, x_max]
                - y_range: [y_min, y_max]
                - occupancy_threshold: min lidar points to mark cell occupied
                - free_threshold: min rays passing through to mark cell free
        """
        self.grid_size = config.get("grid_size", [256, 256])
        self.cell_size = config.get("cell_size", [0.4, 0.4])
        self.x_range = config.get("x_range", [-51.2, 51.2])
        self.y_range = config.get("y_range", [-51.2, 51.2])
        self.occupancy_threshold = config.get("occupancy_threshold", 1)
        self.free_threshold = config.get("free_threshold", 1)

    def _bresenham_2d(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm for 2D grid traversal.
        Returns list of (row, col) cells traversed from (x0, y0) to (x1, y1).
        Does NOT include the endpoint cell.
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cx, cy = x0, y0
        while True:
            if cx == x1 and cy == y1:
                break
            cells.append((cy, cx))  # (row, col)
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

        return cells

    def generate_from_lidar(self, lidar_points, ego_pose=None):
        """
        Generate occupancy ground truth from LiDAR point cloud.

        Args:
            lidar_points: (M, 4) array [x, y, z, intensity]
            ego_pose: optional 4x4 transformation matrix (if points need transformation)

        Returns:
            occupancy_gt: (H, W) array with values:
                0 = free
                1 = occupied
                2 = unknown
        """
        H, W = self.grid_size
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        dx, dy = self.cell_size

        # Transform points if ego_pose is provided
        if ego_pose is not None:
            pts_hom = np.ones((lidar_points.shape[0], 4), dtype=np.float64)
            pts_hom[:, :3] = lidar_points[:, :3]
            transformed = (ego_pose @ pts_hom.T).T
            lidar_points = np.column_stack([transformed[:, :3], lidar_points[:, 3]])

        # Initialize occupancy grid as unknown
        occupancy_gt = np.full((H, W), 2, dtype=np.uint8)

        # Compute grid indices for lidar hit points
        col_indices = np.floor((lidar_points[:, 0] - x_min) / dx).astype(np.int32)
        row_indices = np.floor((lidar_points[:, 1] - y_min) / dy).astype(np.int32)

        # Filter to points within grid
        valid_mask = (
            (col_indices >= 0) & (col_indices < W) &
            (row_indices >= 0) & (row_indices < H)
        )
        valid_cols = col_indices[valid_mask]
        valid_rows = row_indices[valid_mask]

        # Count hits per cell for occupancy
        hit_count = np.zeros((H, W), dtype=np.int32)
        np.add.at(hit_count, (valid_rows, valid_cols), 1)

        # Mark cells with enough hits as occupied
        occupied_mask = hit_count >= self.occupancy_threshold
        occupancy_gt[occupied_mask] = 1

        # Raytrace from sensor origin (0, 0) to each hit point to mark free cells
        # Sensor origin in grid coordinates
        origin_col = int(np.floor((0.0 - x_min) / dx))
        origin_row = int(np.floor((0.0 - y_min) / dy))

        # Clamp origin to grid if outside
        origin_col = np.clip(origin_col, 0, W - 1)
        origin_row = np.clip(origin_row, 0, H - 1)

        # Track free ray count per cell
        free_count = np.zeros((H, W), dtype=np.int32)

        # Raytrace to each valid hit point
        # For efficiency, subsample if too many points
        max_rays = 50000
        if valid_cols.shape[0] > max_rays:
            subsample_idx = np.random.choice(valid_cols.shape[0], max_rays, replace=False)
            ray_cols = valid_cols[subsample_idx]
            ray_rows = valid_rows[subsample_idx]
        else:
            ray_cols = valid_cols
            ray_rows = valid_rows

        for i in range(len(ray_cols)):
            end_col = ray_cols[i]
            end_row = ray_rows[i]

            # Get traversed cells using Bresenham (excludes endpoint)
            traversed = self._bresenham_2d(origin_col, origin_row, end_col, end_row)

            for (r, c) in traversed:
                if 0 <= r < H and 0 <= c < W:
                    free_count[r, c] += 1

        # Mark cells with enough ray traversals as free (if not already occupied)
        free_mask = (free_count >= self.free_threshold) & (~occupied_mask)
        occupancy_gt[free_mask] = 0

        return occupancy_gt


def process_sample(sample_info, pillar_gen, gt_gen, output_dir):
    """
    Process a single sample: load radar/lidar, generate pillars and GT.

    Args:
        sample_info: dict with keys:
            - radar_path: path to radar point cloud file (.bin or .npy)
            - lidar_path: path to lidar point cloud file (.bin or .npy)
            - sample_token: unique sample identifier
            - ego_pose: optional 4x4 pose matrix (as list)
        pillar_gen: PillarGenerator instance
        gt_gen: OccupancyGTGenerator instance
        output_dir: directory to save processed outputs
    """
    try:
        sample_token = sample_info["sample_token"]
        radar_path = sample_info["radar_path"]
        lidar_path = sample_info["lidar_path"]
        ego_pose = sample_info.get("ego_pose", None)

        # Load radar points (N, 6): [x, y, z, rcs, vr_comp, dt]
        if radar_path.endswith(".npy"):
            radar_points = np.load(radar_path)
        elif radar_path.endswith(".bin"):
            radar_points = np.fromfile(radar_path, dtype=np.float32).reshape(-1, 6)
        else:
            raise ValueError(f"Unsupported radar format: {radar_path}")

        # Generate pillar representation
        pillar_features, pillar_indices, num_pillars = pillar_gen.points_to_pillars(radar_points)

        # Load lidar points (M, 4): [x, y, z, intensity]
        if lidar_path.endswith(".npy"):
            lidar_points = np.load(lidar_path)
        elif lidar_path.endswith(".bin"):
            lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        else:
            raise ValueError(f"Unsupported lidar format: {lidar_path}")

        # Convert ego_pose to numpy array if provided
        ego_pose_arr = None
        if ego_pose is not None:
            ego_pose_arr = np.array(ego_pose, dtype=np.float64).reshape(4, 4)

        # Generate occupancy ground truth
        occupancy_gt = gt_gen.generate_from_lidar(lidar_points, ego_pose=ego_pose_arr)

        # Save processed data
        output_path = os.path.join(output_dir, f"{sample_token}.npz")
        np.savez_compressed(
            output_path,
            pillar_features=pillar_features,
            pillar_indices=pillar_indices,
            num_pillars=np.array(num_pillars, dtype=np.int32),
            occupancy_gt=occupancy_gt,
            radar_points=radar_points,
        )

        return {
            "sample_token": sample_token,
            "output_path": output_path,
            "num_radar_points": radar_points.shape[0],
            "num_pillars": int(num_pillars),
            "occupancy_stats": {
                "free": int((occupancy_gt == 0).sum()),
                "occupied": int((occupancy_gt == 1).sum()),
                "unknown": int((occupancy_gt == 2).sum()),
            },
            "status": "success",
        }

    except Exception as e:
        return {
            "sample_token": sample_info.get("sample_token", "unknown"),
            "status": "error",
            "error": str(e),
        }


def _process_sample_wrapper(args):
    """Wrapper for multiprocessing Pool that unpacks arguments."""
    sample_info, config, output_dir = args
    pillar_gen = PillarGenerator(config)
    gt_gen = OccupancyGTGenerator(config)
    return process_sample(sample_info, pillar_gen, gt_gen, output_dir)


def process_split(data_dir, output_dir, config, split, num_workers=4):
    """
    Process all samples for a given split.

    Args:
        data_dir: nuScenes data root directory
        output_dir: output directory for processed data
        config: dict with processing configuration
        split: split name (e.g., 'train', 'val', 'test')
        num_workers: number of parallel workers
    """
    split_output_dir = os.path.join(output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)

    # Discover samples for this split
    # Look for split file listing sample tokens
    split_file = os.path.join(data_dir, "splits", f"{split}.json")

    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            sample_tokens = json.load(f)
    else:
        # Fallback: discover radar files directly
        radar_dir = os.path.join(data_dir, "radar", split)
        if not os.path.isdir(radar_dir):
            # Try flat structure
            radar_dir = os.path.join(data_dir, "radar")

        radar_files = sorted(
            glob.glob(os.path.join(radar_dir, "*.bin")) +
            glob.glob(os.path.join(radar_dir, "*.npy"))
        )
        sample_tokens = [
            os.path.splitext(os.path.basename(f))[0] for f in radar_files
        ]

    if not sample_tokens:
        print(f"[WARNING] No samples found for split '{split}' in {data_dir}")
        return

    print(f"Processing split '{split}': {len(sample_tokens)} samples")

    # Build sample info list
    sample_infos = []
    for token in sample_tokens:
        # Determine file paths (support multiple directory structures)
        radar_path = None
        for ext in [".bin", ".npy"]:
            for subdir in [os.path.join("radar", split), "radar"]:
                candidate = os.path.join(data_dir, subdir, token + ext)
                if os.path.exists(candidate):
                    radar_path = candidate
                    break
            if radar_path:
                break

        lidar_path = None
        for ext in [".bin", ".npy"]:
            for subdir in [os.path.join("lidar", split), "lidar"]:
                candidate = os.path.join(data_dir, subdir, token + ext)
                if os.path.exists(candidate):
                    lidar_path = candidate
                    break
            if lidar_path:
                break

        if radar_path is None or lidar_path is None:
            print(f"  [SKIP] Missing data for sample {token} "
                  f"(radar={'found' if radar_path else 'MISSING'}, "
                  f"lidar={'found' if lidar_path else 'MISSING'})")
            continue

        # Load ego pose if available
        ego_pose = None
        pose_path = os.path.join(data_dir, "poses", f"{token}.npy")
        if os.path.exists(pose_path):
            ego_pose = np.load(pose_path).tolist()

        sample_infos.append({
            "sample_token": token,
            "radar_path": radar_path,
            "lidar_path": lidar_path,
            "ego_pose": ego_pose,
        })

    if not sample_infos:
        print(f"[WARNING] No valid samples to process for split '{split}'")
        return

    # Process samples in parallel
    results = []
    if num_workers <= 1:
        # Single-process mode (useful for debugging)
        for sample_info in tqdm(sample_infos, desc=f"Processing {split}"):
            result = process_sample(
                sample_info,
                PillarGenerator(config),
                OccupancyGTGenerator(config),
                split_output_dir,
            )
            results.append(result)
    else:
        # Multi-process mode
        worker_args = [(info, config, split_output_dir) for info in sample_infos]
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_sample_wrapper, worker_args),
                total=len(worker_args),
                desc=f"Processing {split}",
            ))

    # Compute statistics
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    total_radar_points = sum(r["num_radar_points"] for r in successful)
    total_pillars = sum(r["num_pillars"] for r in successful)

    stats = {
        "split": split,
        "total_samples": len(sample_infos),
        "successful": len(successful),
        "failed": len(failed),
        "total_radar_points": total_radar_points,
        "avg_radar_points": total_radar_points / max(len(successful), 1),
        "avg_pillars": total_pillars / max(len(successful), 1),
        "config": config,
    }

    if failed:
        stats["errors"] = [{"token": r["sample_token"], "error": r["error"]} for r in failed]

    # Save metadata
    metadata = {
        "split": split,
        "num_samples": len(successful),
        "file_list": [r["output_path"] for r in successful],
        "statistics": stats,
    }

    metadata_path = os.path.join(split_output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Completed: {len(successful)} successful, {len(failed)} failed")
    print(f"  Avg radar points/sample: {stats['avg_radar_points']:.1f}")
    print(f"  Avg pillars/sample: {stats['avg_pillars']:.1f}")
    print(f"  Metadata saved to: {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Prepare radar occupancy data: convert radar point clouds to pillars "
                    "and generate occupancy ground truth from LiDAR."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="nuScenes data root directory containing radar/ and lidar/ subdirs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional, uses defaults if not provided)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated list of splits to process (default: train,val)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--max_pillars",
        type=int,
        default=10000,
        help="Maximum number of pillars (default: 10000)",
    )
    parser.add_argument(
        "--max_points_per_pillar",
        type=int,
        default=20,
        help="Maximum points per pillar (default: 20)",
    )

    args = parser.parse_args()

    # Build configuration
    config = {
        "grid_size": [256, 256],
        "cell_size": [0.4, 0.4],
        "x_range": [-51.2, 51.2],
        "y_range": [-51.2, 51.2],
        "max_pillars": args.max_pillars,
        "max_points_per_pillar": args.max_points_per_pillar,
        "feature_dim": 9,
        "occupancy_threshold": 1,
        "free_threshold": 1,
    }

    # Load YAML config if provided (overrides defaults)
    if args.config is not None:
        try:
            import yaml
            with open(args.config, "r") as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                config.update(yaml_config)
        except ImportError:
            print("[WARNING] PyYAML not installed. Ignoring --config argument.")
        except Exception as e:
            print(f"[WARNING] Failed to load config file: {e}. Using defaults.")

    # Apply CLI overrides (they take precedence over YAML)
    config["max_pillars"] = args.max_pillars
    config["max_points_per_pillar"] = args.max_points_per_pillar

    print("=" * 60)
    print("Radar Occupancy Data Preparation")
    print("=" * 60)
    print(f"Data directory:  {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Splits:          {args.splits}")
    print(f"Workers:         {args.num_workers}")
    print(f"Grid size:       {config['grid_size']}")
    print(f"Cell size:       {config['cell_size']} m")
    print(f"X range:         {config['x_range']} m")
    print(f"Y range:         {config['y_range']} m")
    print(f"Max pillars:     {config['max_pillars']}")
    print(f"Max pts/pillar:  {config['max_points_per_pillar']}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each split
    splits = [s.strip() for s in args.splits.split(",")]
    all_metadata = {}

    for split in splits:
        print(f"\n--- Split: {split} ---")
        metadata = process_split(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config,
            split=split,
            num_workers=args.num_workers,
        )
        if metadata:
            all_metadata[split] = metadata

    # Save global metadata
    global_meta_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(global_meta_path, "w") as f:
        json.dump(
            {
                "splits": list(all_metadata.keys()),
                "config": config,
                "data_dir": args.data_dir,
            },
            f,
            indent=2,
        )

    print(f"\nDone. Global metadata saved to: {global_meta_path}")


if __name__ == "__main__":
    main()
