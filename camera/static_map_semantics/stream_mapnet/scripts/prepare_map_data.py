#!/usr/bin/env python3
"""
StreamMapNet - Vectorized Map Ground Truth Generation

Generates ground truth vectorized map annotations for StreamMapNet training
from nuScenes dataset and map expansion pack.

For each sample in nuScenes, this script:
  1. Extracts map elements (lane dividers, road boundaries, pedestrian crossings)
     within the BEV perception range
  2. Discretizes polylines and resamples to K evenly-spaced points
  3. Transforms coordinates from global frame to ego-vehicle frame
  4. Saves per-sample pickle files for efficient training data loading

Usage:
    python prepare_map_data.py --dataroot ./data/nuscenes --version v1.0-trainval
    python prepare_map_data.py --dataroot ./data/nuscenes --version v1.0-mini --bev-range 60
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap
    from nuscenes.utils.splits import create_splits_scenes
    from pyquaternion import Quaternion
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install dependencies:")
    print("  pip install nuscenes-devkit pyquaternion numpy tqdm")
    sys.exit(1)


# =============================================================================
# Constants
# =============================================================================

# Map element categories for StreamMapNet
MAP_CLASSES = ["lane_divider", "road_boundary", "pedestrian_crossing"]

# Number of points to resample each polyline to
NUM_POINTS_PER_ELEMENT = 20

# Default BEV perception range in meters (front, back, left, right)
DEFAULT_BEV_RANGE = 60.0  # meters from ego in each direction

# nuScenes location names
LOCATIONS = [
    "singapore-onenorth",
    "singapore-hollandvillage",
    "singapore-queenstown",
    "boston-seaport",
]


# =============================================================================
# Geometry Utilities
# =============================================================================


def resample_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    """
    Resample a polyline to have exactly num_points evenly-spaced points.

    Uses linear interpolation along the cumulative arc length.

    Args:
        polyline: (N, 2) array of 2D points
        num_points: Number of output points

    Returns:
        (num_points, 2) array of resampled points
    """
    if len(polyline) < 2:
        # Degenerate case: repeat the single point
        return np.repeat(polyline[:1], num_points, axis=0)

    # Compute cumulative arc length
    diffs = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length < 1e-6:
        # Degenerate case: all points are the same
        return np.repeat(polyline[:1], num_points, axis=0)

    # Generate evenly-spaced parameter values
    target_lengths = np.linspace(0, total_length, num_points)

    # Interpolate x and y separately
    resampled = np.zeros((num_points, 2))
    resampled[:, 0] = np.interp(target_lengths, cumulative_length, polyline[:, 0])
    resampled[:, 1] = np.interp(target_lengths, cumulative_length, polyline[:, 1])

    return resampled


def global_to_ego(
    points: np.ndarray,
    ego_translation: np.ndarray,
    ego_rotation: Quaternion,
) -> np.ndarray:
    """
    Transform points from global coordinates to ego-vehicle coordinates.

    Args:
        points: (N, 2) or (N, 3) array of points in global frame
        ego_translation: (3,) ego position in global frame
        ego_rotation: Quaternion representing ego orientation in global frame

    Returns:
        (N, 2) array of points in ego frame
    """
    # Ensure 3D for rotation
    if points.shape[1] == 2:
        points_3d = np.hstack([points, np.zeros((len(points), 1))])
    else:
        points_3d = points.copy()

    # Translate to ego-centered
    points_3d -= ego_translation

    # Rotate to ego frame (inverse rotation)
    rot_matrix = ego_rotation.rotation_matrix
    points_ego = (rot_matrix.T @ points_3d.T).T

    return points_ego[:, :2]


def is_in_bev_range(
    points: np.ndarray,
    bev_range: float,
) -> bool:
    """
    Check if any point of a polyline falls within the BEV perception range.

    Args:
        points: (N, 2) array in ego coordinates
        bev_range: Maximum distance from ego in any direction

    Returns:
        True if at least one point is within range
    """
    return np.any(
        (np.abs(points[:, 0]) <= bev_range) & (np.abs(points[:, 1]) <= bev_range)
    )


def clip_polyline_to_bev(
    points: np.ndarray,
    bev_range: float,
) -> Optional[np.ndarray]:
    """
    Clip a polyline to the BEV range, keeping segments that pass through.

    Simple approach: keep all points within range and adjacent boundary points.

    Args:
        points: (N, 2) array in ego coordinates
        bev_range: Maximum distance from ego in any direction

    Returns:
        Clipped polyline or None if entirely outside range
    """
    in_range = (np.abs(points[:, 0]) <= bev_range) & (
        np.abs(points[:, 1]) <= bev_range
    )

    if not np.any(in_range):
        return None

    # Find contiguous segments within range (with one-point margin)
    extended = np.zeros(len(in_range) + 2, dtype=bool)
    extended[1:-1] = in_range
    # Include neighbors of in-range points for smoother clipping
    extended[:-2] |= in_range
    extended[2:] |= in_range
    mask = extended[1:-1]

    # Get the longest contiguous segment
    clipped = points[mask]
    if len(clipped) < 2:
        return None

    # Final clip to exact BEV bounds
    clipped[:, 0] = np.clip(clipped[:, 0], -bev_range, bev_range)
    clipped[:, 1] = np.clip(clipped[:, 1], -bev_range, bev_range)

    return clipped


# =============================================================================
# Map Element Extraction
# =============================================================================


def extract_lane_dividers(
    nusc_map: NuScenesMap,
    patch: Tuple[float, float, float, float],
    ego_translation: np.ndarray,
    ego_rotation: Quaternion,
    bev_range: float,
    num_points: int,
) -> List[np.ndarray]:
    """
    Extract lane divider polylines within the given patch.

    Args:
        nusc_map: NuScenesMap instance
        patch: (x_min, y_min, x_max, y_max) in global coordinates
        ego_translation: Ego position in global frame
        ego_rotation: Ego orientation as quaternion
        bev_range: BEV perception range in meters
        num_points: Number of points per resampled polyline

    Returns:
        List of (num_points, 2) arrays in ego coordinates
    """
    records = nusc_map.get_records_in_patch(patch, ["lane_divider"], mode="intersect")
    lane_divider_tokens = records.get("lane_divider", [])

    polylines = []
    for token in lane_divider_tokens:
        record = nusc_map.get("lane_divider", token)
        line_token = record["line_token"]
        line = nusc_map.extract_line(line_token)

        if line.is_empty:
            continue

        # Get coordinates
        coords = np.array(line.coords)  # (N, 2) in global frame

        # Transform to ego
        coords_ego = global_to_ego(coords, ego_translation, ego_rotation)

        # Check if within BEV range
        clipped = clip_polyline_to_bev(coords_ego, bev_range)
        if clipped is not None and len(clipped) >= 2:
            resampled = resample_polyline(clipped, num_points)
            polylines.append(resampled)

    return polylines


def extract_road_boundaries(
    nusc_map: NuScenesMap,
    patch: Tuple[float, float, float, float],
    ego_translation: np.ndarray,
    ego_rotation: Quaternion,
    bev_range: float,
    num_points: int,
) -> List[np.ndarray]:
    """
    Extract road boundary polylines within the given patch.

    Road boundaries define the edges of drivable areas.

    Args:
        nusc_map: NuScenesMap instance
        patch: (x_min, y_min, x_max, y_max) in global coordinates
        ego_translation: Ego position in global frame
        ego_rotation: Ego orientation as quaternion
        bev_range: BEV perception range in meters
        num_points: Number of points per resampled polyline

    Returns:
        List of (num_points, 2) arrays in ego coordinates
    """
    records = nusc_map.get_records_in_patch(
        patch, ["road_segment"], mode="intersect"
    )
    road_segment_tokens = records.get("road_segment", [])

    polylines = []
    for token in road_segment_tokens:
        record = nusc_map.get("road_segment", token)
        polygon_token = record["polygon_token"]

        # Get exterior boundary of road segment
        polygon = nusc_map.extract_polygon(polygon_token)
        if polygon.is_empty:
            continue

        # Extract exterior ring as polyline
        exterior_coords = np.array(polygon.exterior.coords)  # (N, 2)

        # Transform to ego coordinates
        coords_ego = global_to_ego(exterior_coords, ego_translation, ego_rotation)

        # Check BEV range and clip
        clipped = clip_polyline_to_bev(coords_ego, bev_range)
        if clipped is not None and len(clipped) >= 2:
            resampled = resample_polyline(clipped, num_points)
            polylines.append(resampled)

    return polylines


def extract_pedestrian_crossings(
    nusc_map: NuScenesMap,
    patch: Tuple[float, float, float, float],
    ego_translation: np.ndarray,
    ego_rotation: Quaternion,
    bev_range: float,
    num_points: int,
) -> List[np.ndarray]:
    """
    Extract pedestrian crossing boundaries as polylines.

    Each pedestrian crossing polygon is converted to its boundary polyline.

    Args:
        nusc_map: NuScenesMap instance
        patch: (x_min, y_min, x_max, y_max) in global coordinates
        ego_translation: Ego position in global frame
        ego_rotation: Ego orientation as quaternion
        bev_range: BEV perception range in meters
        num_points: Number of points per resampled polyline

    Returns:
        List of (num_points, 2) arrays in ego coordinates
    """
    records = nusc_map.get_records_in_patch(
        patch, ["ped_crossing"], mode="intersect"
    )
    ped_crossing_tokens = records.get("ped_crossing", [])

    polylines = []
    for token in ped_crossing_tokens:
        record = nusc_map.get("ped_crossing", token)
        polygon_token = record["polygon_token"]

        polygon = nusc_map.extract_polygon(polygon_token)
        if polygon.is_empty:
            continue

        # Extract boundary as polyline
        exterior_coords = np.array(polygon.exterior.coords)  # (N, 2)

        # Transform to ego coordinates
        coords_ego = global_to_ego(exterior_coords, ego_translation, ego_rotation)

        # Check BEV range and clip
        clipped = clip_polyline_to_bev(coords_ego, bev_range)
        if clipped is not None and len(clipped) >= 2:
            resampled = resample_polyline(clipped, num_points)
            polylines.append(resampled)

    return polylines


# =============================================================================
# Main Processing
# =============================================================================


def get_ego_pose(nusc: NuScenes, sample_token: str) -> Tuple[np.ndarray, Quaternion]:
    """
    Get ego pose for a given sample.

    Args:
        nusc: NuScenes instance
        sample_token: Sample token

    Returns:
        Tuple of (translation (3,), rotation quaternion)
    """
    sample = nusc.get("sample", sample_token)
    # Use LIDAR_TOP as reference for ego pose
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])

    translation = np.array(ego_pose["translation"])
    rotation = Quaternion(ego_pose["rotation"])

    return translation, rotation


def get_map_for_sample(
    nusc: NuScenes, nusc_maps: Dict[str, NuScenesMap], sample_token: str
) -> NuScenesMap:
    """
    Get the NuScenesMap instance for the location of a given sample.

    Args:
        nusc: NuScenes instance
        nusc_maps: Dictionary mapping location names to NuScenesMap instances
        sample_token: Sample token

    Returns:
        NuScenesMap instance for the sample's location
    """
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])
    location = log["location"]

    return nusc_maps[location]


def process_sample(
    nusc: NuScenes,
    nusc_maps: Dict[str, NuScenesMap],
    sample_token: str,
    bev_range: float,
    num_points: int,
) -> Dict:
    """
    Process a single sample and extract all map elements.

    Args:
        nusc: NuScenes instance
        nusc_maps: Dictionary of NuScenesMap instances
        sample_token: Token of the sample to process
        bev_range: BEV perception range in meters
        num_points: Number of resampled points per element

    Returns:
        Dictionary containing:
            - 'sample_token': str
            - 'ego_translation': (3,) array
            - 'ego_rotation': (4,) quaternion array
            - 'lane_dividers': list of (K, 2) arrays
            - 'road_boundaries': list of (K, 2) arrays
            - 'pedestrian_crossings': list of (K, 2) arrays
    """
    # Get ego pose
    ego_translation, ego_rotation = get_ego_pose(nusc, sample_token)

    # Get appropriate map
    nusc_map = get_map_for_sample(nusc, nusc_maps, sample_token)

    # Define patch in global coordinates (centered on ego)
    patch = (
        ego_translation[0] - bev_range,
        ego_translation[1] - bev_range,
        ego_translation[0] + bev_range,
        ego_translation[1] + bev_range,
    )

    # Extract map elements
    lane_dividers = extract_lane_dividers(
        nusc_map, patch, ego_translation, ego_rotation, bev_range, num_points
    )

    road_boundaries = extract_road_boundaries(
        nusc_map, patch, ego_translation, ego_rotation, bev_range, num_points
    )

    pedestrian_crossings = extract_pedestrian_crossings(
        nusc_map, patch, ego_translation, ego_rotation, bev_range, num_points
    )

    return {
        "sample_token": sample_token,
        "ego_translation": ego_translation,
        "ego_rotation": np.array(ego_rotation.elements),
        "lane_dividers": lane_dividers,
        "road_boundaries": road_boundaries,
        "pedestrian_crossings": pedestrian_crossings,
        "map_classes": MAP_CLASSES,
        "num_points": num_points,
        "bev_range": bev_range,
    }


def get_split_samples(
    nusc: NuScenes, split: str
) -> List[str]:
    """
    Get sample tokens for a given split.

    Args:
        nusc: NuScenes instance
        split: Split name ('train' or 'val')

    Returns:
        List of sample tokens in the split
    """
    splits = create_splits_scenes()
    split_scenes = splits[split]

    sample_tokens = []
    for scene in nusc.scene:
        if scene["name"] in split_scenes:
            # Traverse all samples in this scene
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample_tokens.append(sample_token)
                sample = nusc.get("sample", sample_token)
                sample_token = sample["next"]

    return sample_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Generate vectorized map ground truth for StreamMapNet"
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="Path to nuScenes dataset root directory",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        choices=["v1.0-trainval", "v1.0-mini"],
        help="nuScenes dataset version (default: v1.0-trainval)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for pickle files (default: <dataroot>/map_gt/)",
    )
    parser.add_argument(
        "--bev-range",
        type=float,
        default=DEFAULT_BEV_RANGE,
        help=f"BEV perception range in meters (default: {DEFAULT_BEV_RANGE})",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=NUM_POINTS_PER_ELEMENT,
        help=f"Points per polyline after resampling (default: {NUM_POINTS_PER_ELEMENT})",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="Splits to process (default: train val)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    args = parser.parse_args()

    # Validate dataroot
    dataroot = Path(args.dataroot)
    if not dataroot.exists():
        print(f"Error: dataroot does not exist: {dataroot}")
        sys.exit(1)

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else dataroot / "map_gt"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  StreamMapNet - Vectorized Map Ground Truth Generation")
    print("=" * 70)
    print(f"  Dataset root:   {dataroot}")
    print(f"  Version:        {args.version}")
    print(f"  Output dir:     {output_dir}")
    print(f"  BEV range:      {args.bev_range} m")
    print(f"  Points/element: {args.num_points}")
    print(f"  Splits:         {args.splits}")
    print("=" * 70)
    print()

    # Initialize nuScenes
    print("Loading nuScenes database...")
    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=True)

    # Initialize maps for all locations
    print("Loading nuScenes maps...")
    nusc_maps = {}
    for location in LOCATIONS:
        try:
            nusc_maps[location] = NuScenesMap(
                dataroot=str(dataroot), map_name=location
            )
            print(f"  Loaded map: {location}")
        except Exception as e:
            print(f"  Warning: Could not load map for {location}: {e}")

    if not nusc_maps:
        print("Error: No maps could be loaded. Please ensure map expansion is installed.")
        print("  Expected location: <dataroot>/maps/expansion/")
        sys.exit(1)

    # Process each split
    for split in args.splits:
        print(f"\nProcessing split: {split}")
        print("-" * 40)

        # Determine split name for nuScenes
        if args.version == "v1.0-mini":
            ns_split = f"mini_{split}"
        else:
            ns_split = split

        try:
            sample_tokens = get_split_samples(nusc, ns_split)
        except KeyError:
            print(f"  Warning: Split '{ns_split}' not found, skipping.")
            continue

        print(f"  Found {len(sample_tokens)} samples")

        # Create split output directory
        split_output_dir = output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # Process all samples
        results = []
        failed = 0

        for sample_token in tqdm(
            sample_tokens,
            desc=f"  Extracting map GT ({split})",
            unit="sample",
            ncols=80,
        ):
            try:
                result = process_sample(
                    nusc=nusc,
                    nusc_maps=nusc_maps,
                    sample_token=sample_token,
                    bev_range=args.bev_range,
                    num_points=args.num_points,
                )
                results.append(result)

                # Save individual sample pickle
                sample_file = split_output_dir / f"{sample_token}.pkl"
                with open(sample_file, "wb") as f:
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"\n  Warning: Failed to process {sample_token}: {e}")
                elif failed == 6:
                    print(f"\n  ... suppressing further warnings ...")

        # Save combined split file
        combined_file = output_dir / f"{split}_map_gt.pkl"
        with open(combined_file, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Print statistics
        print(f"\n  Split '{split}' complete:")
        print(f"    Processed: {len(results)} samples")
        print(f"    Failed:    {failed} samples")

        if results:
            total_lanes = sum(len(r["lane_dividers"]) for r in results)
            total_boundaries = sum(len(r["road_boundaries"]) for r in results)
            total_crossings = sum(len(r["pedestrian_crossings"]) for r in results)

            print(f"    Total lane dividers:       {total_lanes}")
            print(f"    Total road boundaries:     {total_boundaries}")
            print(f"    Total ped crossings:       {total_crossings}")
            print(f"    Avg elements/sample:       "
                  f"{(total_lanes + total_boundaries + total_crossings) / len(results):.1f}")

        print(f"    Saved to: {combined_file}")

    print("\n" + "=" * 70)
    print("  Map ground truth generation complete!")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
