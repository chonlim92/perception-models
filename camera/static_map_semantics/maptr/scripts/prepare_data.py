"""
MapTR Data Preparation Script

Extracts vectorized map annotations from nuScenes map expansion pack and prepares
them for training. Each sample gets multi-camera calibration info and vectorized
map elements (polylines) in ego-vehicle coordinates.

Usage:
    python scripts/prepare_data.py \
        --nuscenes_root /data/nuscenes \
        --output_dir data/processed \
        --version v1.0-trainval \
        --num_workers 8
"""

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Geometry Utilities
# ============================================================================

def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quaternion
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]], dtype=np.float64)


def compose_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Compose 4x4 transformation matrix from rotation (3x3) and translation (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid body transformation matrix."""
    T_inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_points_3d(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform 3D points using a 4x4 matrix. Points shape: (N, 3)."""
    N = points.shape[0]
    points_h = np.concatenate([points, np.ones((N, 1))], axis=1)  # (N, 4)
    transformed = (T @ points_h.T).T  # (N, 4)
    return transformed[:, :3]


# ============================================================================
# Polyline Utilities
# ============================================================================

def resample_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Resample a polyline to a fixed number of points using arc-length interpolation.

    Args:
        points: (N, 2) or (N, 3) array of polyline vertices
        num_points: Target number of points

    Returns:
        (num_points, 2) or (num_points, 3) resampled points
    """
    if len(points) < 2:
        return np.tile(points[0], (num_points, 1))

    # Compute cumulative arc length
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_lengths = np.zeros(len(points))
    cumulative_lengths[1:] = np.cumsum(segment_lengths)
    total_length = cumulative_lengths[-1]

    if total_length < 1e-6:
        return np.tile(points[0], (num_points, 1))

    # Generate uniformly spaced parameter values
    target_lengths = np.linspace(0, total_length, num_points)

    # Interpolate
    resampled = np.zeros((num_points, points.shape[1]))
    for dim in range(points.shape[1]):
        resampled[:, dim] = np.interp(target_lengths, cumulative_lengths, points[:, dim])

    return resampled


def clip_polyline_to_range(points: np.ndarray, x_range: Tuple[float, float],
                           y_range: Tuple[float, float]) -> Optional[np.ndarray]:
    """
    Clip polyline to perception range. Returns None if entirely outside.

    Uses a simple approach: keep points within range and reconnect segments.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Check which points are inside
    inside_mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    )

    if not np.any(inside_mask):
        return None

    # Find connected segments of inside points
    # For simplicity, take the longest continuous segment
    segments = []
    current_segment = []

    for i, is_inside in enumerate(inside_mask):
        if is_inside:
            current_segment.append(i)
        else:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
    if current_segment:
        segments.append(current_segment)

    if not segments:
        return None

    # Take the longest segment
    longest = max(segments, key=len)
    clipped = points[longest]

    # Need at least 2 points for a valid polyline
    if len(clipped) < 2:
        return None

    return clipped


# ============================================================================
# nuScenes Data Parsing
# ============================================================================

class NuScenesParser:
    """Parser for nuScenes database tables."""

    def __init__(self, dataroot: str, version: str = "v1.0-trainval"):
        self.dataroot = dataroot
        self.version = version
        self.table_dir = os.path.join(dataroot, version)

        # Load database tables
        self.tables = {}
        self._load_tables()

        # Build indexes
        self._build_indexes()

    def _load_table(self, name: str) -> List[Dict]:
        """Load a single database table."""
        path = os.path.join(self.table_dir, f"{name}.json")
        if not os.path.exists(path):
            print(f"  Warning: table {name}.json not found at {path}")
            return []
        with open(path, "r") as f:
            return json.load(f)

    def _load_tables(self):
        """Load all required database tables."""
        table_names = [
            "scene", "sample", "sample_data", "ego_pose",
            "calibrated_sensor", "sensor", "log", "map"
        ]
        for name in table_names:
            self.tables[name] = self._load_table(name)
            print(f"  Loaded {name}: {len(self.tables[name])} records")

    def _build_indexes(self):
        """Build token-to-record indexes for fast lookup."""
        self.idx = {}
        for table_name, records in self.tables.items():
            self.idx[table_name] = {}
            for record in records:
                token = record["token"]
                self.idx[table_name][token] = record

        # Build sensor name index
        self.sensor_by_channel = {}
        for sensor in self.tables.get("sensor", []):
            self.sensor_by_channel[sensor["channel"]] = sensor

        # Build log-to-map index
        self.log_to_map = {}
        for map_record in self.tables.get("map", []):
            for log_token in map_record.get("log_tokens", []):
                self.log_to_map[log_token] = map_record

    def get(self, table: str, token: str) -> Dict:
        """Get a record by table name and token."""
        return self.idx[table][token]

    def get_sample_data_for_sample(self, sample_token: str, channel: str) -> Optional[Dict]:
        """Get sample_data record for a specific sample and sensor channel."""
        sample = self.get("sample", sample_token)
        data_token = sample.get("data", {}).get(channel)
        if data_token is None:
            return None
        return self.get("sample_data", data_token)

    def get_ego_pose(self, sample_data: Dict) -> Dict:
        """Get ego pose for a sample_data record."""
        return self.get("ego_pose", sample_data["ego_pose_token"])

    def get_calibration(self, sample_data: Dict) -> Dict:
        """Get calibrated sensor for a sample_data record."""
        return self.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

    def get_map_for_scene(self, scene_token: str) -> Optional[Dict]:
        """Get map record for a scene."""
        scene = self.get("scene", scene_token)
        log_token = scene["log_token"]
        return self.log_to_map.get(log_token)


# ============================================================================
# Map Element Extraction
# ============================================================================

class MapExtractor:
    """Extracts vectorized map elements from nuScenes map expansion."""

    # Map element categories
    CATEGORIES = {
        "ped_crossing": 0,
        "divider": 1,
        "boundary": 2,
    }

    # Map layers to query for each category
    LAYER_MAPPING = {
        "ped_crossing": ["ped_crossing"],
        "divider": ["lane_divider", "road_divider"],
        "boundary": ["road_segment"],  # Extract boundary from road segments
    }

    def __init__(self, dataroot: str, perception_range: Tuple[float, float, float, float] = (-30, -15, 30, 15),
                 num_points: int = 20):
        """
        Args:
            dataroot: Path to nuScenes root
            perception_range: (x_min, y_min, x_max, y_max) in meters
            num_points: Number of points to sample per polyline
        """
        self.dataroot = dataroot
        self.x_min, self.y_min, self.x_max, self.y_max = perception_range
        self.num_points = num_points

        # Load map expansion data
        self.map_data = {}
        self._load_map_expansion()

    def _load_map_expansion(self):
        """Load map expansion JSON files for all locations."""
        maps_dir = os.path.join(self.dataroot, "maps")
        expansion_dir = os.path.join(maps_dir, "expansion")

        # Try expansion subdirectory first, then maps root
        search_dirs = [expansion_dir, maps_dir]

        locations = [
            "singapore-onenorth",
            "singapore-hollandvillage",
            "singapore-queenstown",
            "boston-seaport",
        ]

        for location in locations:
            for search_dir in search_dirs:
                filepath = os.path.join(search_dir, f"{location}.json")
                if os.path.exists(filepath):
                    print(f"  Loading map: {location}")
                    with open(filepath, "r") as f:
                        self.map_data[location] = json.load(f)
                    break
            else:
                print(f"  Warning: Map file not found for {location}")

    def get_location_from_log(self, log_token: str, nusc: NuScenesParser) -> Optional[str]:
        """Get location name from log token."""
        log = nusc.get("log", log_token)
        location = log.get("location", "")
        # Normalize location name
        location_lower = location.lower().replace(" ", "-")
        for key in self.map_data.keys():
            if key in location_lower or location_lower in key:
                return key
        return None

    def extract_elements(self, location: str, ego_pose_matrix: np.ndarray) -> List[Dict]:
        """
        Extract map elements within perception range around ego vehicle.

        Args:
            location: Map location name
            ego_pose_matrix: 4x4 ego-to-global transformation

        Returns:
            List of dict with keys: 'category', 'label', 'points'
        """
        if location not in self.map_data:
            return []

        map_json = self.map_data[location]
        ego_inv = invert_transform(ego_pose_matrix)
        elements = []

        # Extract pedestrian crossings
        for pc in map_json.get("ped_crossing", []):
            polygon = np.array(pc.get("polygon", []))
            if len(polygon) < 3:
                continue
            # Use the polygon boundary as the polyline (close it)
            polyline_3d = np.column_stack([polygon[:, :2], np.zeros(len(polygon))])
            ego_points = transform_points_3d(polyline_3d, ego_inv)[:, :2]

            clipped = clip_polyline_to_range(
                ego_points, (self.x_min, self.x_max), (self.y_min, self.y_max)
            )
            if clipped is not None and len(clipped) >= 2:
                resampled = resample_polyline(clipped, self.num_points)
                elements.append({
                    "category": "ped_crossing",
                    "label": self.CATEGORIES["ped_crossing"],
                    "points": resampled.astype(np.float32),
                })

        # Extract lane dividers
        for divider in map_json.get("lane_divider", []):
            line = np.array(divider.get("line", divider.get("geom", {}).get("coordinates", [])))
            if len(line) < 2:
                continue
            if line.shape[1] == 2:
                line_3d = np.column_stack([line, np.zeros(len(line))])
            else:
                line_3d = line[:, :3]

            ego_points = transform_points_3d(line_3d, ego_inv)[:, :2]
            clipped = clip_polyline_to_range(
                ego_points, (self.x_min, self.x_max), (self.y_min, self.y_max)
            )
            if clipped is not None and len(clipped) >= 2:
                resampled = resample_polyline(clipped, self.num_points)
                elements.append({
                    "category": "divider",
                    "label": self.CATEGORIES["divider"],
                    "points": resampled.astype(np.float32),
                })

        # Extract road dividers
        for divider in map_json.get("road_divider", []):
            line = np.array(divider.get("line", divider.get("geom", {}).get("coordinates", [])))
            if len(line) < 2:
                continue
            if line.shape[1] == 2:
                line_3d = np.column_stack([line, np.zeros(len(line))])
            else:
                line_3d = line[:, :3]

            ego_points = transform_points_3d(line_3d, ego_inv)[:, :2]
            clipped = clip_polyline_to_range(
                ego_points, (self.x_min, self.x_max), (self.y_min, self.y_max)
            )
            if clipped is not None and len(clipped) >= 2:
                resampled = resample_polyline(clipped, self.num_points)
                elements.append({
                    "category": "divider",
                    "label": self.CATEGORIES["divider"],
                    "points": resampled.astype(np.float32),
                })

        # Extract road boundaries from road segments
        for segment in map_json.get("road_segment", []):
            polygon = np.array(segment.get("polygon", segment.get("exterior_coords", [])))
            if len(polygon) < 3:
                continue
            # Extract boundary as the polygon perimeter
            if polygon.shape[1] == 2:
                polygon_3d = np.column_stack([polygon, np.zeros(len(polygon))])
            else:
                polygon_3d = polygon[:, :3]

            ego_points = transform_points_3d(polygon_3d, ego_inv)[:, :2]

            # Split polygon boundary into segments based on proximity
            boundary_segments = self._split_boundary(ego_points)

            for seg in boundary_segments:
                clipped = clip_polyline_to_range(
                    seg, (self.x_min, self.x_max), (self.y_min, self.y_max)
                )
                if clipped is not None and len(clipped) >= 2:
                    resampled = resample_polyline(clipped, self.num_points)
                    elements.append({
                        "category": "boundary",
                        "label": self.CATEGORIES["boundary"],
                        "points": resampled.astype(np.float32),
                    })

        return elements

    def _split_boundary(self, polygon_pts: np.ndarray, max_segment_length: float = 30.0) -> List[np.ndarray]:
        """Split a polygon boundary into manageable polyline segments."""
        segments = []
        total_pts = len(polygon_pts)

        if total_pts < 4:
            return [polygon_pts]

        # Compute distances between consecutive points
        dists = np.linalg.norm(np.diff(polygon_pts, axis=0), axis=1)
        cumulative = np.cumsum(dists)
        total_length = cumulative[-1] if len(cumulative) > 0 else 0

        if total_length < 1e-3:
            return []

        # Split into segments of roughly max_segment_length
        num_segments = max(1, int(np.ceil(total_length / max_segment_length)))
        pts_per_segment = max(3, total_pts // num_segments)

        for i in range(0, total_pts - 1, pts_per_segment):
            end_idx = min(i + pts_per_segment + 1, total_pts)
            seg = polygon_pts[i:end_idx]
            if len(seg) >= 2:
                segments.append(seg)

        return segments


# ============================================================================
# Sample Processing
# ============================================================================

CAMERA_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def process_sample(args: Tuple) -> Optional[Dict]:
    """
    Process a single nuScenes sample.

    Args:
        Tuple of (sample_token, nusc_data, map_extractor_config)

    Returns:
        Dict with sample annotation data or None on failure
    """
    sample_token, nusc_root, version, perception_range, num_points = args

    try:
        # Reload nuScenes parser in subprocess (can't pickle complex objects)
        nusc = NuScenesParser(nusc_root, version)
        extractor = MapExtractor(nusc_root, perception_range, num_points)

        sample = nusc.get("sample", sample_token)
        scene_token = sample["scene_token"]
        scene = nusc.get("scene", scene_token)
        log_token = scene["log_token"]

        # Get location for map lookup
        location = extractor.get_location_from_log(log_token, nusc)
        if location is None:
            return None

        # Get ego pose from front camera's sample data
        front_sd = nusc.get_sample_data_for_sample(sample_token, "CAM_FRONT")
        if front_sd is None:
            return None

        ego_pose_record = nusc.get_ego_pose(front_sd)
        ego_rotation = quaternion_to_rotation_matrix(
            np.array(ego_pose_record["rotation"])
        )
        ego_translation = np.array(ego_pose_record["translation"])
        ego_pose_matrix = compose_transform(ego_rotation, ego_translation)

        # Get camera calibration for all 6 cameras
        camera_info = {}
        for channel in CAMERA_CHANNELS:
            sd = nusc.get_sample_data_for_sample(sample_token, channel)
            if sd is None:
                return None

            # Get calibration
            calib = nusc.get_calibration(sd)
            intrinsic = np.array(calib["camera_intrinsic"])  # 3x3

            # Sensor-to-ego transform
            sensor_rotation = quaternion_to_rotation_matrix(
                np.array(calib["rotation"])
            )
            sensor_translation = np.array(calib["translation"])
            sensor_to_ego = compose_transform(sensor_rotation, sensor_translation)

            # Sensor ego pose
            sd_ego = nusc.get_ego_pose(sd)
            sd_ego_rot = quaternion_to_rotation_matrix(np.array(sd_ego["rotation"]))
            sd_ego_trans = np.array(sd_ego["translation"])
            sd_ego_to_global = compose_transform(sd_ego_rot, sd_ego_trans)

            # Full extrinsic: sensor to ego (we use ego-frame as reference)
            camera_info[channel] = {
                "intrinsic": intrinsic.astype(np.float32),
                "extrinsic": sensor_to_ego.astype(np.float32),
                "image_path": sd["filename"],
                "width": sd.get("width", 1600),
                "height": sd.get("height", 900),
            }

        # Extract map elements
        map_elements = extractor.extract_elements(location, ego_pose_matrix)

        # Build annotation
        annotation = {
            "token": sample_token,
            "scene_token": scene_token,
            "timestamp": sample.get("timestamp", 0),
            "location": location,
            "ego_pose": ego_pose_matrix.astype(np.float32),
            "cameras": camera_info,
            "map_elements": map_elements,
            "num_elements": len(map_elements),
        }

        return annotation

    except Exception as e:
        print(f"  Error processing sample {sample_token}: {e}")
        return None


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def get_split_scenes(nusc: NuScenesParser, version: str) -> Tuple[List[str], List[str]]:
    """Get train/val scene splits."""
    # nuScenes official split
    # For v1.0-mini, use simple even/odd split
    all_scenes = nusc.tables["scene"]

    if "mini" in version:
        train_scenes = [s["token"] for i, s in enumerate(all_scenes) if i % 2 == 0]
        val_scenes = [s["token"] for i, s in enumerate(all_scenes) if i % 2 == 1]
    else:
        # Official split: scenes 0-699 for train, 700-849 for val
        # Use scene name numbering if available
        train_scenes = []
        val_scenes = []
        for scene in all_scenes:
            name = scene.get("name", "")
            # Extract scene number from name like "scene-0001"
            try:
                scene_num = int(name.split("-")[-1])
                if scene_num < 700:
                    train_scenes.append(scene["token"])
                else:
                    val_scenes.append(scene["token"])
            except (ValueError, IndexError):
                train_scenes.append(scene["token"])

    return train_scenes, val_scenes


def get_samples_for_scenes(nusc: NuScenesParser, scene_tokens: List[str]) -> List[str]:
    """Get all sample tokens for given scenes."""
    sample_tokens = []
    for scene_token in scene_tokens:
        scene = nusc.get("scene", scene_token)
        sample_token = scene.get("first_sample_token", "")
        while sample_token:
            sample_tokens.append(sample_token)
            sample_record = nusc.get("sample", sample_token)
            sample_token = sample_record.get("next", "")
    return sample_tokens


def process_split(nusc: NuScenesParser, sample_tokens: List[str],
                  nusc_root: str, version: str,
                  perception_range: Tuple[float, float, float, float],
                  num_points: int, num_workers: int,
                  split_name: str) -> List[Dict]:
    """Process all samples in a split."""
    print(f"\nProcessing {split_name} split: {len(sample_tokens)} samples")

    # Prepare arguments for parallel processing
    args_list = [
        (token, nusc_root, version, perception_range, num_points)
        for token in sample_tokens
    ]

    annotations = []

    if num_workers <= 1:
        # Sequential processing
        for i, args in enumerate(args_list):
            result = process_sample(args)
            if result is not None:
                annotations.append(result)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(args_list)} samples "
                      f"({len(annotations)} valid)")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_sample, args): i
                       for i, args in enumerate(args_list)}

            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result is not None:
                    annotations.append(result)
                if completed % 100 == 0:
                    print(f"  Processed {completed}/{len(args_list)} samples "
                          f"({len(annotations)} valid)")

    print(f"  {split_name}: {len(annotations)} valid annotations from "
          f"{len(sample_tokens)} samples")
    return annotations


def compute_statistics(annotations: List[Dict]) -> Dict:
    """Compute dataset statistics."""
    stats = {
        "num_samples": len(annotations),
        "elements_per_sample": [],
        "category_counts": {"ped_crossing": 0, "divider": 0, "boundary": 0},
        "points_per_element": [],
    }

    for ann in annotations:
        num_elements = ann["num_elements"]
        stats["elements_per_sample"].append(num_elements)
        for elem in ann["map_elements"]:
            stats["category_counts"][elem["category"]] += 1
            stats["points_per_element"].append(len(elem["points"]))

    eps = stats["elements_per_sample"]
    stats["avg_elements_per_sample"] = np.mean(eps) if eps else 0
    stats["max_elements_per_sample"] = np.max(eps) if eps else 0
    stats["min_elements_per_sample"] = np.min(eps) if eps else 0

    return stats


# ============================================================================
# Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare nuScenes vectorized map data for MapTR training"
    )
    parser.add_argument(
        "--nuscenes_root", type=str, required=True,
        help="Path to nuScenes dataset root directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed",
        help="Output directory for processed annotations"
    )
    parser.add_argument(
        "--version", type=str, default="v1.0-trainval",
        choices=["v1.0-trainval", "v1.0-mini", "v1.0-test"],
        help="nuScenes dataset version"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of parallel workers for processing"
    )
    parser.add_argument(
        "--num_points", type=int, default=20,
        help="Number of points to sample per polyline element"
    )
    parser.add_argument(
        "--x_range", type=float, nargs=2, default=[-30.0, 30.0],
        help="Perception range in x direction (meters)"
    )
    parser.add_argument(
        "--y_range", type=float, nargs=2, default=[-15.0, 15.0],
        help="Perception range in y direction (meters)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("MapTR Data Preparation")
    print("=" * 60)
    print(f"  nuScenes root: {args.nuscenes_root}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Version:       {args.version}")
    print(f"  Workers:       {args.num_workers}")
    print(f"  Points/elem:   {args.num_points}")
    print(f"  X range:       {args.x_range}")
    print(f"  Y range:       {args.y_range}")
    print()

    # Validate input
    if not os.path.exists(args.nuscenes_root):
        print(f"Error: nuScenes root not found: {args.nuscenes_root}")
        sys.exit(1)

    version_dir = os.path.join(args.nuscenes_root, args.version)
    if not os.path.exists(version_dir):
        print(f"Error: Version directory not found: {version_dir}")
        print(f"  Expected tables like sample.json, scene.json, etc.")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load nuScenes
    print("Loading nuScenes database...")
    nusc = NuScenesParser(args.nuscenes_root, args.version)

    # Get splits
    perception_range = (args.x_range[0], args.y_range[0], args.x_range[1], args.y_range[1])

    train_scenes, val_scenes = get_split_scenes(nusc, args.version)
    print(f"\nSplit: {len(train_scenes)} train scenes, {len(val_scenes)} val scenes")

    train_samples = get_samples_for_scenes(nusc, train_scenes)
    val_samples = get_samples_for_scenes(nusc, val_scenes)
    print(f"Samples: {len(train_samples)} train, {len(val_samples)} val")

    # Process train split
    train_annotations = process_split(
        nusc, train_samples, args.nuscenes_root, args.version,
        perception_range, args.num_points, args.num_workers, "train"
    )

    # Process val split
    val_annotations = process_split(
        nusc, val_samples, args.nuscenes_root, args.version,
        perception_range, args.num_points, args.num_workers, "val"
    )

    # Save processed annotations
    print("\nSaving processed annotations...")

    train_path = os.path.join(args.output_dir, "maptr_train.pkl")
    with open(train_path, "wb") as f:
        pickle.dump(train_annotations, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved train: {train_path} ({len(train_annotations)} samples)")

    val_path = os.path.join(args.output_dir, "maptr_val.pkl")
    with open(val_path, "wb") as f:
        pickle.dump(val_annotations, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved val: {val_path} ({len(val_annotations)} samples)")

    # Save metadata
    metadata = {
        "version": args.version,
        "num_points": args.num_points,
        "perception_range": {
            "x_min": args.x_range[0], "x_max": args.x_range[1],
            "y_min": args.y_range[0], "y_max": args.y_range[1],
        },
        "categories": MapExtractor.CATEGORIES,
        "camera_channels": CAMERA_CHANNELS,
        "train_samples": len(train_annotations),
        "val_samples": len(val_annotations),
    }

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    # Compute and print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    for split_name, annotations in [("Train", train_annotations), ("Val", val_annotations)]:
        stats = compute_statistics(annotations)
        print(f"\n  {split_name}:")
        print(f"    Samples: {stats['num_samples']}")
        print(f"    Avg elements/sample: {stats['avg_elements_per_sample']:.1f}")
        print(f"    Max elements/sample: {stats['max_elements_per_sample']}")
        print(f"    Category counts:")
        for cat, count in stats["category_counts"].items():
            print(f"      {cat}: {count}")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
