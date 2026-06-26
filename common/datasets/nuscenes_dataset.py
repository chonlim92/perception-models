"""nuScenes dataset loader for autonomous driving perception models.

Provides a fully-featured data loader that handles:
- 6 surround cameras (FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT)
- LiDAR point clouds from .pcd.bin files
- Radar point clouds (5 radars)
- 3D bounding box annotations (location, dimensions, rotation as quaternion)
- HD map annotations (lane dividers, road boundaries, pedestrian crossings)
- Ego pose and calibration matrices (intrinsic + extrinsic)
- Temporal sequences (past N frames for video-based models)
- Both PyTorch (torch.utils.data.Dataset) and TensorFlow (tf.data.Dataset) interfaces

Usage
-----
PyTorch::

    from common.datasets.nuscenes_dataset import NuScenesDataset

    dataset = NuScenesDataset(
        dataroot="/data/nuscenes",
        version="v1.0-trainval",
        split="train",
        cameras=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                 "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"],
        load_lidar=True,
        load_radar=True,
        load_map=True,
        n_history_frames=3,
    )
    sample = dataset[0]

TensorFlow::

    tf_dataset = NuScenesDataset.as_tf_dataset(
        dataroot="/data/nuscenes",
        version="v1.0-trainval",
        split="train",
    )
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    TorchDataset = object  # type: ignore[assignment,misc]

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.map_expansion.map_api import NuScenesMap

    _NUSCENES_AVAILABLE = True
except ImportError:
    _NUSCENES_AVAILABLE = False

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from common.registry import DATASETS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMERA_CHANNELS: List[str] = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

RADAR_CHANNELS: List[str] = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]

LIDAR_CHANNEL: str = "LIDAR_TOP"

# nuScenes detection classes
DETECTION_CLASSES: List[str] = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.

    Parameters
    ----------
    quaternion : np.ndarray
        Shape (4,) quaternion in [w, x, y, z] order.

    Returns
    -------
    np.ndarray
        Shape (3, 3) rotation matrix.
    """
    w, x, y, z = quaternion
    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=np.float64,
    )
    return R


def _build_transformation_matrix(
    translation: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    """Build a 4x4 homogeneous transformation matrix from translation and quaternion.

    Parameters
    ----------
    translation : np.ndarray
        Shape (3,) translation vector.
    rotation : np.ndarray
        Shape (4,) quaternion in [w, x, y, z] order.

    Returns
    -------
    np.ndarray
        Shape (4, 4) homogeneous transformation matrix.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quaternion_to_rotation_matrix(rotation)
    T[:3, 3] = translation
    return T


def _load_image(filepath: str) -> np.ndarray:
    """Load an image from disk and return as a numpy array (H, W, 3) in RGB.

    Parameters
    ----------
    filepath : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image array with shape (H, W, 3), dtype uint8, RGB channel order.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    RuntimeError
        If PIL is not available.
    """
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for image loading. Install with: pip install Pillow")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")
    img = Image.open(filepath).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Main Dataset Class
# ---------------------------------------------------------------------------


@DATASETS.register("nuscenes")
class NuScenesDataset(TorchDataset):
    """nuScenes dataset for multi-modal 3D perception.

    This dataset supports loading synchronized multi-sensor data from the
    nuScenes dataset, including cameras, LiDAR, radar, annotations, ego pose,
    calibration, and HD map information.

    Parameters
    ----------
    dataroot : str
        Path to the nuScenes dataset root directory.
    version : str
        Dataset version (e.g., ``"v1.0-trainval"``, ``"v1.0-mini"``).
    split : str
        Data split: ``"train"``, ``"val"``, or ``"test"``.
    cameras : list of str, optional
        Camera channels to load. Defaults to all 6 surround cameras.
    load_lidar : bool
        Whether to load LiDAR point clouds. Default ``True``.
    load_radar : bool
        Whether to load radar point clouds. Default ``False``.
    load_map : bool
        Whether to load HD map annotations. Default ``False``.
    n_history_frames : int
        Number of past frames to include for temporal sequences. Default ``0`` (current only).
    max_lidar_points : int
        Maximum number of LiDAR points to keep. Points are randomly subsampled if exceeded.
        Default ``-1`` (no limit).
    image_size : tuple of int, optional
        If provided, resize images to (height, width).
    transform : callable, optional
        A function/transform applied to the full sample dictionary after loading.
    point_cloud_range : list of float, optional
        Filter point cloud to [x_min, y_min, z_min, x_max, y_max, z_max].
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-trainval",
        split: str = "train",
        cameras: Optional[List[str]] = None,
        load_lidar: bool = True,
        load_radar: bool = False,
        load_map: bool = False,
        n_history_frames: int = 0,
        max_lidar_points: int = -1,
        image_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        point_cloud_range: Optional[List[float]] = None,
    ) -> None:
        if not _NUSCENES_AVAILABLE:
            raise ImportError(
                "nuscenes-devkit is required for NuScenesDataset. "
                "Install with: pip install nuscenes-devkit"
            )
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for NuScenesDataset. "
                "Install with: pip install torch"
            )

        super().__init__()

        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.cameras = cameras if cameras is not None else list(CAMERA_CHANNELS)
        self.load_lidar = load_lidar
        self.load_radar = load_radar
        self.load_map = load_map
        self.n_history_frames = n_history_frames
        self.max_lidar_points = max_lidar_points
        self.image_size = image_size
        self.transform = transform
        self.point_cloud_range = point_cloud_range

        # Validate cameras
        for cam in self.cameras:
            if cam not in CAMERA_CHANNELS:
                raise ValueError(
                    f"Unknown camera channel: {cam!r}. "
                    f"Valid channels: {CAMERA_CHANNELS}"
                )

        # Initialize nuScenes
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        # Get scenes for this split
        split_scenes = self._get_split_scenes()
        self.scene_names = split_scenes

        # Collect all sample tokens for the split
        self.sample_tokens: List[str] = []
        for scene in self.nusc.scene:
            if scene["name"] in self.scene_names:
                sample_token = scene["first_sample_token"]
                while sample_token:
                    self.sample_tokens.append(sample_token)
                    sample = self.nusc.get("sample", sample_token)
                    sample_token = sample["next"]

        # Initialize map API if needed
        self._maps: Dict[str, Any] = {}
        if self.load_map:
            map_locations = set()
            for scene in self.nusc.scene:
                if scene["name"] in self.scene_names:
                    log = self.nusc.get("log", scene["log_token"])
                    map_locations.add(log["location"])
            for location in map_locations:
                try:
                    self._maps[location] = NuScenesMap(
                        dataroot=dataroot, map_name=location
                    )
                except Exception as e:
                    warnings.warn(
                        f"Failed to load map for location {location!r}: {e}"
                    )

    def _get_split_scenes(self) -> List[str]:
        """Get scene names for the current split.

        Returns
        -------
        list of str
            Scene names belonging to the requested split.
        """
        splits = create_splits_scenes()

        # Handle split name mapping
        split_key = self.split
        if self.version == "v1.0-mini":
            if self.split in ("train", "val"):
                split_key = f"mini_{self.split}"
            else:
                split_key = "mini_train"
        elif self.version == "v1.0-trainval":
            split_key = self.split  # "train" or "val"
        elif self.version == "v1.0-test":
            split_key = "test"

        if split_key not in splits:
            available = list(splits.keys())
            raise ValueError(
                f"Split {split_key!r} not found. Available splits: {available}"
            )

        return splits[split_key]

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        return len(self.sample_tokens)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Load and return a single sample with all requested modalities.

        Parameters
        ----------
        index : int
            Index of the sample to load.

        Returns
        -------
        dict
            A dictionary containing all requested data. Keys include:

            - ``"sample_token"`` : str
            - ``"timestamp"`` : int (microseconds)
            - ``"images"`` : dict mapping camera name to (H, W, 3) uint8 arrays
            - ``"lidar_points"`` : (N, 5) float32 array [x, y, z, intensity, ring] or None
            - ``"radar_points"`` : dict mapping radar name to (M, 18) float32 arrays or None
            - ``"annotations"`` : list of annotation dicts
            - ``"ego_pose"`` : (4, 4) float64 transformation matrix
            - ``"calibration"`` : dict mapping sensor to calibration info
            - ``"map_data"`` : dict with map annotations or None
            - ``"history"`` : list of past frame data dicts (if n_history_frames > 0)
        """
        sample_token = self.sample_tokens[index]
        sample = self.nusc.get("sample", sample_token)

        result: Dict[str, Any] = {
            "sample_token": sample_token,
            "timestamp": sample["timestamp"],
        }

        # Load camera images and calibration
        images, calibration = self._load_cameras(sample)
        result["images"] = images
        result["calibration"] = calibration

        # Load LiDAR
        if self.load_lidar:
            lidar_data, lidar_calib = self._load_lidar(sample)
            result["lidar_points"] = lidar_data
            result["calibration"]["LIDAR_TOP"] = lidar_calib
        else:
            result["lidar_points"] = None

        # Load Radar
        if self.load_radar:
            radar_data, radar_calibs = self._load_radar(sample)
            result["radar_points"] = radar_data
            for radar_name, radar_calib in radar_calibs.items():
                result["calibration"][radar_name] = radar_calib
        else:
            result["radar_points"] = None

        # Load ego pose (from LiDAR sample_data as reference)
        ego_pose = self._load_ego_pose(sample)
        result["ego_pose"] = ego_pose

        # Load annotations
        annotations = self._load_annotations(sample)
        result["annotations"] = annotations

        # Load map data
        if self.load_map:
            map_data = self._load_map_data(sample, ego_pose)
            result["map_data"] = map_data
        else:
            result["map_data"] = None

        # Load temporal history
        if self.n_history_frames > 0:
            history = self._load_history(sample, self.n_history_frames)
            result["history"] = history
        else:
            result["history"] = []

        # Apply user transform
        if self.transform is not None:
            result = self.transform(result)

        return result

    # ------------------------------------------------------------------
    # Camera loading
    # ------------------------------------------------------------------

    def _load_cameras(
        self, sample: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """Load camera images and calibration for all requested cameras.

        Parameters
        ----------
        sample : dict
            nuScenes sample record.

        Returns
        -------
        images : dict
            Mapping from camera name to (H, W, 3) uint8 numpy array.
        calibration : dict
            Mapping from camera name to dict with keys:
            - ``"intrinsic"`` : (3, 3) camera intrinsic matrix
            - ``"extrinsic"`` : (4, 4) sensor-to-ego transformation
        """
        images: Dict[str, np.ndarray] = {}
        calibration: Dict[str, Dict[str, np.ndarray]] = {}

        for cam_name in self.cameras:
            if cam_name not in sample["data"]:
                warnings.warn(f"Camera {cam_name!r} not found in sample {sample['token']}")
                continue

            sd_token = sample["data"][cam_name]
            sd_record = self.nusc.get("sample_data", sd_token)
            cs_record = self.nusc.get(
                "calibrated_sensor", sd_record["calibrated_sensor_token"]
            )

            # Load image
            img_path = os.path.join(self.dataroot, sd_record["filename"])
            try:
                img = _load_image(img_path)
            except FileNotFoundError:
                warnings.warn(f"Image not found: {img_path}")
                img = np.zeros((900, 1600, 3), dtype=np.uint8)

            # Resize if requested
            if self.image_size is not None and _PIL_AVAILABLE:
                h, w = self.image_size
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((w, h), Image.BILINEAR)
                img = np.array(pil_img, dtype=np.uint8)

            images[cam_name] = img

            # Calibration: intrinsic (3x3) and extrinsic (4x4 sensor-to-ego)
            intrinsic = np.array(cs_record["camera_intrinsic"], dtype=np.float64)
            extrinsic = _build_transformation_matrix(
                translation=np.array(cs_record["translation"], dtype=np.float64),
                rotation=np.array(cs_record["rotation"], dtype=np.float64),
            )
            calibration[cam_name] = {
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
            }

        return images, calibration

    # ------------------------------------------------------------------
    # LiDAR loading
    # ------------------------------------------------------------------

    def _load_lidar(
        self, sample: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Load LiDAR point cloud and calibration.

        Parameters
        ----------
        sample : dict
            nuScenes sample record.

        Returns
        -------
        points : np.ndarray
            Shape (N, 5) point cloud [x, y, z, intensity, ring_index].
        calibration : dict
            Calibration info with ``"extrinsic"`` key (4x4 lidar-to-ego).
        """
        sd_token = sample["data"][LIDAR_CHANNEL]
        sd_record = self.nusc.get("sample_data", sd_token)
        cs_record = self.nusc.get(
            "calibrated_sensor", sd_record["calibrated_sensor_token"]
        )

        # Load point cloud
        pcl_path = os.path.join(self.dataroot, sd_record["filename"])
        if os.path.isfile(pcl_path):
            pc = LidarPointCloud.from_file(pcl_path)
            # pc.points is (4, N): [x, y, z, intensity]
            points = pc.points.T.astype(np.float32)  # (N, 4)
            # Add a dummy ring index column if not available (nuScenes LIDAR_TOP has 5 dims)
            if points.shape[1] == 4:
                ring_index = np.zeros((points.shape[0], 1), dtype=np.float32)
                points = np.concatenate([points, ring_index], axis=1)
        else:
            warnings.warn(f"LiDAR file not found: {pcl_path}")
            points = np.zeros((0, 5), dtype=np.float32)

        # Filter by range if specified
        if self.point_cloud_range is not None and points.shape[0] > 0:
            pcr = self.point_cloud_range
            mask = (
                (points[:, 0] >= pcr[0])
                & (points[:, 1] >= pcr[1])
                & (points[:, 2] >= pcr[2])
                & (points[:, 0] <= pcr[3])
                & (points[:, 1] <= pcr[4])
                & (points[:, 2] <= pcr[5])
            )
            points = points[mask]

        # Subsample if needed
        if self.max_lidar_points > 0 and points.shape[0] > self.max_lidar_points:
            indices = np.random.choice(
                points.shape[0], self.max_lidar_points, replace=False
            )
            points = points[indices]

        # Calibration: lidar to ego
        extrinsic = _build_transformation_matrix(
            translation=np.array(cs_record["translation"], dtype=np.float64),
            rotation=np.array(cs_record["rotation"], dtype=np.float64),
        )
        calibration = {"extrinsic": extrinsic}

        return points, calibration

    # ------------------------------------------------------------------
    # Radar loading
    # ------------------------------------------------------------------

    def _load_radar(
        self, sample: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """Load radar point clouds for all 5 radar sensors.

        Parameters
        ----------
        sample : dict
            nuScenes sample record.

        Returns
        -------
        radar_points : dict
            Mapping from radar channel name to (M, 18) float32 array.
            Columns: x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp,
            is_quality_valid, ambig_state, x_rms, y_rms, invalid_state,
            pdh0, vx_rms, vy_rms.
        calibrations : dict
            Mapping from radar channel to calibration dict with ``"extrinsic"`` key.
        """
        radar_points: Dict[str, np.ndarray] = {}
        calibrations: Dict[str, Dict[str, np.ndarray]] = {}

        for radar_name in RADAR_CHANNELS:
            if radar_name not in sample["data"]:
                continue

            sd_token = sample["data"][radar_name]
            sd_record = self.nusc.get("sample_data", sd_token)
            cs_record = self.nusc.get(
                "calibrated_sensor", sd_record["calibrated_sensor_token"]
            )

            # Load radar point cloud
            radar_path = os.path.join(self.dataroot, sd_record["filename"])
            if os.path.isfile(radar_path):
                pc = RadarPointCloud.from_file(radar_path)
                # pc.points is (18, M): radar-specific features
                points = pc.points.T.astype(np.float32)  # (M, 18)
            else:
                warnings.warn(f"Radar file not found: {radar_path}")
                points = np.zeros((0, 18), dtype=np.float32)

            radar_points[radar_name] = points

            # Calibration
            extrinsic = _build_transformation_matrix(
                translation=np.array(cs_record["translation"], dtype=np.float64),
                rotation=np.array(cs_record["rotation"], dtype=np.float64),
            )
            calibrations[radar_name] = {"extrinsic": extrinsic}

        return radar_points, calibrations

    # ------------------------------------------------------------------
    # Ego pose
    # ------------------------------------------------------------------

    def _load_ego_pose(self, sample: Dict[str, Any]) -> np.ndarray:
        """Load the ego vehicle pose for the sample (from LiDAR timestamp).

        Parameters
        ----------
        sample : dict
            nuScenes sample record.

        Returns
        -------
        np.ndarray
            Shape (4, 4) ego-to-global transformation matrix.
        """
        sd_token = sample["data"][LIDAR_CHANNEL]
        sd_record = self.nusc.get("sample_data", sd_token)
        ep_record = self.nusc.get("ego_pose", sd_record["ego_pose_token"])

        ego_pose = _build_transformation_matrix(
            translation=np.array(ep_record["translation"], dtype=np.float64),
            rotation=np.array(ep_record["rotation"], dtype=np.float64),
        )
        return ego_pose

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def _load_annotations(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load 3D bounding box annotations for the sample.

        Parameters
        ----------
        sample : dict
            nuScenes sample record.

        Returns
        -------
        list of dict
            Each annotation dict contains:
            - ``"token"`` : str - annotation token
            - ``"category_name"`` : str - full category name
            - ``"detection_name"`` : str - mapped detection class or "unknown"
            - ``"location"`` : np.ndarray (3,) - center x, y, z in global frame
            - ``"dimensions"`` : np.ndarray (3,) - width, length, height
            - ``"rotation"`` : np.ndarray (4,) - quaternion [w, x, y, z]
            - ``"velocity"`` : np.ndarray (2,) - vx, vy (NaN if unavailable)
            - ``"num_lidar_pts"`` : int
            - ``"num_radar_pts"`` : int
            - ``"visibility"`` : int - visibility level (1-4)
        """
        annotations: List[Dict[str, Any]] = []

        for ann_token in sample["anns"]:
            ann_record = self.nusc.get("sample_annotation", ann_token)

            # Map to detection class
            category = ann_record["category_name"]
            detection_name = self._map_category_to_detection(category)

            # Velocity (nuScenes provides box velocity in global frame)
            try:
                velocity = self.nusc.box_velocity(ann_token)[:2]  # vx, vy
            except Exception:
                velocity = np.array([np.nan, np.nan], dtype=np.float64)

            # Visibility
            visibility_token = ann_record.get("visibility_token", "")
            if visibility_token:
                try:
                    vis_record = self.nusc.get("visibility", visibility_token)
                    visibility = int(vis_record["level"])
                except Exception:
                    visibility = 0
            else:
                visibility = 0

            annotation = {
                "token": ann_token,
                "category_name": category,
                "detection_name": detection_name,
                "location": np.array(ann_record["translation"], dtype=np.float64),
                "dimensions": np.array(ann_record["size"], dtype=np.float64),  # wlh
                "rotation": np.array(ann_record["rotation"], dtype=np.float64),  # wxyz
                "velocity": velocity.astype(np.float64),
                "num_lidar_pts": ann_record.get("num_lidar_pts", 0),
                "num_radar_pts": ann_record.get("num_radar_pts", 0),
                "visibility": visibility,
            }
            annotations.append(annotation)

        return annotations

    @staticmethod
    def _map_category_to_detection(category_name: str) -> str:
        """Map a nuScenes category to a detection class name.

        Parameters
        ----------
        category_name : str
            Full nuScenes category (e.g., ``"vehicle.car"``).

        Returns
        -------
        str
            Mapped detection class or ``"unknown"``.
        """
        mapping = {
            "vehicle.car": "car",
            "vehicle.truck": "truck",
            "vehicle.construction": "construction_vehicle",
            "vehicle.bus.bendy": "bus",
            "vehicle.bus.rigid": "bus",
            "vehicle.trailer": "trailer",
            "movable_object.barrier": "barrier",
            "vehicle.motorcycle": "motorcycle",
            "vehicle.bicycle": "bicycle",
            "human.pedestrian.adult": "pedestrian",
            "human.pedestrian.child": "pedestrian",
            "human.pedestrian.construction_worker": "pedestrian",
            "human.pedestrian.police_officer": "pedestrian",
            "movable_object.trafficcone": "traffic_cone",
        }
        # Try exact match first
        if category_name in mapping:
            return mapping[category_name]
        # Try prefix match
        for prefix, det_name in mapping.items():
            if category_name.startswith(prefix.rsplit(".", 1)[0]):
                return det_name
        return "unknown"

    # ------------------------------------------------------------------
    # Map data
    # ------------------------------------------------------------------

    def _load_map_data(
        self, sample: Dict[str, Any], ego_pose: np.ndarray
    ) -> Dict[str, Any]:
        """Load HD map annotations near the ego vehicle.

        Parameters
        ----------
        sample : dict
            nuScenes sample record.
        ego_pose : np.ndarray
            Shape (4, 4) ego pose matrix.

        Returns
        -------
        dict
            Map data with keys:
            - ``"lane_dividers"`` : list of (N, 2) arrays (x, y polylines)
            - ``"road_boundaries"`` : list of (N, 2) arrays
            - ``"pedestrian_crossings"`` : list of (N, 2) arrays
            - ``"location"`` : str - map location name
        """
        # Determine the map location for this sample
        scene_token = sample["scene_token"]
        scene = self.nusc.get("scene", scene_token)
        log = self.nusc.get("log", scene["log_token"])
        location = log["location"]

        result: Dict[str, Any] = {
            "lane_dividers": [],
            "road_boundaries": [],
            "pedestrian_crossings": [],
            "location": location,
        }

        if location not in self._maps:
            return result

        nusc_map = self._maps[location]
        ego_x, ego_y = ego_pose[0, 3], ego_pose[1, 3]
        patch_radius = 60.0  # meters around ego
        patch_box = (ego_x, ego_y, patch_radius * 2, patch_radius * 2)
        patch_angle = 0.0  # we retrieve in global, transform later if needed

        # Lane dividers
        try:
            lane_records = nusc_map.get_records_in_patch(
                patch_box, ["lane_connector", "lane"], mode="intersect"
            )
            for lane_token in lane_records.get("lane", []):
                lane = nusc_map.get_arcline_path(lane_token)
                # Discretize lane into points
                pts = nusc_map.discretize_lanes([lane_token], resolution_meters=1.0)
                if lane_token in pts and len(pts[lane_token]) > 0:
                    polyline = np.array(pts[lane_token], dtype=np.float64)[:, :2]
                    result["lane_dividers"].append(polyline)
        except Exception:
            pass

        # Road boundaries (road segments)
        try:
            road_records = nusc_map.get_records_in_patch(
                patch_box, ["road_segment"], mode="intersect"
            )
            for seg_token in road_records.get("road_segment", []):
                seg = nusc_map.get("road_segment", seg_token)
                if "exterior_node_tokens" in seg:
                    nodes = []
                    for node_token in seg["exterior_node_tokens"]:
                        node = nusc_map.get("node", node_token)
                        nodes.append([node["x"], node["y"]])
                    if nodes:
                        result["road_boundaries"].append(
                            np.array(nodes, dtype=np.float64)
                        )
        except Exception:
            pass

        # Pedestrian crossings
        try:
            ped_records = nusc_map.get_records_in_patch(
                patch_box, ["ped_crossing"], mode="intersect"
            )
            for ped_token in ped_records.get("ped_crossing", []):
                ped = nusc_map.get("ped_crossing", ped_token)
                if "exterior_node_tokens" in ped:
                    nodes = []
                    for node_token in ped["exterior_node_tokens"]:
                        node = nusc_map.get("node", node_token)
                        nodes.append([node["x"], node["y"]])
                    if nodes:
                        result["pedestrian_crossings"].append(
                            np.array(nodes, dtype=np.float64)
                        )
        except Exception:
            pass

        return result

    # ------------------------------------------------------------------
    # Temporal history
    # ------------------------------------------------------------------

    def _load_history(
        self, current_sample: Dict[str, Any], n_frames: int
    ) -> List[Dict[str, Any]]:
        """Load past N frames of data for temporal modeling.

        Parameters
        ----------
        current_sample : dict
            The current nuScenes sample record.
        n_frames : int
            Number of past frames to load.

        Returns
        -------
        list of dict
            List of past frame data, ordered from most recent to oldest.
            Each frame contains ego_pose, lidar_points, images, and annotations.
        """
        history: List[Dict[str, Any]] = []
        sample_token = current_sample["prev"]

        for _ in range(n_frames):
            if not sample_token:
                break

            past_sample = self.nusc.get("sample", sample_token)

            frame_data: Dict[str, Any] = {
                "sample_token": sample_token,
                "timestamp": past_sample["timestamp"],
            }

            # Load ego pose
            frame_data["ego_pose"] = self._load_ego_pose(past_sample)

            # Load LiDAR if requested
            if self.load_lidar:
                lidar_pts, _ = self._load_lidar(past_sample)
                frame_data["lidar_points"] = lidar_pts
            else:
                frame_data["lidar_points"] = None

            # Load camera images
            images, calibration = self._load_cameras(past_sample)
            frame_data["images"] = images
            frame_data["calibration"] = calibration

            # Load annotations
            frame_data["annotations"] = self._load_annotations(past_sample)

            history.append(frame_data)
            sample_token = past_sample["prev"]

        return history

    # ------------------------------------------------------------------
    # TensorFlow interface
    # ------------------------------------------------------------------

    @classmethod
    def as_tf_dataset(
        cls,
        dataroot: str,
        version: str = "v1.0-trainval",
        split: str = "train",
        cameras: Optional[List[str]] = None,
        load_lidar: bool = True,
        load_radar: bool = False,
        load_map: bool = False,
        n_history_frames: int = 0,
        max_lidar_points: int = 34720,
        image_size: Optional[Tuple[int, int]] = None,
        batch_size: int = 1,
        shuffle: bool = True,
        num_parallel_calls: int = 4,
        prefetch_buffer: int = 2,
    ) -> Any:
        """Create a TensorFlow tf.data.Dataset from the nuScenes data.

        This factory method constructs a tf.data.Dataset that lazily loads
        nuScenes samples on demand. Each element is a nested dictionary of
        tensors matching the PyTorch interface output structure.

        Parameters
        ----------
        dataroot : str
            Path to the nuScenes dataset root.
        version : str
            Dataset version.
        split : str
            Data split.
        cameras : list of str, optional
            Camera channels to load.
        load_lidar : bool
            Whether to load LiDAR.
        load_radar : bool
            Whether to load radar.
        load_map : bool
            Whether to load map data.
        n_history_frames : int
            Number of past frames.
        max_lidar_points : int
            Maximum LiDAR points (required for fixed tensor shapes in TF).
        image_size : tuple of int, optional
            Target image size (height, width). Required for batching in TF.
        batch_size : int
            Batch size for the tf.data.Dataset.
        shuffle : bool
            Whether to shuffle the dataset.
        num_parallel_calls : int
            Number of parallel calls for map operations.
        prefetch_buffer : int
            Number of batches to prefetch.

        Returns
        -------
        tf.data.Dataset
            A TensorFlow dataset yielding batched samples.

        Raises
        ------
        ImportError
            If TensorFlow is not installed.
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required for as_tf_dataset(). "
                "Install with: pip install tensorflow"
            )

        # Create the PyTorch dataset instance for data access
        # (we use it as a data source, not for PyTorch-specific features)
        cams = cameras if cameras is not None else list(CAMERA_CHANNELS)
        img_h, img_w = image_size if image_size is not None else (900, 1600)

        pt_dataset = cls(
            dataroot=dataroot,
            version=version,
            split=split,
            cameras=cams,
            load_lidar=load_lidar,
            load_radar=load_radar,
            load_map=load_map,
            n_history_frames=n_history_frames,
            max_lidar_points=max_lidar_points,
            image_size=(img_h, img_w) if image_size is not None else None,
        )

        num_samples = len(pt_dataset)
        num_cams = len(cams)

        # Define output signature
        output_signature = {
            "images": tf.TensorSpec(
                shape=(num_cams, img_h, img_w, 3), dtype=tf.uint8
            ),
            "ego_pose": tf.TensorSpec(shape=(4, 4), dtype=tf.float64),
            "timestamp": tf.TensorSpec(shape=(), dtype=tf.int64),
        }

        if load_lidar:
            output_signature["lidar_points"] = tf.TensorSpec(
                shape=(max_lidar_points, 5), dtype=tf.float32
            )
            output_signature["lidar_mask"] = tf.TensorSpec(
                shape=(max_lidar_points,), dtype=tf.bool
            )

        def _generator():
            """Generator that yields samples as TF-compatible dicts."""
            for idx in range(num_samples):
                sample = pt_dataset[idx]

                # Stack camera images into a single tensor
                img_stack = np.stack(
                    [sample["images"].get(cam, np.zeros((img_h, img_w, 3), dtype=np.uint8))
                     for cam in cams],
                    axis=0,
                )

                output = {
                    "images": img_stack,
                    "ego_pose": sample["ego_pose"],
                    "timestamp": np.int64(sample["timestamp"]),
                }

                if load_lidar and sample["lidar_points"] is not None:
                    pts = sample["lidar_points"]
                    n_pts = pts.shape[0]
                    # Pad or truncate to fixed size
                    padded = np.zeros((max_lidar_points, 5), dtype=np.float32)
                    mask = np.zeros(max_lidar_points, dtype=bool)
                    actual = min(n_pts, max_lidar_points)
                    padded[:actual] = pts[:actual]
                    mask[:actual] = True
                    output["lidar_points"] = padded
                    output["lidar_mask"] = mask

                yield output

        tf_dataset = tf.data.Dataset.from_generator(
            _generator, output_signature=output_signature
        )

        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=min(1000, num_samples))

        tf_dataset = tf_dataset.batch(batch_size)
        tf_dataset = tf_dataset.prefetch(prefetch_buffer)

        return tf_dataset

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NuScenesDataset("
            f"version={self.version!r}, split={self.split!r}, "
            f"samples={len(self)}, cameras={self.cameras}, "
            f"lidar={self.load_lidar}, radar={self.load_radar}, "
            f"map={self.load_map}, history={self.n_history_frames})"
        )
