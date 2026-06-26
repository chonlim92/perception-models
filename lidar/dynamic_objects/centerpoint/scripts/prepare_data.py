"""
CenterPoint nuScenes Data Preparation Script.

Prepares nuScenes dataset for CenterPoint training:
1. Parses nuScenes database JSON tables
2. Aggregates 10 LiDAR sweeps with ego-motion compensation
3. Creates info pickle files for train/val splits
4. Builds GT database for GT-sampling augmentation
5. Prints dataset statistics

Usage:
    python prepare_data.py --data-root /data/nuscenes --version v1.0-trainval --workers 16
"""

import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# nuScenes Official Train/Val Scene Splits
# =============================================================================

NUSCENES_TRAINVAL_TRAIN_SCENES = [
    "scene-0001", "scene-0002", "scene-0004", "scene-0005", "scene-0006",
    "scene-0007", "scene-0008", "scene-0009", "scene-0010", "scene-0011",
    "scene-0019", "scene-0020", "scene-0021", "scene-0022", "scene-0023",
    "scene-0024", "scene-0025", "scene-0026", "scene-0027", "scene-0028",
    "scene-0029", "scene-0030", "scene-0031", "scene-0032", "scene-0033",
    "scene-0034", "scene-0041", "scene-0042", "scene-0043", "scene-0044",
    "scene-0045", "scene-0046", "scene-0047", "scene-0048", "scene-0049",
    "scene-0050", "scene-0051", "scene-0052", "scene-0053", "scene-0054",
    "scene-0055", "scene-0056", "scene-0057", "scene-0058", "scene-0059",
    "scene-0060", "scene-0061", "scene-0062", "scene-0063", "scene-0064",
    "scene-0065", "scene-0066", "scene-0067", "scene-0068", "scene-0069",
    "scene-0070", "scene-0071", "scene-0072", "scene-0073", "scene-0074",
    "scene-0075", "scene-0076", "scene-0077", "scene-0078", "scene-0079",
    "scene-0080", "scene-0081", "scene-0082", "scene-0083", "scene-0084",
    "scene-0085", "scene-0086", "scene-0087", "scene-0088", "scene-0089",
    "scene-0090", "scene-0091", "scene-0092", "scene-0093", "scene-0094",
    "scene-0095", "scene-0096", "scene-0097", "scene-0098", "scene-0099",
    "scene-0100", "scene-0101", "scene-0102", "scene-0103", "scene-0104",
    "scene-0105", "scene-0106", "scene-0107", "scene-0108", "scene-0109",
    "scene-0110", "scene-0111", "scene-0112", "scene-0113", "scene-0114",
    "scene-0115", "scene-0116", "scene-0117", "scene-0118", "scene-0119",
    "scene-0120", "scene-0121", "scene-0122", "scene-0123", "scene-0124",
    "scene-0125", "scene-0126", "scene-0127", "scene-0128", "scene-0129",
    "scene-0130", "scene-0131", "scene-0132", "scene-0133", "scene-0134",
    "scene-0135", "scene-0138", "scene-0139", "scene-0149", "scene-0150",
    "scene-0151", "scene-0152", "scene-0154", "scene-0155", "scene-0157",
    "scene-0158", "scene-0159", "scene-0160", "scene-0161", "scene-0162",
    "scene-0163", "scene-0164", "scene-0165", "scene-0166", "scene-0167",
    "scene-0168", "scene-0170", "scene-0171", "scene-0172", "scene-0173",
    "scene-0174", "scene-0175", "scene-0176", "scene-0177", "scene-0178",
    "scene-0179", "scene-0180", "scene-0181", "scene-0182", "scene-0183",
    "scene-0184", "scene-0185", "scene-0187", "scene-0188", "scene-0190",
    "scene-0191", "scene-0192", "scene-0193", "scene-0194", "scene-0195",
    "scene-0196", "scene-0199", "scene-0200", "scene-0202", "scene-0203",
    "scene-0204", "scene-0206", "scene-0207", "scene-0208", "scene-0209",
    "scene-0210", "scene-0211", "scene-0212", "scene-0213", "scene-0214",
    "scene-0218", "scene-0219", "scene-0220", "scene-0222", "scene-0224",
    "scene-0225", "scene-0226", "scene-0227", "scene-0228", "scene-0229",
    "scene-0230", "scene-0231", "scene-0232", "scene-0233", "scene-0234",
    "scene-0235", "scene-0236", "scene-0237", "scene-0238", "scene-0239",
    "scene-0240", "scene-0241", "scene-0242", "scene-0243", "scene-0244",
    "scene-0245", "scene-0246", "scene-0247", "scene-0248", "scene-0249",
    "scene-0250", "scene-0251", "scene-0252", "scene-0253", "scene-0254",
    "scene-0255", "scene-0256", "scene-0257", "scene-0258", "scene-0259",
    "scene-0260", "scene-0261", "scene-0262", "scene-0263", "scene-0264",
    "scene-0283", "scene-0284", "scene-0285", "scene-0286", "scene-0287",
    "scene-0288", "scene-0289", "scene-0290", "scene-0291", "scene-0292",
    "scene-0293", "scene-0294", "scene-0295", "scene-0296", "scene-0297",
    "scene-0298", "scene-0299", "scene-0300", "scene-0301", "scene-0302",
    "scene-0303", "scene-0304", "scene-0305", "scene-0306", "scene-0315",
    "scene-0316", "scene-0317", "scene-0318", "scene-0321", "scene-0323",
    "scene-0324", "scene-0328", "scene-0347", "scene-0348", "scene-0349",
    "scene-0350", "scene-0351", "scene-0352", "scene-0353", "scene-0354",
    "scene-0355", "scene-0356", "scene-0357", "scene-0358", "scene-0359",
    "scene-0360", "scene-0361", "scene-0362", "scene-0363", "scene-0364",
    "scene-0365", "scene-0366", "scene-0367", "scene-0368", "scene-0369",
    "scene-0370", "scene-0371", "scene-0372", "scene-0373", "scene-0374",
    "scene-0375", "scene-0376", "scene-0377", "scene-0378", "scene-0379",
    "scene-0380", "scene-0381", "scene-0382", "scene-0383", "scene-0384",
    "scene-0385", "scene-0386", "scene-0388", "scene-0389", "scene-0390",
    "scene-0391", "scene-0392", "scene-0393", "scene-0394", "scene-0395",
    "scene-0396", "scene-0397", "scene-0398", "scene-0399", "scene-0400",
    "scene-0401", "scene-0402", "scene-0403", "scene-0405", "scene-0406",
    "scene-0407", "scene-0408", "scene-0410", "scene-0411", "scene-0412",
    "scene-0413", "scene-0414", "scene-0415", "scene-0416", "scene-0417",
    "scene-0418", "scene-0419", "scene-0420", "scene-0421", "scene-0422",
    "scene-0423", "scene-0424", "scene-0425", "scene-0426", "scene-0427",
    "scene-0428", "scene-0429", "scene-0430", "scene-0431", "scene-0432",
    "scene-0433", "scene-0434", "scene-0435", "scene-0436", "scene-0437",
    "scene-0438", "scene-0439", "scene-0440", "scene-0441", "scene-0442",
    "scene-0443", "scene-0444", "scene-0445", "scene-0446", "scene-0447",
    "scene-0448", "scene-0449", "scene-0450", "scene-0451", "scene-0452",
    "scene-0453", "scene-0454", "scene-0455", "scene-0456", "scene-0457",
    "scene-0458", "scene-0459", "scene-0461", "scene-0462", "scene-0463",
    "scene-0464", "scene-0465", "scene-0467", "scene-0468", "scene-0469",
    "scene-0471", "scene-0472", "scene-0474", "scene-0475", "scene-0476",
    "scene-0477", "scene-0478", "scene-0479", "scene-0480", "scene-0499",
    "scene-0500", "scene-0501", "scene-0502", "scene-0504", "scene-0505",
    "scene-0506", "scene-0507", "scene-0508", "scene-0509", "scene-0510",
    "scene-0511", "scene-0512", "scene-0513", "scene-0514", "scene-0515",
    "scene-0517", "scene-0518", "scene-0525", "scene-0526", "scene-0527",
    "scene-0528", "scene-0529", "scene-0530", "scene-0531", "scene-0532",
    "scene-0533", "scene-0534", "scene-0535", "scene-0536", "scene-0537",
    "scene-0538", "scene-0539", "scene-0541", "scene-0542", "scene-0543",
    "scene-0544", "scene-0545", "scene-0546", "scene-0547", "scene-0548",
    "scene-0549", "scene-0550", "scene-0551", "scene-0552", "scene-0553",
    "scene-0554", "scene-0555", "scene-0556", "scene-0557", "scene-0558",
    "scene-0559", "scene-0560", "scene-0561", "scene-0562", "scene-0563",
    "scene-0564", "scene-0565", "scene-0625", "scene-0626", "scene-0627",
    "scene-0629", "scene-0630", "scene-0632", "scene-0633", "scene-0634",
    "scene-0635", "scene-0636", "scene-0637", "scene-0638", "scene-0770",
    "scene-0771", "scene-0775", "scene-0777", "scene-0778", "scene-0780",
    "scene-0781", "scene-0782", "scene-0783", "scene-0784", "scene-0794",
    "scene-0795", "scene-0796", "scene-0797", "scene-0798", "scene-0799",
    "scene-0800", "scene-0802", "scene-0803", "scene-0804", "scene-0805",
    "scene-0806", "scene-0808", "scene-0809", "scene-0810", "scene-0811",
    "scene-0812", "scene-0813", "scene-0815", "scene-0816", "scene-0817",
    "scene-0819", "scene-0820", "scene-0821", "scene-0822", "scene-0847",
    "scene-0848", "scene-0849", "scene-0850", "scene-0851", "scene-0852",
    "scene-0853", "scene-0854", "scene-0855", "scene-0856", "scene-0858",
    "scene-0860", "scene-0861", "scene-0862", "scene-0863", "scene-0864",
    "scene-0865", "scene-0866", "scene-0868", "scene-0869", "scene-0870",
    "scene-0871", "scene-0872", "scene-0873", "scene-0875", "scene-0876",
    "scene-0877", "scene-0878", "scene-0880", "scene-0882", "scene-0883",
    "scene-0884", "scene-0885", "scene-0886", "scene-0887", "scene-0888",
    "scene-0889", "scene-0890", "scene-0891", "scene-0892", "scene-0893",
    "scene-0894", "scene-0895", "scene-0896", "scene-0897", "scene-0898",
    "scene-0899", "scene-0900", "scene-0901", "scene-0902", "scene-0903",
    "scene-0945", "scene-0947", "scene-0949", "scene-0952", "scene-0953",
    "scene-0955", "scene-0956", "scene-0957", "scene-0958", "scene-0959",
    "scene-0960", "scene-0961", "scene-0975", "scene-0976", "scene-0977",
    "scene-0978", "scene-0979", "scene-0980", "scene-0981", "scene-0982",
    "scene-0983", "scene-0984", "scene-0988", "scene-0989", "scene-0990",
    "scene-0991", "scene-0992", "scene-0994", "scene-0995", "scene-0996",
    "scene-0997", "scene-0998", "scene-0999", "scene-1000", "scene-1001",
    "scene-1002", "scene-1003", "scene-1004", "scene-1005", "scene-1006",
    "scene-1007", "scene-1008", "scene-1009", "scene-1010", "scene-1011",
    "scene-1012", "scene-1013", "scene-1014", "scene-1015", "scene-1016",
    "scene-1017", "scene-1018", "scene-1019", "scene-1020", "scene-1021",
    "scene-1022", "scene-1023", "scene-1024", "scene-1025", "scene-1044",
    "scene-1045", "scene-1046", "scene-1047", "scene-1048", "scene-1049",
    "scene-1050", "scene-1051", "scene-1052", "scene-1053", "scene-1054",
    "scene-1055", "scene-1056", "scene-1057", "scene-1058", "scene-1074",
    "scene-1075", "scene-1076", "scene-1077", "scene-1078", "scene-1079",
    "scene-1080", "scene-1081", "scene-1082", "scene-1083", "scene-1084",
    "scene-1085", "scene-1086", "scene-1087", "scene-1088", "scene-1089",
    "scene-1090", "scene-1091", "scene-1092", "scene-1093", "scene-1094",
    "scene-1095", "scene-1096", "scene-1097", "scene-1098", "scene-1099",
    "scene-1100", "scene-1101", "scene-1102", "scene-1104", "scene-1105",
    "scene-1106", "scene-1107", "scene-1108", "scene-1109", "scene-1110",
]

NUSCENES_TRAINVAL_VAL_SCENES = [
    "scene-0003", "scene-0012", "scene-0013", "scene-0014", "scene-0015",
    "scene-0016", "scene-0017", "scene-0018", "scene-0035", "scene-0036",
    "scene-0037", "scene-0038", "scene-0039", "scene-0040", "scene-0136",
    "scene-0137", "scene-0138", "scene-0140", "scene-0141", "scene-0142",
    "scene-0143", "scene-0144", "scene-0145", "scene-0146", "scene-0147",
    "scene-0148", "scene-0149", "scene-0153", "scene-0154", "scene-0156",
    "scene-0186", "scene-0189", "scene-0190", "scene-0197", "scene-0198",
    "scene-0199", "scene-0200", "scene-0201", "scene-0202", "scene-0203",
    "scene-0204", "scene-0205", "scene-0206", "scene-0207", "scene-0208",
    "scene-0209", "scene-0210", "scene-0211", "scene-0212", "scene-0213",
    "scene-0214", "scene-0215", "scene-0216", "scene-0217", "scene-0218",
    "scene-0219", "scene-0266", "scene-0267", "scene-0268", "scene-0269",
    "scene-0270", "scene-0271", "scene-0272", "scene-0273", "scene-0274",
    "scene-0275", "scene-0276", "scene-0277", "scene-0278", "scene-0279",
    "scene-0280", "scene-0281", "scene-0282", "scene-0283", "scene-0284",
    "scene-0285", "scene-0286", "scene-0287", "scene-0288", "scene-0289",
    "scene-0290", "scene-0291", "scene-0292", "scene-0293", "scene-0294",
    "scene-0295", "scene-0296", "scene-0297", "scene-0298", "scene-0299",
    "scene-0300", "scene-0301", "scene-0302", "scene-0303", "scene-0304",
    "scene-0305", "scene-0306", "scene-0315", "scene-0316", "scene-0317",
    "scene-0318", "scene-0321", "scene-0323", "scene-0324", "scene-0328",
    "scene-0329", "scene-0330", "scene-0331", "scene-0332", "scene-0344",
    "scene-0345", "scene-0346", "scene-0519", "scene-0520", "scene-0521",
    "scene-0522", "scene-0523", "scene-0524", "scene-0552", "scene-0553",
    "scene-0554", "scene-0555", "scene-0556", "scene-0557", "scene-0558",
    "scene-0559", "scene-0560", "scene-0561", "scene-0562", "scene-0563",
    "scene-0564", "scene-0565", "scene-0625", "scene-0626", "scene-0627",
    "scene-0629", "scene-0630", "scene-0632", "scene-0633", "scene-0634",
    "scene-0635", "scene-0636", "scene-0637", "scene-0638", "scene-0904",
    "scene-0905", "scene-0906", "scene-0907", "scene-0908", "scene-0909",
    "scene-0910", "scene-0911", "scene-0912", "scene-0913", "scene-0914",
]

NUSCENES_MINI_TRAIN_SCENES = [
    "scene-0061", "scene-0553", "scene-0655", "scene-0757", "scene-0796",
    "scene-1077", "scene-1094", "scene-1100",
]

NUSCENES_MINI_VAL_SCENES = [
    "scene-0103", "scene-0916",
]

# nuScenes detection classes
NUSCENES_CLASSES = [
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

# Mapping from nuScenes category names to detection class names
CATEGORY_TO_DETECTION_NAME = {
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "car",
    "vehicle.emergency.police": "car",
    "vehicle.trailer": "trailer",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": None,
    "movable_object.debris": None,
    "static_object.bicycle_rack": None,
    "animal": None,
}


# =============================================================================
# Geometry Utilities
# =============================================================================


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = quaternion
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)
    return rotation_matrix


def make_transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous transform from rotation matrix and translation."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def get_sensor_to_global_transform(
    ego_pose: Dict[str, Any],
    calibrated_sensor: Dict[str, Any],
) -> np.ndarray:
    """Compute sensor-to-global transform via ego vehicle frame.

    Transform chain: sensor -> ego -> global
    """
    # Sensor to ego
    sensor_rotation = quaternion_to_rotation_matrix(
        np.array(calibrated_sensor["rotation"])
    )
    sensor_translation = np.array(calibrated_sensor["translation"])
    sensor_to_ego = make_transform_matrix(sensor_rotation, sensor_translation)

    # Ego to global
    ego_rotation = quaternion_to_rotation_matrix(np.array(ego_pose["rotation"]))
    ego_translation = np.array(ego_pose["translation"])
    ego_to_global = make_transform_matrix(ego_rotation, ego_translation)

    return ego_to_global @ sensor_to_ego


def points_in_box(points: np.ndarray, box_center: np.ndarray, box_size: np.ndarray,
                  box_yaw: float) -> np.ndarray:
    """Check which points are inside a rotated 3D bounding box.

    Args:
        points: (N, 3+) array of points (only xyz used).
        box_center: (3,) center of the box (x, y, z).
        box_size: (3,) dimensions of box (width, length, height) = (dx, dy, dz).
        box_yaw: rotation angle around z-axis in radians.

    Returns:
        Boolean mask of shape (N,) indicating points inside the box.
    """
    # Translate points to box center
    points_centered = points[:, :3] - box_center[np.newaxis, :]

    # Rotate points to box-aligned coordinate frame (inverse rotation)
    cos_yaw = np.cos(-box_yaw)
    sin_yaw = np.sin(-box_yaw)
    rotation_z = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    points_aligned = points_centered @ rotation_z.T

    # Check if points fall within half-extents
    half_size = box_size / 2.0
    mask_x = np.abs(points_aligned[:, 0]) <= half_size[0]
    mask_y = np.abs(points_aligned[:, 1]) <= half_size[1]
    mask_z = np.abs(points_aligned[:, 2]) <= half_size[2]

    return mask_x & mask_y & mask_z


def compute_box_velocity(
    nusc_db: "NuScenesDatabase",
    sample_token: str,
    annotation_token: str,
) -> np.ndarray:
    """Compute velocity (vx, vy) of an annotation in the global frame.

    Uses centered finite difference between prev and next annotations when
    available, otherwise forward/backward difference. Returns (0, 0) if
    annotation has no temporal neighbors.
    """
    current_ann = nusc_db.get("sample_annotation", annotation_token)
    current_sample = nusc_db.get("sample", sample_token)
    current_time = current_sample["timestamp"] * 1e-6  # microseconds to seconds

    has_prev = current_ann["prev"] != ""
    has_next = current_ann["next"] != ""

    if not has_prev and not has_next:
        return np.array([0.0, 0.0], dtype=np.float32)

    if has_prev and has_next:
        prev_ann = nusc_db.get("sample_annotation", current_ann["prev"])
        next_ann = nusc_db.get("sample_annotation", current_ann["next"])
        prev_sample = nusc_db.get("sample", prev_ann["sample_token"])
        next_sample = nusc_db.get("sample", next_ann["sample_token"])
        prev_time = prev_sample["timestamp"] * 1e-6
        next_time = next_sample["timestamp"] * 1e-6
        dt = next_time - prev_time
        if dt < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)
        diff = np.array(next_ann["translation"]) - np.array(prev_ann["translation"])
        velocity = diff[:2] / dt
    elif has_next:
        next_ann = nusc_db.get("sample_annotation", current_ann["next"])
        next_sample = nusc_db.get("sample", next_ann["sample_token"])
        next_time = next_sample["timestamp"] * 1e-6
        dt = next_time - current_time
        if dt < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)
        diff = np.array(next_ann["translation"]) - np.array(current_ann["translation"])
        velocity = diff[:2] / dt
    else:
        prev_ann = nusc_db.get("sample_annotation", current_ann["prev"])
        prev_sample = nusc_db.get("sample", prev_ann["sample_token"])
        prev_time = prev_sample["timestamp"] * 1e-6
        dt = current_time - prev_time
        if dt < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)
        diff = np.array(current_ann["translation"]) - np.array(prev_ann["translation"])
        velocity = diff[:2] / dt

    return velocity.astype(np.float32)


# =============================================================================
# nuScenes Database Loader
# =============================================================================


class NuScenesDatabase:
    """Lightweight loader for nuScenes database JSON tables.

    Loads the following tables:
        sample, sample_data, ego_pose, sample_annotation,
        calibrated_sensor, log, scene, category, instance
    """

    TABLE_NAMES = [
        "sample",
        "sample_data",
        "ego_pose",
        "sample_annotation",
        "calibrated_sensor",
        "log",
        "scene",
        "category",
        "instance",
    ]

    def __init__(self, data_root: str, version: str):
        self.data_root = data_root
        self.version = version
        self.table_root = os.path.join(data_root, version)

        # Raw table data: {table_name: [record, ...]}
        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        # Lookup by token: {table_name: {token: record}}
        self.token_lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}

        self._load_tables()

    def _load_tables(self):
        """Load all JSON table files and build token lookup indices."""
        print(f"Loading nuScenes database from: {self.table_root}")
        for table_name in self.TABLE_NAMES:
            table_path = os.path.join(self.table_root, f"{table_name}.json")
            if not os.path.isfile(table_path):
                raise FileNotFoundError(
                    f"Table file not found: {table_path}. "
                    f"Check --data-root and --version arguments."
                )
            with open(table_path, "r") as f:
                records = json.load(f)

            self.tables[table_name] = records
            self.token_lookup[table_name] = {
                record["token"]: record for record in records
            }
            print(f"  Loaded {table_name}: {len(records)} records")

    def get(self, table_name: str, token: str) -> Dict[str, Any]:
        """Get a record by token from the specified table."""
        return self.token_lookup[table_name][token]

    def get_records(self, table_name: str) -> List[Dict[str, Any]]:
        """Get all records from a table."""
        return self.tables[table_name]

    def get_scene_name(self, scene_token: str) -> str:
        """Get the scene name (e.g., 'scene-0001') for a scene token."""
        scene = self.get("scene", scene_token)
        return scene["name"]

    def get_sample_lidar_token(self, sample_token: str) -> str:
        """Get the LIDAR_TOP sample_data token for a given sample."""
        sample = self.get("sample", sample_token)
        return sample["data"]["LIDAR_TOP"]

    def get_category_name(self, instance_token: str) -> str:
        """Get the category name for an instance."""
        instance = self.get("instance", instance_token)
        category = self.get("category", instance["category_token"])
        return category["name"]


# =============================================================================
# Point Cloud I/O and Sweep Aggregation
# =============================================================================


def load_pointcloud(filepath: str) -> np.ndarray:
    """Load a nuScenes point cloud from a .bin file.

    nuScenes LiDAR point clouds are stored as binary files with 5 float32
    values per point: (x, y, z, intensity, ring_index).

    Args:
        filepath: Path to the .bin file.

    Returns:
        Array of shape (N, 5) with columns (x, y, z, intensity, ring_index).
    """
    points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)
    return points


def aggregate_sweeps(
    nusc_db: NuScenesDatabase,
    sample_token: str,
    data_root: str,
    num_sweeps: int = 10,
) -> np.ndarray:
    """Aggregate multiple LiDAR sweeps with ego-motion compensation.

    Collects the current keyframe plus up to (num_sweeps - 1) previous sweeps.
    Each past sweep's points are transformed to the current keyframe's LiDAR
    coordinate frame via the transform chain:
        past_lidar -> past_ego -> global -> current_ego -> current_lidar

    The time_lag (seconds since current keyframe) is appended as a 5th feature.

    Args:
        nusc_db: Loaded nuScenes database.
        sample_token: Token of the current keyframe sample.
        data_root: Root directory of nuScenes data (for resolving file paths).
        num_sweeps: Number of sweeps to aggregate (default 10).

    Returns:
        Array of shape (N, 5): (x, y, z, intensity, time_lag).
    """
    # Get current keyframe info
    sample = nusc_db.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    current_sd = nusc_db.get("sample_data", lidar_token)

    # Compute current keyframe's sensor-to-global transform
    current_ego_pose = nusc_db.get("ego_pose", current_sd["ego_pose_token"])
    current_cs = nusc_db.get(
        "calibrated_sensor", current_sd["calibrated_sensor_token"]
    )
    current_sensor_to_global = get_sensor_to_global_transform(
        current_ego_pose, current_cs
    )
    # Inverse: global -> current_lidar
    global_to_current_lidar = np.linalg.inv(current_sensor_to_global)

    current_timestamp = current_sd["timestamp"]

    # Collect sweep sample_data tokens (current + prev sweeps)
    sweep_tokens = [lidar_token]
    sd = current_sd
    while len(sweep_tokens) < num_sweeps and sd["prev"] != "":
        sd = nusc_db.get("sample_data", sd["prev"])
        sweep_tokens.append(sd["token"])

    all_points = []

    for sweep_token in sweep_tokens:
        sweep_sd = nusc_db.get("sample_data", sweep_token)

        # Load raw point cloud
        pc_path = os.path.join(data_root, sweep_sd["filename"])
        points = load_pointcloud(pc_path)  # (N, 5): x,y,z,intensity,ring_index

        # Compute time lag
        time_lag = (current_timestamp - sweep_sd["timestamp"]) * 1e-6  # seconds

        if sweep_token == lidar_token:
            # Current keyframe: no transformation needed, time_lag = 0
            sweep_points = np.column_stack([
                points[:, :3],
                points[:, 3],  # intensity
                np.zeros(len(points), dtype=np.float32),  # time_lag = 0
            ])
        else:
            # Past sweep: apply ego-motion compensation
            sweep_ego_pose = nusc_db.get("ego_pose", sweep_sd["ego_pose_token"])
            sweep_cs = nusc_db.get(
                "calibrated_sensor", sweep_sd["calibrated_sensor_token"]
            )
            sweep_sensor_to_global = get_sensor_to_global_transform(
                sweep_ego_pose, sweep_cs
            )

            # Full transform: past_lidar -> global -> current_lidar
            past_to_current = global_to_current_lidar @ sweep_sensor_to_global

            # Apply transform to points (homogeneous coordinates)
            num_points = points.shape[0]
            xyz = points[:, :3]
            ones = np.ones((num_points, 1), dtype=np.float64)
            xyz_hom = np.hstack([xyz.astype(np.float64), ones])  # (N, 4)
            xyz_transformed = (past_to_current @ xyz_hom.T).T[:, :3]  # (N, 3)

            sweep_points = np.column_stack([
                xyz_transformed.astype(np.float32),
                points[:, 3],  # intensity preserved
                np.full(num_points, time_lag, dtype=np.float32),  # time_lag
            ])

        all_points.append(sweep_points)

    aggregated = np.concatenate(all_points, axis=0)
    return aggregated  # (N, 5): x, y, z, intensity, time_lag


# =============================================================================
# Ground Truth Box Extraction
# =============================================================================


def get_sample_annotations_in_lidar(
    nusc_db: NuScenesDatabase,
    sample_token: str,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Get all ground truth boxes for a sample in the LiDAR coordinate frame.

    Args:
        nusc_db: Loaded nuScenes database.
        sample_token: Token of the keyframe sample.

    Returns:
        gt_boxes: (M, 7) array with (x, y, z, w, l, h, yaw) in LiDAR frame.
        gt_names: List of M detection class names.
        gt_velocity: (M, 2) array with (vx, vy) in global frame.
    """
    sample = nusc_db.get("sample", sample_token)

    # Get transform from global to current lidar
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc_db.get("sample_data", lidar_token)
    ego_pose = nusc_db.get("ego_pose", lidar_sd["ego_pose_token"])
    cs = nusc_db.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    sensor_to_global = get_sensor_to_global_transform(ego_pose, cs)
    global_to_lidar = np.linalg.inv(sensor_to_global)

    gt_boxes_list = []
    gt_names_list = []
    gt_velocity_list = []

    for ann_token in sample["anns"]:
        ann = nusc_db.get("sample_annotation", ann_token)

        # Get detection class name
        category_name = nusc_db.get_category_name(ann["instance_token"])
        detection_name = CATEGORY_TO_DETECTION_NAME.get(category_name, None)
        if detection_name is None:
            continue
        if detection_name not in NUSCENES_CLASSES:
            continue

        # Transform box center to LiDAR frame
        center_global = np.array(ann["translation"])  # (3,)
        center_hom = np.append(center_global, 1.0)
        center_lidar = (global_to_lidar @ center_hom)[:3]

        # Box size: nuScenes uses (width, length, height)
        size = np.array(ann["size"])  # (w, l, h)

        # Compute yaw in LiDAR frame
        # nuScenes quaternion -> rotation matrix -> extract yaw
        ann_rotation = quaternion_to_rotation_matrix(np.array(ann["rotation"]))
        # Transform rotation to lidar frame
        lidar_rotation = global_to_lidar[:3, :3] @ ann_rotation
        # Extract yaw (rotation around z-axis)
        yaw = np.arctan2(lidar_rotation[1, 0], lidar_rotation[0, 0])

        # Box: (x, y, z, w, l, h, yaw)
        box = np.array([
            center_lidar[0], center_lidar[1], center_lidar[2],
            size[0], size[1], size[2], yaw
        ], dtype=np.float32)

        gt_boxes_list.append(box)
        gt_names_list.append(detection_name)

        # Compute velocity
        velocity = compute_box_velocity(nusc_db, sample_token, ann_token)
        gt_velocity_list.append(velocity)

    if len(gt_boxes_list) == 0:
        gt_boxes = np.zeros((0, 7), dtype=np.float32)
        gt_velocity = np.zeros((0, 2), dtype=np.float32)
    else:
        gt_boxes = np.stack(gt_boxes_list, axis=0)
        gt_velocity = np.stack(gt_velocity_list, axis=0)

    return gt_boxes, gt_names_list, gt_velocity


# =============================================================================
# Info File Creation
# =============================================================================


def create_sample_info(
    nusc_db: NuScenesDatabase,
    sample_token: str,
    data_root: str,
) -> Dict[str, Any]:
    """Create an info dict for a single sample.

    Contains:
        - lidar_path: relative path to LiDAR .bin file
        - lidar_points: aggregated point cloud (Nx5)
        - sweeps: list of sweep metadata dicts
        - gt_boxes: (M, 7) ground truth boxes
        - gt_names: list of M class names
        - gt_velocity: (M, 2) ground truth velocities
        - timestamp: sample timestamp in microseconds
        - scene_token: scene token
        - sample_token: sample token
    """
    sample = nusc_db.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc_db.get("sample_data", lidar_token)

    # Aggregate sweeps
    aggregated_points = aggregate_sweeps(nusc_db, sample_token, data_root)

    # Get sweep metadata
    sweeps_info = []
    sd = lidar_sd
    sweep_count = 0
    while sd["prev"] != "" and sweep_count < 9:
        sd = nusc_db.get("sample_data", sd["prev"])
        sweep_ego = nusc_db.get("ego_pose", sd["ego_pose_token"])
        sweep_cs = nusc_db.get("calibrated_sensor", sd["calibrated_sensor_token"])
        sweep_info = {
            "data_path": sd["filename"],
            "timestamp": sd["timestamp"],
            "ego_pose": {
                "translation": sweep_ego["translation"],
                "rotation": sweep_ego["rotation"],
            },
            "calibrated_sensor": {
                "translation": sweep_cs["translation"],
                "rotation": sweep_cs["rotation"],
            },
        }
        sweeps_info.append(sweep_info)
        sweep_count += 1

    # Get ground truth annotations
    gt_boxes, gt_names, gt_velocity = get_sample_annotations_in_lidar(
        nusc_db, sample_token
    )

    info = {
        "lidar_path": lidar_sd["filename"],
        "lidar_points": aggregated_points,
        "sweeps": sweeps_info,
        "gt_boxes": gt_boxes,
        "gt_names": gt_names,
        "gt_velocity": gt_velocity,
        "timestamp": sample["timestamp"],
        "scene_token": sample["scene_token"],
        "sample_token": sample_token,
    }

    return info


def process_sample_wrapper(args: Tuple[str, str, str, str]) -> Optional[Dict[str, Any]]:
    """Wrapper for multiprocessing that reconstructs the database connection.

    Since NuScenesDatabase is not easily picklable for multiprocessing,
    we pass the construction args and rebuild in each worker.
    """
    sample_token, data_root, version, table_root = args
    try:
        nusc_db = NuScenesDatabase(data_root, version)
        return create_sample_info(nusc_db, sample_token, data_root)
    except Exception as e:
        print(f"  ERROR processing sample {sample_token}: {e}")
        return None


def process_sample_single(
    nusc_db: NuScenesDatabase,
    sample_token: str,
    data_root: str,
) -> Optional[Dict[str, Any]]:
    """Process a single sample (used in single-threaded mode or pre-loaded db)."""
    try:
        return create_sample_info(nusc_db, sample_token, data_root)
    except Exception as e:
        print(f"  ERROR processing sample {sample_token}: {e}")
        return None


# =============================================================================
# GT Database Creation
# =============================================================================


def create_gt_database(
    infos: List[Dict[str, Any]],
    data_root: str,
    db_save_path: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Create ground truth database for GT-sampling augmentation.

    For each ground truth box in the training set, extracts all points inside
    the box and saves them as individual .bin files.

    Args:
        infos: List of training sample info dicts.
        data_root: Root path for nuScenes data.
        db_save_path: Directory to save individual GT point cloud files.

    Returns:
        gt_database_info: Dict mapping class name to list of GT database entries.
    """
    os.makedirs(db_save_path, exist_ok=True)

    gt_database_info: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_boxes = 0
    saved_boxes = 0

    for idx, info in enumerate(infos):
        if idx % 100 == 0:
            print(f"  Creating GT database: {idx}/{len(infos)} samples processed")

        points = info["lidar_points"]  # (N, 5)
        gt_boxes = info["gt_boxes"]  # (M, 7)
        gt_names = info["gt_names"]  # list of M names
        sample_token = info["sample_token"]

        for box_idx in range(len(gt_names)):
            total_boxes += 1
            class_name = gt_names[box_idx]
            box = gt_boxes[box_idx]  # (7,): x, y, z, w, l, h, yaw

            box_center = box[:3]
            box_size = box[3:6]  # w, l, h
            box_yaw = box[6]

            # Find points inside the box
            mask = points_in_box(points, box_center, box_size, box_yaw)
            box_points = points[mask]  # (K, 5)

            if box_points.shape[0] == 0:
                continue

            # Save points relative to box center (for later placement during augmentation)
            box_points_centered = box_points.copy()
            box_points_centered[:, :3] -= box_center

            # Save to file
            filename = f"{sample_token}_{class_name}_{box_idx}.bin"
            filepath = os.path.join(db_save_path, filename)
            box_points_centered.astype(np.float32).tofile(filepath)

            db_entry = {
                "path": os.path.join("gt_database", filename),
                "box": box,
                "num_points": box_points.shape[0],
                "class_name": class_name,
                "sample_token": sample_token,
            }
            gt_database_info[class_name].append(db_entry)
            saved_boxes += 1

    print(f"  GT database complete: {saved_boxes}/{total_boxes} boxes saved "
          f"({len(gt_database_info)} classes)")

    return dict(gt_database_info)


# =============================================================================
# Split Management
# =============================================================================


def get_split_scenes(version: str, split: str) -> List[str]:
    """Get list of scene names for a given split.

    Args:
        version: Dataset version ('v1.0-trainval' or 'v1.0-mini').
        split: Either 'train' or 'val'.

    Returns:
        List of scene name strings.
    """
    if version == "v1.0-trainval":
        if split == "train":
            return NUSCENES_TRAINVAL_TRAIN_SCENES
        elif split == "val":
            return NUSCENES_TRAINVAL_VAL_SCENES
        else:
            raise ValueError(f"Unknown split: {split}")
    elif version == "v1.0-mini":
        if split == "train":
            return NUSCENES_MINI_TRAIN_SCENES
        elif split == "val":
            return NUSCENES_MINI_VAL_SCENES
        else:
            raise ValueError(f"Unknown split: {split}")
    else:
        raise ValueError(f"Unknown version: {version}")


def get_samples_for_split(
    nusc_db: NuScenesDatabase,
    split_scenes: List[str],
) -> List[str]:
    """Get all sample tokens belonging to scenes in the given split.

    Navigates: scene -> first_sample_token -> iterate via next pointer.

    Args:
        nusc_db: Loaded database.
        split_scenes: List of scene name strings for the split.

    Returns:
        List of sample tokens in the split.
    """
    split_scene_set = set(split_scenes)
    sample_tokens = []

    for scene in nusc_db.get_records("scene"):
        if scene["name"] not in split_scene_set:
            continue

        # Iterate through all samples in this scene
        sample_token = scene["first_sample_token"]
        while sample_token != "":
            sample_tokens.append(sample_token)
            sample = nusc_db.get("sample", sample_token)
            sample_token = sample["next"]

    return sample_tokens


# =============================================================================
# Statistics
# =============================================================================


def print_statistics(
    train_infos: List[Dict[str, Any]],
    val_infos: List[Dict[str, Any]],
    gt_database_info: Optional[Dict[str, List[Dict[str, Any]]]] = None,
):
    """Print dataset statistics."""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    print(f"\n  Train samples: {len(train_infos)}")
    print(f"  Val samples:   {len(val_infos)}")
    print(f"  Total samples: {len(train_infos) + len(val_infos)}")

    # Class distribution for training set
    print("\n  Class Distribution (Training Set):")
    print("  " + "-" * 50)
    class_counts = defaultdict(int)
    for info in train_infos:
        for name in info["gt_names"]:
            class_counts[name] += 1

    total_boxes = sum(class_counts.values())
    for cls_name in NUSCENES_CLASSES:
        count = class_counts.get(cls_name, 0)
        pct = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"    {cls_name:<25s}: {count:>7d} ({pct:5.1f}%)")
    print(f"    {'TOTAL':<25s}: {total_boxes:>7d}")

    # Class distribution for validation set
    print("\n  Class Distribution (Validation Set):")
    print("  " + "-" * 50)
    val_class_counts = defaultdict(int)
    for info in val_infos:
        for name in info["gt_names"]:
            val_class_counts[name] += 1

    val_total_boxes = sum(val_class_counts.values())
    for cls_name in NUSCENES_CLASSES:
        count = val_class_counts.get(cls_name, 0)
        pct = (count / val_total_boxes * 100) if val_total_boxes > 0 else 0
        print(f"    {cls_name:<25s}: {count:>7d} ({pct:5.1f}%)")
    print(f"    {'TOTAL':<25s}: {val_total_boxes:>7d}")

    # Points per box histogram (training set)
    print("\n  Points per GT Box Histogram (Training Set):")
    print("  " + "-" * 50)
    points_per_box = []
    for info in train_infos:
        points = info["lidar_points"]
        gt_boxes = info["gt_boxes"]
        for box_idx in range(gt_boxes.shape[0]):
            box = gt_boxes[box_idx]
            mask = points_in_box(points, box[:3], box[3:6], box[6])
            points_per_box.append(mask.sum())

    if len(points_per_box) > 0:
        points_arr = np.array(points_per_box)
        bin_edges = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, float("inf")]
        bin_labels = [
            "0", "1-4", "5-9", "10-19", "20-49",
            "50-99", "100-199", "200-499", "500-999", "1000+"
        ]
        for i in range(len(bin_edges) - 1):
            mask = (points_arr >= bin_edges[i]) & (points_arr < bin_edges[i + 1])
            count = mask.sum()
            pct = count / len(points_arr) * 100
            print(f"    {bin_labels[i]:<12s} points: {count:>7d} boxes ({pct:5.1f}%)")

        print(f"\n    Mean points per box:   {points_arr.mean():.1f}")
        print(f"    Median points per box: {np.median(points_arr):.1f}")
        print(f"    Max points per box:    {points_arr.max()}")
        print(f"    Min points per box:    {points_arr.min()}")

    # GT database statistics
    if gt_database_info is not None:
        print("\n  GT Database Statistics:")
        print("  " + "-" * 50)
        for cls_name in NUSCENES_CLASSES:
            entries = gt_database_info.get(cls_name, [])
            if entries:
                num_points_list = [e["num_points"] for e in entries]
                mean_pts = np.mean(num_points_list)
                print(f"    {cls_name:<25s}: {len(entries):>7d} samples "
                      f"(mean {mean_pts:.0f} pts/box)")
            else:
                print(f"    {cls_name:<25s}: {0:>7d} samples")

    print("\n" + "=" * 70)


# =============================================================================
# Main Pipeline
# =============================================================================


def create_infos_for_split(
    nusc_db: NuScenesDatabase,
    sample_tokens: List[str],
    data_root: str,
    split_name: str,
    workers: int,
) -> List[Dict[str, Any]]:
    """Create info dicts for all samples in a split.

    Args:
        nusc_db: Loaded nuScenes database.
        sample_tokens: List of sample tokens to process.
        data_root: Root directory of nuScenes data.
        split_name: Name of the split for logging.
        workers: Number of parallel workers.

    Returns:
        List of info dicts.
    """
    print(f"\n  Processing {split_name} split: {len(sample_tokens)} samples "
          f"with {workers} workers...")
    start_time = time.time()

    infos = []

    if workers <= 1:
        # Single-threaded processing
        for i, sample_token in enumerate(sample_tokens):
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"    [{split_name}] {i}/{len(sample_tokens)} "
                      f"({rate:.1f} samples/s)")
            info = process_sample_single(nusc_db, sample_token, data_root)
            if info is not None:
                infos.append(info)
    else:
        # Multi-process: pass construction args since db is not picklable
        version = nusc_db.version
        table_root = nusc_db.table_root
        args_list = [
            (token, data_root, version, table_root)
            for token in sample_tokens
        ]

        with Pool(processes=workers) as pool:
            results = pool.map(process_sample_wrapper, args_list)

        for result in results:
            if result is not None:
                infos.append(result)

    elapsed = time.time() - start_time
    print(f"    [{split_name}] Complete: {len(infos)} infos created "
          f"in {elapsed:.1f}s ({len(infos)/elapsed:.1f} samples/s)")

    return infos


def main():
    parser = argparse.ArgumentParser(
        description="Prepare nuScenes data for CenterPoint training."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of nuScenes dataset (containing v1.0-trainval/ "
             "and samples/ directories).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        choices=["v1.0-trainval", "v1.0-mini"],
        help="Dataset version (default: v1.0-trainval).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of multiprocessing workers (default: 4). "
             "Set to 1 for single-threaded execution.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for info files. Defaults to <data-root>/infos/.",
    )

    args = parser.parse_args()

    data_root = args.data_root
    version = args.version
    workers = args.workers
    output_dir = args.output_dir if args.output_dir else os.path.join(data_root, "infos")

    print("=" * 70)
    print("CenterPoint nuScenes Data Preparation")
    print("=" * 70)
    print(f"  Data root:  {data_root}")
    print(f"  Version:    {version}")
    print(f"  Workers:    {workers}")
    print(f"  Output dir: {output_dir}")
    print("=" * 70)

    # Validate data root
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not os.path.isdir(os.path.join(data_root, version)):
        raise FileNotFoundError(
            f"Version directory not found: {os.path.join(data_root, version)}"
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load nuScenes database
    print("\n[1/5] Loading nuScenes database...")
    nusc_db = NuScenesDatabase(data_root, version)

    # Get train/val splits
    print("\n[2/5] Determining train/val splits...")
    train_scenes = get_split_scenes(version, "train")
    val_scenes = get_split_scenes(version, "val")

    train_sample_tokens = get_samples_for_split(nusc_db, train_scenes)
    val_sample_tokens = get_samples_for_split(nusc_db, val_scenes)

    print(f"  Train scenes: {len(train_scenes)}, samples: {len(train_sample_tokens)}")
    print(f"  Val scenes:   {len(val_scenes)}, samples: {len(val_sample_tokens)}")

    # Create info files
    print("\n[3/5] Creating info files with sweep aggregation...")

    train_infos = create_infos_for_split(
        nusc_db, train_sample_tokens, data_root, "train", workers
    )
    val_infos = create_infos_for_split(
        nusc_db, val_sample_tokens, data_root, "val", workers
    )

    # Save info files
    train_infos_path = os.path.join(output_dir, "train_infos.pkl")
    val_infos_path = os.path.join(output_dir, "val_infos.pkl")

    print(f"\n  Saving train infos to: {train_infos_path}")
    with open(train_infos_path, "wb") as f:
        pickle.dump(train_infos, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Saving val infos to: {val_infos_path}")
    with open(val_infos_path, "wb") as f:
        pickle.dump(val_infos, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Create GT database (training set only)
    print("\n[4/5] Creating GT database for GT-sampling augmentation...")
    gt_db_path = os.path.join(data_root, "gt_database")
    gt_database_info = create_gt_database(train_infos, data_root, gt_db_path)

    gt_db_info_path = os.path.join(output_dir, "gt_database_info.pkl")
    print(f"  Saving GT database info to: {gt_db_info_path}")
    with open(gt_db_info_path, "wb") as f:
        pickle.dump(gt_database_info, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Print statistics
    print("\n[5/5] Computing statistics...")
    print_statistics(train_infos, val_infos, gt_database_info)

    print("\nData preparation complete!")
    print(f"  Output files:")
    print(f"    {train_infos_path}")
    print(f"    {val_infos_path}")
    print(f"    {gt_db_info_path}")
    print(f"    {gt_db_path}/ ({sum(len(v) for v in gt_database_info.values())} files)")


if __name__ == "__main__":
    main()
