#!/usr/bin/env python3
"""
Prepare nuScenes data for BEVFormer training.

Parses nuScenes database JSON files and generates info pickle files with
temporal information, camera calibrations, ego poses, and annotations.

Usage:
    python prepare_data.py --data_root data/nuscenes --output_dir data/nuscenes/bevformer_infos
    python prepare_data.py --data_root data/nuscenes --version v1.0-mini --num_temporal_frames 4
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion


# nuScenes official train/val splits
NUSCENES_TRAINVAL_SCENES = {
    "train": [
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
        "scene-0075", "scene-0076", "scene-0120", "scene-0121", "scene-0122",
        "scene-0123", "scene-0124", "scene-0125", "scene-0126", "scene-0127",
        "scene-0128", "scene-0129", "scene-0130", "scene-0131", "scene-0132",
        "scene-0133", "scene-0134", "scene-0135", "scene-0138", "scene-0139",
        "scene-0149", "scene-0150", "scene-0151", "scene-0152", "scene-0154",
        "scene-0155", "scene-0157", "scene-0158", "scene-0159", "scene-0160",
        "scene-0161", "scene-0162", "scene-0163", "scene-0164", "scene-0165",
        "scene-0166", "scene-0167", "scene-0168", "scene-0170", "scene-0171",
        "scene-0172", "scene-0173", "scene-0174", "scene-0175", "scene-0176",
        "scene-0177", "scene-0178", "scene-0179", "scene-0180", "scene-0181",
        "scene-0182", "scene-0183", "scene-0184", "scene-0185", "scene-0187",
        "scene-0188", "scene-0190", "scene-0191", "scene-0192", "scene-0193",
        "scene-0194", "scene-0195", "scene-0196", "scene-0199", "scene-0200",
        "scene-0202", "scene-0203", "scene-0204", "scene-0206", "scene-0207",
        "scene-0208", "scene-0209", "scene-0210", "scene-0211", "scene-0212",
        "scene-0213", "scene-0214", "scene-0218", "scene-0219", "scene-0220",
        "scene-0222", "scene-0224", "scene-0225", "scene-0226", "scene-0227",
        "scene-0228", "scene-0229", "scene-0230", "scene-0231", "scene-0232",
        "scene-0233", "scene-0234", "scene-0235", "scene-0236", "scene-0237",
        "scene-0238", "scene-0239", "scene-0240", "scene-0241", "scene-0242",
        "scene-0243", "scene-0244", "scene-0245", "scene-0246", "scene-0247",
        "scene-0248", "scene-0249", "scene-0250", "scene-0251", "scene-0252",
        "scene-0253", "scene-0254", "scene-0255", "scene-0256", "scene-0257",
        "scene-0258", "scene-0259", "scene-0260", "scene-0261", "scene-0262",
        "scene-0263", "scene-0264", "scene-0283", "scene-0284", "scene-0285",
        "scene-0286", "scene-0287", "scene-0288", "scene-0289", "scene-0290",
        "scene-0291", "scene-0292", "scene-0293", "scene-0294", "scene-0295",
        "scene-0296", "scene-0297", "scene-0298", "scene-0299", "scene-0300",
        "scene-0301", "scene-0302", "scene-0303", "scene-0304", "scene-0305",
        "scene-0306", "scene-0315", "scene-0316", "scene-0317", "scene-0318",
        "scene-0321", "scene-0323", "scene-0324", "scene-0328", "scene-0347",
        "scene-0348", "scene-0349", "scene-0350", "scene-0351", "scene-0352",
        "scene-0353", "scene-0354", "scene-0355", "scene-0356", "scene-0357",
        "scene-0358", "scene-0359", "scene-0360", "scene-0361", "scene-0362",
        "scene-0363", "scene-0364", "scene-0365", "scene-0366", "scene-0367",
        "scene-0368", "scene-0369", "scene-0370", "scene-0371", "scene-0372",
        "scene-0373", "scene-0374", "scene-0375", "scene-0376", "scene-0377",
        "scene-0378", "scene-0379", "scene-0380", "scene-0381", "scene-0382",
        "scene-0383", "scene-0384", "scene-0385", "scene-0386", "scene-0388",
        "scene-0389", "scene-0390", "scene-0391", "scene-0392", "scene-0393",
        "scene-0394", "scene-0395", "scene-0396", "scene-0397", "scene-0398",
        "scene-0399", "scene-0400", "scene-0401", "scene-0402", "scene-0403",
        "scene-0405", "scene-0406", "scene-0407", "scene-0408", "scene-0410",
        "scene-0411", "scene-0412", "scene-0413", "scene-0414", "scene-0415",
        "scene-0416", "scene-0417", "scene-0418", "scene-0419", "scene-0420",
        "scene-0421", "scene-0422", "scene-0423", "scene-0424", "scene-0425",
        "scene-0426", "scene-0427", "scene-0428", "scene-0429", "scene-0430",
        "scene-0431", "scene-0432", "scene-0433", "scene-0434", "scene-0435",
        "scene-0436", "scene-0437", "scene-0438", "scene-0439", "scene-0440",
        "scene-0441", "scene-0442", "scene-0443", "scene-0444", "scene-0445",
        "scene-0446", "scene-0447", "scene-0448", "scene-0449", "scene-0450",
        "scene-0451", "scene-0452", "scene-0453", "scene-0454", "scene-0455",
        "scene-0456", "scene-0457", "scene-0458", "scene-0459", "scene-0461",
        "scene-0462", "scene-0463", "scene-0464", "scene-0465", "scene-0467",
        "scene-0468", "scene-0469", "scene-0471", "scene-0472", "scene-0474",
        "scene-0475", "scene-0476", "scene-0477", "scene-0478", "scene-0479",
        "scene-0480", "scene-0499", "scene-0500", "scene-0501", "scene-0502",
        "scene-0504", "scene-0505", "scene-0506", "scene-0507", "scene-0508",
        "scene-0509", "scene-0510", "scene-0511", "scene-0512", "scene-0513",
        "scene-0514", "scene-0515", "scene-0517", "scene-0518", "scene-0525",
        "scene-0526", "scene-0527", "scene-0528", "scene-0529", "scene-0530",
        "scene-0531", "scene-0532", "scene-0533", "scene-0534", "scene-0535",
        "scene-0536", "scene-0537", "scene-0538", "scene-0539", "scene-0541",
        "scene-0542", "scene-0543", "scene-0544", "scene-0545", "scene-0546",
        "scene-0566", "scene-0568", "scene-0570", "scene-0571", "scene-0572",
        "scene-0573", "scene-0574", "scene-0575", "scene-0576", "scene-0577",
        "scene-0578", "scene-0580", "scene-0582", "scene-0583", "scene-0584",
        "scene-0585", "scene-0586", "scene-0587", "scene-0588", "scene-0589",
        "scene-0590", "scene-0591", "scene-0592", "scene-0593", "scene-0594",
        "scene-0595", "scene-0596", "scene-0597", "scene-0598", "scene-0599",
        "scene-0600", "scene-0639", "scene-0640", "scene-0641", "scene-0642",
        "scene-0643", "scene-0644", "scene-0645", "scene-0646", "scene-0647",
        "scene-0648", "scene-0649", "scene-0650", "scene-0651", "scene-0652",
        "scene-0653", "scene-0654", "scene-0655", "scene-0656", "scene-0657",
        "scene-0658", "scene-0659", "scene-0660", "scene-0661", "scene-0662",
        "scene-0663", "scene-0664", "scene-0665", "scene-0666", "scene-0667",
        "scene-0668", "scene-0669", "scene-0670", "scene-0671", "scene-0672",
        "scene-0673", "scene-0674", "scene-0675", "scene-0676", "scene-0677",
        "scene-0678", "scene-0679", "scene-0681", "scene-0683", "scene-0684",
        "scene-0685", "scene-0686", "scene-0687", "scene-0688", "scene-0689",
        "scene-0695", "scene-0696", "scene-0697", "scene-0698", "scene-0700",
        "scene-0701", "scene-0703", "scene-0704", "scene-0705", "scene-0706",
        "scene-0707", "scene-0708", "scene-0709", "scene-0710", "scene-0711",
        "scene-0712", "scene-0713", "scene-0714", "scene-0715", "scene-0716",
        "scene-0717", "scene-0718", "scene-0719", "scene-0726", "scene-0727",
        "scene-0728", "scene-0730", "scene-0731", "scene-0733", "scene-0734",
        "scene-0735", "scene-0736", "scene-0737", "scene-0738", "scene-0739",
        "scene-0740", "scene-0741", "scene-0744", "scene-0746", "scene-0747",
        "scene-0749", "scene-0750", "scene-0751", "scene-0752", "scene-0757",
        "scene-0758", "scene-0759", "scene-0760", "scene-0761", "scene-0762",
        "scene-0763", "scene-0764", "scene-0765", "scene-0767", "scene-0768",
        "scene-0769", "scene-0786", "scene-0787", "scene-0789", "scene-0790",
        "scene-0791", "scene-0792", "scene-0803", "scene-0804", "scene-0805",
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
    ],
    "val": [
        "scene-0003", "scene-0012", "scene-0013", "scene-0014", "scene-0015",
        "scene-0016", "scene-0017", "scene-0018", "scene-0035", "scene-0036",
        "scene-0037", "scene-0038", "scene-0039", "scene-0040", "scene-0077",
        "scene-0078", "scene-0079", "scene-0080", "scene-0081", "scene-0082",
        "scene-0083", "scene-0084", "scene-0085", "scene-0086", "scene-0087",
        "scene-0088", "scene-0089", "scene-0090", "scene-0091", "scene-0092",
        "scene-0093", "scene-0094", "scene-0095", "scene-0096", "scene-0097",
        "scene-0098", "scene-0099", "scene-0100", "scene-0101", "scene-0102",
        "scene-0103", "scene-0104", "scene-0105", "scene-0106", "scene-0107",
        "scene-0108", "scene-0109", "scene-0110", "scene-0221", "scene-0268",
        "scene-0269", "scene-0270", "scene-0271", "scene-0272", "scene-0273",
        "scene-0274", "scene-0275", "scene-0276", "scene-0277", "scene-0278",
        "scene-0329", "scene-0330", "scene-0331", "scene-0332", "scene-0344",
        "scene-0345", "scene-0346", "scene-0519", "scene-0520", "scene-0521",
        "scene-0522", "scene-0523", "scene-0524", "scene-0552", "scene-0553",
        "scene-0554", "scene-0555", "scene-0556", "scene-0557", "scene-0558",
        "scene-0559", "scene-0560", "scene-0561", "scene-0562", "scene-0563",
        "scene-0564", "scene-0565", "scene-0625", "scene-0626", "scene-0627",
        "scene-0629", "scene-0630", "scene-0632", "scene-0633", "scene-0634",
        "scene-0635", "scene-0636", "scene-0637", "scene-0638", "scene-0770",
        "scene-0771", "scene-0775", "scene-0777", "scene-0778", "scene-0780",
        "scene-0781", "scene-0782", "scene-0783", "scene-0784", "scene-0794",
        "scene-0795", "scene-0796", "scene-0797", "scene-0798", "scene-0799",
        "scene-0800", "scene-0802", "scene-0904", "scene-0905", "scene-0906",
        "scene-0907", "scene-0908", "scene-0909", "scene-0910", "scene-0911",
        "scene-0912", "scene-0913", "scene-0914", "scene-0915", "scene-0916",
        "scene-0917", "scene-0918", "scene-0919", "scene-0920", "scene-0921",
        "scene-0922", "scene-0923", "scene-0924", "scene-0925", "scene-0926",
        "scene-0927", "scene-0928", "scene-0929", "scene-0930", "scene-0931",
        "scene-0962", "scene-0963", "scene-0966", "scene-0967", "scene-0968",
        "scene-0969", "scene-0971", "scene-1059", "scene-1060", "scene-1061",
        "scene-1062", "scene-1063", "scene-1064", "scene-1065", "scene-1066",
        "scene-1067", "scene-1068", "scene-1069", "scene-1070", "scene-1071",
        "scene-1072", "scene-1073",
    ],
}

# Mini split scenes
NUSCENES_MINI_SCENES = {
    "train": [
        "scene-0061", "scene-0553", "scene-0655", "scene-0757",
        "scene-0796", "scene-1077", "scene-1094", "scene-1100",
    ],
    "val": [
        "scene-0103", "scene-0916",
    ],
}

# nuScenes camera names
CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# nuScenes detection categories for BEVFormer
DETECTION_CATEGORIES = [
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

# Category name mapping from nuScenes full names to detection names
CATEGORY_MAP = {
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
    "human.pedestrian.wheelchair": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
}


class NuScenesDatabase:
    """Lightweight nuScenes database parser without the full devkit dependency."""

    def __init__(self, data_root: str, version: str = "v1.0-trainval"):
        self.data_root = data_root
        self.version = version
        self.table_root = os.path.join(data_root, version)

        # Load all tables
        self.scene = self._load_table("scene")
        self.sample = self._load_table("sample")
        self.sample_data = self._load_table("sample_data")
        self.ego_pose = self._load_table("ego_pose")
        self.calibrated_sensor = self._load_table("calibrated_sensor")
        self.sensor = self._load_table("sensor")
        self.sample_annotation = self._load_table("sample_annotation")
        self.instance = self._load_table("instance")
        self.category = self._load_table("category")
        self.attribute = self._load_table("attribute")
        self.log = self._load_table("log")
        self.map = self._load_table("map")

        # Build lookup indices
        self._build_indices()

    def _load_table(self, table_name: str) -> List[Dict]:
        """Load a nuScenes JSON table."""
        filepath = os.path.join(self.table_root, f"{table_name}.json")
        if not os.path.exists(filepath):
            print(f"[WARNING] Table not found: {filepath}")
            return []
        with open(filepath, "r") as f:
            return json.load(f)

    def _build_indices(self):
        """Build token-to-record lookup dictionaries for fast access."""
        self.scene_by_token = {r["token"]: r for r in self.scene}
        self.sample_by_token = {r["token"]: r for r in self.sample}
        self.sample_data_by_token = {r["token"]: r for r in self.sample_data}
        self.ego_pose_by_token = {r["token"]: r for r in self.ego_pose}
        self.calibrated_sensor_by_token = {r["token"]: r for r in self.calibrated_sensor}
        self.sensor_by_token = {r["token"]: r for r in self.sensor}
        self.instance_by_token = {r["token"]: r for r in self.instance}
        self.category_by_token = {r["token"]: r for r in self.category}
        self.attribute_by_token = {r["token"]: r for r in self.attribute}

        # Build scene name to token mapping
        self.scene_name_to_token = {r["name"]: r["token"] for r in self.scene}

        # Build sample to sample_data mapping (for each sensor channel)
        self.sample_to_sample_data = defaultdict(list)
        for sd in self.sample_data:
            if sd["is_key_frame"]:
                self.sample_to_sample_data[sd["sample_token"]].append(sd)

        # Build sample to annotations mapping
        self.sample_to_annotations = defaultdict(list)
        for ann in self.sample_annotation:
            self.sample_to_annotations[ann["sample_token"]].append(ann)

        # Build scene to samples ordered list
        self.scene_to_samples = defaultdict(list)
        for scene in self.scene:
            sample_token = scene["first_sample_token"]
            while sample_token:
                self.scene_to_samples[scene["token"]].append(sample_token)
                sample_rec = self.sample_by_token[sample_token]
                sample_token = sample_rec["next"] if sample_rec["next"] != "" else None


def quaternion_to_rotation_matrix(quaternion: List[float]) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    q = Quaternion(quaternion)
    return q.rotation_matrix


def make_transform_matrix(translation: List[float], rotation: List[float]) -> np.ndarray:
    """Create a 4x4 transformation matrix from translation and rotation (quaternion)."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(rotation)
    transform[:3, 3] = np.array(translation)
    return transform


def get_sensor_transform(db: NuScenesDatabase, sample_data_token: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the ego pose and sensor calibration for a given sample_data record.

    Returns:
        ego2global: 4x4 ego vehicle to global transform
        sensor2ego: 4x4 sensor to ego vehicle transform
    """
    sd_record = db.sample_data_by_token[sample_data_token]
    ego_record = db.ego_pose_by_token[sd_record["ego_pose_token"]]
    cs_record = db.calibrated_sensor_by_token[sd_record["calibrated_sensor_token"]]

    ego2global = make_transform_matrix(
        ego_record["translation"], ego_record["rotation"]
    )
    sensor2ego = make_transform_matrix(
        cs_record["translation"], cs_record["rotation"]
    )

    return ego2global, sensor2ego


def get_camera_intrinsics(db: NuScenesDatabase, sample_data_token: str) -> np.ndarray:
    """Get 3x3 camera intrinsic matrix."""
    sd_record = db.sample_data_by_token[sample_data_token]
    cs_record = db.calibrated_sensor_by_token[sd_record["calibrated_sensor_token"]]
    return np.array(cs_record["camera_intrinsic"], dtype=np.float64)


def get_lidar_sample_data(db: NuScenesDatabase, sample_token: str) -> Optional[Dict]:
    """Get the LIDAR_TOP sample_data record for a given sample."""
    for sd in db.sample_to_sample_data[sample_token]:
        sensor_record = db.sensor_by_token[
            db.calibrated_sensor_by_token[sd["calibrated_sensor_token"]]["sensor_token"]
        ]
        if sensor_record["channel"] == "LIDAR_TOP":
            return sd
    return None


def get_camera_sample_data(db: NuScenesDatabase, sample_token: str) -> Dict[str, Dict]:
    """Get camera sample_data records for all 6 cameras for a given sample."""
    cameras = {}
    for sd in db.sample_to_sample_data[sample_token]:
        sensor_record = db.sensor_by_token[
            db.calibrated_sensor_by_token[sd["calibrated_sensor_token"]]["sensor_token"]
        ]
        if sensor_record["modality"] == "camera":
            cameras[sensor_record["channel"]] = sd
    return cameras


def compute_lidar2img(
    lidar2ego: np.ndarray,
    ego2global_lidar: np.ndarray,
    ego2global_cam: np.ndarray,
    cam2ego: np.ndarray,
    cam_intrinsic: np.ndarray,
) -> np.ndarray:
    """
    Compute the lidar-to-image projection matrix.

    lidar -> ego (lidar time) -> global -> ego (cam time) -> cam -> image

    Returns:
        4x4 lidar2img projection matrix
    """
    # lidar to global
    lidar2global = ego2global_lidar @ lidar2ego

    # global to cam ego
    global2ego_cam = np.linalg.inv(ego2global_cam)

    # cam ego to cam
    ego2cam = np.linalg.inv(cam2ego)

    # Full transform: lidar -> global -> cam_ego -> cam
    lidar2cam = ego2cam @ global2ego_cam @ lidar2global

    # Add intrinsics: cam 3D -> image 2D (as 4x4)
    intrinsic_4x4 = np.eye(4, dtype=np.float64)
    intrinsic_4x4[:3, :3] = cam_intrinsic

    lidar2img = intrinsic_4x4 @ lidar2cam
    return lidar2img


def get_annotation_info(
    db: NuScenesDatabase,
    sample_token: str,
    ego2global: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Get annotation information for a sample.

    Returns list of dicts with:
        - bbox_3d: [cx, cy, cz, w, l, h, yaw] in ego frame
        - velocity: [vx, vy] in ego frame
        - category: detection category name
        - attribute: attribute name
        - instance_token: instance token
        - num_lidar_pts: number of lidar points in box
    """
    annotations = []
    global2ego = np.linalg.inv(ego2global)

    for ann in db.sample_to_annotations[sample_token]:
        # Get category
        instance = db.instance_by_token[ann["instance_token"]]
        category = db.category_by_token[instance["category_token"]]
        cat_name = category["name"]

        # Map to detection category
        det_category = None
        for prefix, det_name in CATEGORY_MAP.items():
            if cat_name.startswith(prefix):
                det_category = det_name
                break
        if det_category is None:
            continue  # Skip non-detection categories

        # Transform center to ego frame
        center_global = np.array([*ann["translation"], 1.0])
        center_ego = global2ego @ center_global

        # Transform rotation to ego frame
        global_rotation = Quaternion(ann["rotation"])
        ego_rotation_quat = Quaternion(matrix=global2ego[:3, :3])
        box_rotation_ego = ego_rotation_quat * global_rotation
        yaw = box_rotation_ego.yaw_pitch_roll[0]

        # Size: [width, length, height] in nuScenes
        w, l, h = ann["size"]

        # Velocity (if available)
        if ann.get("velocity") is not None and len(ann["velocity"]) >= 2:
            velocity_global = np.array([ann["velocity"][0], ann["velocity"][1], 0.0, 0.0])
            velocity_ego = global2ego @ velocity_global
            vx, vy = velocity_ego[0], velocity_ego[1]
        else:
            vx, vy = 0.0, 0.0

        # Attribute
        attr_name = ""
        if ann.get("attribute_tokens") and len(ann["attribute_tokens"]) > 0:
            attr_record = db.attribute_by_token.get(ann["attribute_tokens"][0])
            if attr_record:
                attr_name = attr_record["name"]

        annotations.append({
            "bbox_3d": [
                float(center_ego[0]),
                float(center_ego[1]),
                float(center_ego[2]),
                float(w),
                float(l),
                float(h),
                float(yaw),
            ],
            "velocity": [float(vx), float(vy)],
            "category": det_category,
            "attribute": attr_name,
            "instance_token": ann["instance_token"],
            "num_lidar_pts": ann.get("num_lidar_pts", 0),
        })

    return annotations


def get_temporal_info(
    db: NuScenesDatabase,
    sample_token: str,
    num_temporal_frames: int,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Get previous sample tokens and their relative ego-motion transforms.

    For each previous sample, computes the transform from the previous
    ego frame to the current ego frame.

    Returns:
        prev_tokens: list of previous sample tokens (most recent first)
        ego_transforms: list of 4x4 transforms (prev_ego -> current_ego)
    """
    # Get current ego pose
    lidar_sd = get_lidar_sample_data(db, sample_token)
    if lidar_sd is None:
        return [], []

    current_ego2global, _ = get_sensor_transform(db, lidar_sd["token"])
    global2current_ego = np.linalg.inv(current_ego2global)

    prev_tokens = []
    ego_transforms = []

    current_sample = db.sample_by_token[sample_token]
    prev_sample_token = current_sample["prev"] if current_sample["prev"] != "" else None

    for _ in range(num_temporal_frames):
        if prev_sample_token is None:
            # Pad with current sample if no previous available
            prev_tokens.append(sample_token)
            ego_transforms.append(np.eye(4, dtype=np.float64))
        else:
            prev_tokens.append(prev_sample_token)

            # Get prev ego pose
            prev_lidar_sd = get_lidar_sample_data(db, prev_sample_token)
            if prev_lidar_sd is not None:
                prev_ego2global, _ = get_sensor_transform(db, prev_lidar_sd["token"])
                # Transform: prev_ego -> global -> current_ego
                prev2current = global2current_ego @ prev_ego2global
                ego_transforms.append(prev2current)
            else:
                ego_transforms.append(np.eye(4, dtype=np.float64))

            # Move to next previous
            prev_sample = db.sample_by_token[prev_sample_token]
            prev_sample_token = prev_sample["prev"] if prev_sample["prev"] != "" else None

    return prev_tokens, ego_transforms


def create_bev_grid(
    x_range: Tuple[float, float] = (-51.2, 51.2),
    y_range: Tuple[float, float] = (-51.2, 51.2),
    grid_size: Tuple[int, int] = (200, 200),
) -> Dict[str, np.ndarray]:
    """
    Create BEV grid coordinates.

    The grid covers x_range x y_range meters with grid_size resolution.
    Each cell center is computed and stored.

    Returns:
        Dictionary with grid coordinates and metadata.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    nx, ny = grid_size

    # Cell size
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    # Cell centers
    x_centers = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)
    y_centers = np.linspace(y_min + dy / 2, y_max - dy / 2, ny)

    # Create meshgrid (200x200x2) of (x,y) coordinates
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    grid_coords = np.stack([xx, yy], axis=-1)  # (200, 200, 2)

    # Also create the full 3D grid at z=0 for projection purposes
    zz = np.zeros_like(xx)
    grid_coords_3d = np.stack([xx, yy, zz], axis=-1)  # (200, 200, 3)

    return {
        "grid_coords_2d": grid_coords,
        "grid_coords_3d": grid_coords_3d,
        "x_range": x_range,
        "y_range": y_range,
        "grid_size": grid_size,
        "resolution": (dx, dy),
        "x_centers": x_centers,
        "y_centers": y_centers,
    }


def process_sample(
    db: NuScenesDatabase,
    sample_token: str,
    num_temporal_frames: int,
) -> Optional[Dict[str, Any]]:
    """
    Process a single sample and generate its info dictionary.

    Returns None if the sample cannot be processed (missing data).
    """
    sample = db.sample_by_token[sample_token]

    # Get LIDAR_TOP sample data for reference frame
    lidar_sd = get_lidar_sample_data(db, sample_token)
    if lidar_sd is None:
        print(f"[WARNING] No LIDAR_TOP data for sample {sample_token}, skipping.")
        return None

    # Get ego pose at lidar timestamp (reference)
    ego2global_lidar, lidar2ego = get_sensor_transform(db, lidar_sd["token"])

    # Get all camera sample data
    camera_sds = get_camera_sample_data(db, sample_token)

    # Process each camera
    camera_infos = []
    lidar2img_list = []

    for cam_name in CAMERA_NAMES:
        if cam_name not in camera_sds:
            print(f"[WARNING] Missing camera {cam_name} for sample {sample_token}")
            return None

        cam_sd = camera_sds[cam_name]

        # Get camera calibration and ego pose
        ego2global_cam, cam2ego = get_sensor_transform(db, cam_sd["token"])
        cam_intrinsic = get_camera_intrinsics(db, cam_sd["token"])

        # Compute lidar2cam transform
        # lidar -> ego (lidar) -> global -> ego (cam) -> cam
        lidar2global = ego2global_lidar @ lidar2ego
        global2cam_ego = np.linalg.inv(ego2global_cam)
        ego2cam = np.linalg.inv(cam2ego)
        lidar2cam = ego2cam @ global2cam_ego @ lidar2global

        # Compute cam2img (3x4 projection from 3D cam coords to 2D image)
        cam2img = np.zeros((4, 4), dtype=np.float64)
        cam2img[:3, :3] = cam_intrinsic
        cam2img[3, 3] = 1.0

        # Compute lidar2img
        lidar2img = compute_lidar2img(
            lidar2ego, ego2global_lidar, ego2global_cam, cam2ego, cam_intrinsic
        )

        camera_infos.append({
            "filename": cam_sd["filename"],
            "cam_name": cam_name,
            "cam_intrinsic": cam_intrinsic.tolist(),
            "cam2ego": cam2ego.tolist(),
            "ego2global": ego2global_cam.tolist(),
            "lidar2cam": lidar2cam.tolist(),
            "cam2img": cam2img.tolist(),
            "timestamp": cam_sd["timestamp"],
            "width": cam_sd.get("width", 1600),
            "height": cam_sd.get("height", 900),
        })

        lidar2img_list.append(lidar2img.tolist())

    # Get annotations
    annotations = get_annotation_info(db, sample_token, ego2global_lidar)

    # Get temporal info
    prev_tokens, ego_transforms = get_temporal_info(
        db, sample_token, num_temporal_frames
    )

    # Build info dict
    info = {
        "token": sample_token,
        "timestamp": sample["timestamp"],
        "scene_token": sample["scene_token"],
        "lidar_filename": lidar_sd["filename"],
        "lidar2ego": lidar2ego.tolist(),
        "ego2global": ego2global_lidar.tolist(),
        "cameras": camera_infos,
        "lidar2img": lidar2img_list,
        "annotations": annotations,
        "num_annotations": len(annotations),
        "temporal": {
            "prev_tokens": prev_tokens,
            "ego_transforms": [t.tolist() for t in ego_transforms],
            "num_temporal_frames": num_temporal_frames,
        },
    }

    return info


def get_scene_split(
    db: NuScenesDatabase,
    version: str,
) -> Tuple[List[str], List[str]]:
    """Get train/val sample token lists based on official splits."""
    if version == "v1.0-mini":
        splits = NUSCENES_MINI_SCENES
    else:
        splits = NUSCENES_TRAINVAL_SCENES

    train_tokens = []
    val_tokens = []

    for scene in db.scene:
        scene_name = scene["name"]
        if scene_name in splits["train"]:
            # Get all samples in this scene
            for sample_token in db.scene_to_samples[scene["token"]]:
                train_tokens.append(sample_token)
        elif scene_name in splits["val"]:
            for sample_token in db.scene_to_samples[scene["token"]]:
                val_tokens.append(sample_token)

    return train_tokens, val_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Prepare nuScenes data for BEVFormer training."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to nuScenes dataset root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for info files. Defaults to data_root.",
    )
    parser.add_argument(
        "--num_temporal_frames",
        type=int,
        default=4,
        help="Number of previous temporal frames to include (default: 4).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        choices=["v1.0-trainval", "v1.0-mini"],
        help="nuScenes dataset version (default: v1.0-trainval).",
    )

    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.data_root
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  nuScenes Data Preparation for BEVFormer")
    print("=" * 60)
    print(f"  Data root:         {args.data_root}")
    print(f"  Output dir:        {output_dir}")
    print(f"  Version:           {args.version}")
    print(f"  Temporal frames:   {args.num_temporal_frames}")
    print("=" * 60)

    # Load database
    print("\n[1/5] Loading nuScenes database...")
    db = NuScenesDatabase(args.data_root, args.version)
    print(f"  Loaded {len(db.scene)} scenes, {len(db.sample)} samples, "
          f"{len(db.sample_annotation)} annotations")

    # Get train/val splits
    print("\n[2/5] Splitting into train/val...")
    train_tokens, val_tokens = get_scene_split(db, args.version)
    print(f"  Train samples: {len(train_tokens)}")
    print(f"  Val samples:   {len(val_tokens)}")

    # Process train samples
    print(f"\n[3/5] Processing train samples ({len(train_tokens)} total)...")
    train_infos = []
    for i, token in enumerate(train_tokens):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing train sample {i + 1}/{len(train_tokens)}...")
        info = process_sample(db, token, args.num_temporal_frames)
        if info is not None:
            train_infos.append(info)

    print(f"  Successfully processed {len(train_infos)}/{len(train_tokens)} train samples")

    # Process val samples
    print(f"\n[4/5] Processing val samples ({len(val_tokens)} total)...")
    val_infos = []
    for i, token in enumerate(val_tokens):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing val sample {i + 1}/{len(val_tokens)}...")
        info = process_sample(db, token, args.num_temporal_frames)
        if info is not None:
            val_infos.append(info)

    print(f"  Successfully processed {len(val_infos)}/{len(val_tokens)} val samples")

    # Create BEV grid
    print("\n[5/5] Creating BEV grid coordinates...")
    bev_grid = create_bev_grid(
        x_range=(-51.2, 51.2),
        y_range=(-51.2, 51.2),
        grid_size=(200, 200),
    )
    print(f"  Grid size: {bev_grid['grid_size']}")
    print(f"  X range: {bev_grid['x_range']} m")
    print(f"  Y range: {bev_grid['y_range']} m")
    print(f"  Resolution: {bev_grid['resolution'][0]:.4f} m/cell")

    # Save info files
    train_info_path = os.path.join(output_dir, "nuscenes_infos_temporal_train.pkl")
    val_info_path = os.path.join(output_dir, "nuscenes_infos_temporal_val.pkl")
    bev_grid_path = os.path.join(output_dir, "bev_grid_coords.pkl")

    print(f"\n  Saving train infos to: {train_info_path}")
    with open(train_info_path, "wb") as f:
        pickle.dump({
            "infos": train_infos,
            "metadata": {
                "version": args.version,
                "num_temporal_frames": args.num_temporal_frames,
                "num_samples": len(train_infos),
                "split": "train",
                "categories": DETECTION_CATEGORIES,
            },
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Saving val infos to: {val_info_path}")
    with open(val_info_path, "wb") as f:
        pickle.dump({
            "infos": val_infos,
            "metadata": {
                "version": args.version,
                "num_temporal_frames": args.num_temporal_frames,
                "num_samples": len(val_infos),
                "split": "val",
                "categories": DETECTION_CATEGORIES,
            },
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Saving BEV grid to: {bev_grid_path}")
    with open(bev_grid_path, "wb") as f:
        pickle.dump(bev_grid, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Print summary
    print("\n" + "=" * 60)
    print("  Preparation Complete!")
    print("=" * 60)
    print(f"  Train info: {train_info_path}")
    print(f"    Samples: {len(train_infos)}")
    print(f"  Val info:   {val_info_path}")
    print(f"    Samples: {len(val_infos)}")
    print(f"  BEV grid:   {bev_grid_path}")
    print(f"    Grid shape: {bev_grid['grid_coords_2d'].shape}")

    # Annotation statistics
    total_anns_train = sum(info["num_annotations"] for info in train_infos)
    total_anns_val = sum(info["num_annotations"] for info in val_infos)
    print(f"\n  Total annotations (train): {total_anns_train}")
    print(f"  Total annotations (val):   {total_anns_val}")

    # Per-category counts
    cat_counts = defaultdict(int)
    for info in train_infos + val_infos:
        for ann in info["annotations"]:
            cat_counts[ann["category"]] += 1

    print("\n  Category distribution:")
    for cat in DETECTION_CATEGORIES:
        print(f"    {cat:25s}: {cat_counts.get(cat, 0):>8d}")

    print("\n  Done!")


if __name__ == "__main__":
    main()
