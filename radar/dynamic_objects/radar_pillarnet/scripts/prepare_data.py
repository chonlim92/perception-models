"""
Radar data preprocessing script for nuScenes dataset.

Prepares radar point clouds for PillarNet-based 3D object detection:
- Parses raw radar PCD files into structured numpy arrays
- Accumulates multi-sweep radar data with ego-motion compensation
- Generates train/val info pickle files
- Creates GT database for data augmentation (GT-sampling)
"""

import argparse
import os
import pickle
import struct
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from tqdm import tqdm

# All 5 radar sensors in nuScenes
RADAR_SENSORS = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]

# nuScenes radar point cloud features (18 fields)
RADAR_FEATURES = [
    "x",
    "y",
    "z",
    "dyn_prop",
    "id",
    "rcs",
    "vx",
    "vy",
    "vx_comp",
    "vy_comp",
    "is_quality_valid",
    "ambig_state",
    "x_rms",
    "y_rms",
    "invalid_state",
    "pdh0",
    "vx_rms",
    "vy_rms",
]

# nuScenes detection class mapping
DETECTION_NAMES = [
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

# Map from nuScenes general category to detection name
CATEGORY_TO_DETECTION = {
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


def get_radar_point_cloud(nusc, sample_data_token):
    """
    Load a single radar PCD file and parse the binary format.

    nuScenes radar PCD files use a binary format with 18 float32 fields per point.

    Args:
        nusc: NuScenes instance
        sample_data_token: Token for the sample_data record

    Returns:
        numpy array of shape (N, 18) with all radar features, or empty (0, 18) array
    """
    sample_data = nusc.get("sample_data", sample_data_token)
    pcd_path = os.path.join(nusc.dataroot, sample_data["filename"])

    # Parse PCD file header and binary data
    points = _parse_radar_pcd(pcd_path)
    return points


def _parse_radar_pcd(pcd_path):
    """
    Parse a nuScenes radar PCD file (binary format).

    The PCD file has a text header followed by binary point data.
    Each point has 18 float32 fields.

    Args:
        pcd_path: Path to the .pcd file

    Returns:
        numpy array of shape (N, 18)
    """
    with open(pcd_path, "rb") as f:
        # Read header lines until DATA line
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line.startswith("DATA"):
                break

        # Parse header for point count and format
        num_points = 0
        fields = []
        sizes = []
        types = []
        for line in header_lines:
            if line.startswith("POINTS"):
                num_points = int(line.split()[-1])
            elif line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("SIZE"):
                sizes = [int(s) for s in line.split()[1:]]
            elif line.startswith("TYPE"):
                types = line.split()[1:]

        if num_points == 0:
            return np.zeros((0, 18), dtype=np.float32)

        # Determine if binary or ascii
        data_format = header_lines[-1].split()[-1].lower()

        if data_format == "binary":
            # Calculate bytes per point from sizes
            bytes_per_point = sum(sizes)
            raw_data = f.read(num_points * bytes_per_point)

            # Build struct format string
            fmt = ""
            for size, type_char in zip(sizes, types):
                if type_char == "F":
                    if size == 4:
                        fmt += "f"
                    elif size == 8:
                        fmt += "d"
                elif type_char == "U":
                    if size == 1:
                        fmt += "B"
                    elif size == 2:
                        fmt += "H"
                    elif size == 4:
                        fmt += "I"
                elif type_char == "I":
                    if size == 1:
                        fmt += "b"
                    elif size == 2:
                        fmt += "h"
                    elif size == 4:
                        fmt += "i"

            point_size = struct.calcsize(fmt)
            points = np.zeros((num_points, len(fields)), dtype=np.float32)

            for i in range(num_points):
                offset = i * point_size
                values = struct.unpack_from(fmt, raw_data, offset)
                points[i] = np.array(values, dtype=np.float32)

        elif data_format == "binary_compressed":
            # Read compressed size and uncompressed size
            compressed_size = struct.unpack("<I", f.read(4))[0]
            uncompressed_size = struct.unpack("<I", f.read(4))[0]
            compressed_data = f.read(compressed_size)

            # LZF decompression
            try:
                import lzf

                raw_data = lzf.decompress(compressed_data, uncompressed_size)
            except ImportError:
                # Fallback: try manual LZF decompression
                raw_data = _lzf_decompress(compressed_data, uncompressed_size)

            # Data is stored column-major in binary_compressed format
            points = np.zeros((num_points, len(fields)), dtype=np.float32)
            offset = 0
            for col_idx, (size, type_char) in enumerate(zip(sizes, types)):
                col_bytes = raw_data[offset : offset + num_points * size]
                offset += num_points * size

                if type_char == "F" and size == 4:
                    col_data = np.frombuffer(col_bytes, dtype=np.float32)
                elif type_char == "F" and size == 8:
                    col_data = np.frombuffer(col_bytes, dtype=np.float64).astype(
                        np.float32
                    )
                elif type_char == "U" and size == 4:
                    col_data = np.frombuffer(col_bytes, dtype=np.uint32).astype(
                        np.float32
                    )
                elif type_char == "U" and size == 2:
                    col_data = np.frombuffer(col_bytes, dtype=np.uint16).astype(
                        np.float32
                    )
                elif type_char == "U" and size == 1:
                    col_data = np.frombuffer(col_bytes, dtype=np.uint8).astype(
                        np.float32
                    )
                elif type_char == "I" and size == 4:
                    col_data = np.frombuffer(col_bytes, dtype=np.int32).astype(
                        np.float32
                    )
                elif type_char == "I" and size == 2:
                    col_data = np.frombuffer(col_bytes, dtype=np.int16).astype(
                        np.float32
                    )
                elif type_char == "I" and size == 1:
                    col_data = np.frombuffer(col_bytes, dtype=np.int8).astype(
                        np.float32
                    )
                else:
                    col_data = np.zeros(num_points, dtype=np.float32)

                points[:, col_idx] = col_data[:num_points]

        else:
            # ASCII format
            points = np.zeros((num_points, len(fields)), dtype=np.float32)
            for i in range(num_points):
                line = f.readline().decode("ascii", errors="ignore").strip()
                values = line.split()
                points[i] = np.array([float(v) for v in values], dtype=np.float32)

    # Ensure we have exactly 18 columns (pad or truncate)
    num_cols = points.shape[1] if len(points.shape) > 1 else 0
    if num_cols < 18:
        padded = np.zeros((points.shape[0], 18), dtype=np.float32)
        padded[:, :num_cols] = points
        points = padded
    elif num_cols > 18:
        points = points[:, :18]

    return points


def _lzf_decompress(compressed, uncompressed_size):
    """
    Simple LZF decompression implementation.

    Args:
        compressed: Compressed bytes
        uncompressed_size: Expected size of decompressed data

    Returns:
        Decompressed bytes
    """
    output = bytearray(uncompressed_size)
    i = 0  # input index
    o = 0  # output index

    while i < len(compressed):
        ctrl = compressed[i]
        i += 1

        if ctrl < 32:
            # Literal run: copy ctrl+1 bytes
            length = ctrl + 1
            output[o : o + length] = compressed[i : i + length]
            i += length
            o += length
        else:
            # Back-reference
            length = (ctrl >> 5) + 2
            if length == 9:  # length was 7 (max in 3 bits) + 2
                length += compressed[i]
                i += 1
            offset = ((ctrl & 0x1F) << 8) + compressed[i] + 1
            i += 1

            # Copy from back-reference (may overlap)
            ref = o - offset
            for _ in range(length):
                output[o] = output[ref]
                o += 1
                ref += 1

    return bytes(output)


def accumulate_radar_sweeps(nusc, sample_data, num_sweeps, ref_from_car, car_from_global):
    """
    Accumulate multi-sweep radar data with ego-motion compensation.

    Transforms points from each sweep's coordinate frame to the reference frame
    using calibration extrinsics and ego poses.

    Transform chain for each point:
        point_ref = ref_from_car @ car_from_global @ global_from_car_sweep @ car_sweep_from_sensor @ point_sensor

    Args:
        nusc: NuScenes instance
        sample_data: sample_data record dict for the current radar sample
        num_sweeps: Number of sweeps to accumulate (including current)
        ref_from_car: 4x4 transform from ego vehicle to reference sensor frame
        car_from_global: 4x4 transform from global to ego vehicle frame (at ref time)

    Returns:
        numpy array of shape (N, 21) - 18 radar features + 3 (time_lag, sensor_x_offset, sensor_y_offset)
    """
    all_points = []
    current_sd = sample_data
    ref_time = sample_data["timestamp"]

    for sweep_idx in range(num_sweeps):
        if current_sd is None:
            break

        # Get calibration: sensor -> ego vehicle
        cs_record = nusc.get("calibrated_sensor", current_sd["calibrated_sensor_token"])
        sensor_from_car = np.eye(4)
        sensor_from_car[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
        sensor_from_car[:3, 3] = np.array(cs_record["translation"])
        # car_from_sensor is the transform from sensor frame to ego frame
        car_from_sensor = sensor_from_car  # This IS car_from_sensor (sensor extrinsic)

        # Get ego pose at sweep time: ego -> global
        ego_pose = nusc.get("ego_pose", current_sd["ego_pose_token"])
        global_from_car_sweep = np.eye(4)
        global_from_car_sweep[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
        global_from_car_sweep[:3, 3] = np.array(ego_pose["translation"])

        # Full transform: sensor_sweep -> ref_sensor
        # point_ref = ref_from_car @ car_from_global @ global_from_car_sweep @ car_from_sensor @ point_sensor
        sweep_to_ref = ref_from_car @ car_from_global @ global_from_car_sweep @ car_from_sensor

        # Load points for this sweep
        points = get_radar_point_cloud(nusc, current_sd["token"])

        if points.shape[0] > 0:
            # Transform xyz coordinates
            num_pts = points.shape[0]
            xyz = points[:, :3]

            # Convert to homogeneous coordinates
            ones = np.ones((num_pts, 1), dtype=np.float32)
            xyz_hom = np.concatenate([xyz, ones], axis=1)  # (N, 4)

            # Apply transform
            xyz_transformed = (sweep_to_ref @ xyz_hom.T).T[:, :3]  # (N, 3)

            # Transform velocity components (vx, vy at indices 6,7 and vx_comp, vy_comp at 8,9)
            # Velocity is a vector, so we only apply rotation (no translation)
            rot_matrix = sweep_to_ref[:3, :3]

            # Compensated velocities (indices 8, 9) - transform to reference frame
            vx_comp = points[:, 8]
            vy_comp = points[:, 9]
            vz_comp = np.zeros_like(vx_comp)
            vel_sensor = np.stack([vx_comp, vy_comp, vz_comp], axis=1)  # (N, 3)
            vel_ref = (rot_matrix @ vel_sensor.T).T  # (N, 3)

            # Build output: original 18 features with transformed xyz and velocity
            sweep_points = points.copy()
            sweep_points[:, 0:3] = xyz_transformed
            sweep_points[:, 8] = vel_ref[:, 0]  # vx_comp in ref frame
            sweep_points[:, 9] = vel_ref[:, 1]  # vy_comp in ref frame

            # Compute time lag in seconds
            time_lag = (ref_time - current_sd["timestamp"]) * 1e-6  # microseconds to seconds
            time_lag_col = np.full((num_pts, 1), time_lag, dtype=np.float32)

            # Sensor position offset in reference frame (for multi-sensor fusion context)
            sensor_offset = sweep_to_ref[:2, 3]  # x, y offset of sensor origin in ref frame
            sensor_offset_col = np.tile(sensor_offset, (num_pts, 1)).astype(np.float32)

            # Concatenate: 18 features + time_lag + sensor_x_offset + sensor_y_offset = 21
            sweep_points_aug = np.concatenate(
                [sweep_points, time_lag_col, sensor_offset_col], axis=1
            )
            all_points.append(sweep_points_aug)

        # Move to previous sweep
        if current_sd["prev"] == "":
            # No more previous sweeps; repeat current
            break
        else:
            current_sd = nusc.get("sample_data", current_sd["prev"])

    if len(all_points) == 0:
        return np.zeros((0, 21), dtype=np.float32)

    return np.concatenate(all_points, axis=0).astype(np.float32)


def get_sample_info(nusc, sample):
    """
    Extract complete info dict for one sample.

    Args:
        nusc: NuScenes instance
        sample: sample record dict

    Returns:
        Dictionary with sample metadata, calibration, GT boxes, and radar paths
    """
    info = {
        "token": sample["token"],
        "timestamp": sample["timestamp"],
        "scene_token": sample["scene_token"],
    }

    # Reference sensor: use RADAR_FRONT as the reference frame
    ref_sensor = "RADAR_FRONT"
    ref_sd_token = sample["data"][ref_sensor]
    ref_sd = nusc.get("sample_data", ref_sd_token)

    # Reference ego pose (at the time of the reference sensor measurement)
    ref_ego_pose = nusc.get("ego_pose", ref_sd["ego_pose_token"])
    info["ego_pose"] = {
        "translation": ref_ego_pose["translation"],
        "rotation": ref_ego_pose["rotation"],
    }

    # Reference calibration (sensor -> ego)
    ref_cs = nusc.get("calibrated_sensor", ref_sd["calibrated_sensor_token"])

    # Compute reference transforms
    # ref_from_car: transform from ego vehicle frame to reference sensor frame
    ref_sensor_from_car = np.eye(4)
    ref_sensor_from_car[:3, :3] = Quaternion(ref_cs["rotation"]).rotation_matrix
    ref_sensor_from_car[:3, 3] = np.array(ref_cs["translation"])
    # ref_from_car is the inverse: car -> sensor (i.e., sensor_from_car inverted)
    ref_from_car = np.linalg.inv(ref_sensor_from_car)

    # car_from_global: transform from global frame to ego vehicle frame at ref time
    global_from_car_ref = np.eye(4)
    global_from_car_ref[:3, :3] = Quaternion(ref_ego_pose["rotation"]).rotation_matrix
    global_from_car_ref[:3, 3] = np.array(ref_ego_pose["translation"])
    car_from_global = np.linalg.inv(global_from_car_ref)

    info["ref_from_car"] = ref_from_car.tolist()
    info["car_from_global"] = car_from_global.tolist()

    # Calibration for each radar sensor
    calibrations = {}
    for sensor_name in RADAR_SENSORS:
        sd_token = sample["data"][sensor_name]
        sd = nusc.get("sample_data", sd_token)
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        calibrations[sensor_name] = {
            "translation": cs["translation"],
            "rotation": cs["rotation"],
            "sensor_token": cs["token"],
            "sample_data_token": sd_token,
        }
    info["calibrations"] = calibrations

    # Radar file paths for all 5 sensors
    radar_paths = {}
    for sensor_name in RADAR_SENSORS:
        sd_token = sample["data"][sensor_name]
        sd = nusc.get("sample_data", sd_token)
        radar_paths[sensor_name] = sd["filename"]
    info["radar_paths"] = radar_paths

    # Ground truth boxes (only for non-test sets)
    if sample["anns"]:
        gt_boxes = []
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)

            # Check if this category is one we care about
            category = ann["category_name"]
            detection_name = None
            for cat_prefix, det_name in CATEGORY_TO_DETECTION.items():
                if category.startswith(cat_prefix):
                    detection_name = det_name
                    break

            if detection_name is None:
                continue

            # Transform box from global to reference sensor frame
            # Box center in global frame
            center_global = np.array(ann["translation"])
            rotation_global = Quaternion(ann["rotation"])

            # Global -> ego vehicle -> reference sensor
            center_hom = np.array([*center_global, 1.0])
            center_ref = (ref_from_car @ car_from_global @ center_hom)[:3]

            # Rotation: compose the transforms
            # rotation_ref = ref_from_car_rot * car_from_global_rot * rotation_global
            ref_from_car_rot = Quaternion(matrix=ref_from_car[:3, :3])
            car_from_global_rot = Quaternion(matrix=car_from_global[:3, :3])
            rotation_ref = ref_from_car_rot * car_from_global_rot * rotation_global

            # Velocity in global frame
            velocity_global = nusc.box_velocity(ann_token)  # returns (3,) or nan
            if np.any(np.isnan(velocity_global)):
                velocity_ref = np.array([np.nan, np.nan, np.nan])
            else:
                # Transform velocity (vector, rotation only)
                rot_global_to_ref = (ref_from_car @ car_from_global)[:3, :3]
                velocity_ref = rot_global_to_ref @ velocity_global

            gt_box = {
                "center": center_ref.tolist(),
                "size": ann["size"],  # [width, length, height] in nuScenes
                "rotation": rotation_ref.elements.tolist(),  # [w, x, y, z]
                "velocity": velocity_ref.tolist(),
                "category_name": category,
                "detection_name": detection_name,
                "num_lidar_pts": ann["num_lidar_pts"],
                "num_radar_pts": ann["num_radar_pts"],
                "token": ann_token,
            }
            gt_boxes.append(gt_box)

        info["gt_boxes"] = gt_boxes
    else:
        info["gt_boxes"] = []

    return info


def create_info_files(nusc, output_dir, num_sweeps):
    """
    Create train_infos.pkl and val_infos.pkl files.

    Uses nuScenes official train/val splits.

    Args:
        nusc: NuScenes instance
        output_dir: Directory to write output pkl files
        num_sweeps: Number of radar sweeps to record in metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get official splits
    splits = create_splits_scenes()

    # Determine which splits to process based on version
    version = nusc.version
    if "mini" in version:
        train_scenes = splits["mini_train"]
        val_scenes = splits["mini_val"]
    elif "test" in version:
        train_scenes = []
        val_scenes = []
        test_scenes = splits["test"]
    else:
        train_scenes = splits["train"]
        val_scenes = splits["val"]

    # Map scene names to tokens
    scene_name_to_token = {scene["name"]: scene["token"] for scene in nusc.scene}

    def process_split(scene_names, split_name):
        """Process all samples in the given scenes."""
        infos = []
        scene_tokens = set()
        for name in scene_names:
            if name in scene_name_to_token:
                scene_tokens.add(scene_name_to_token[name])

        for sample in tqdm(nusc.sample, desc=f"Processing {split_name}"):
            if sample["scene_token"] not in scene_tokens:
                continue
            info = get_sample_info(nusc, sample)
            info["num_sweeps"] = num_sweeps
            infos.append(info)

        return infos

    if "test" in version:
        test_infos = process_split(test_scenes, "test")
        output_path = output_dir / "test_infos.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(test_infos, f)
        print(f"Saved {len(test_infos)} test infos to {output_path}")
    else:
        train_infos = process_split(train_scenes, "train")
        val_infos = process_split(val_scenes, "val")

        train_path = output_dir / "train_infos.pkl"
        val_path = output_dir / "val_infos.pkl"

        with open(train_path, "wb") as f:
            pickle.dump(train_infos, f)
        print(f"Saved {len(train_infos)} train infos to {train_path}")

        with open(val_path, "wb") as f:
            pickle.dump(val_infos, f)
        print(f"Saved {len(val_infos)} val infos to {val_path}")

    return train_infos if "test" not in version else test_infos


def create_gt_database(nusc, infos, output_dir, num_sweeps):
    """
    Generate GT database for GT-sampling augmentation.

    For each GT box in the training set, crops the radar points inside the box
    and saves them as individual .bin files. Also produces a dbinfos dict with
    metadata for each GT instance.

    Args:
        nusc: NuScenes instance
        infos: List of sample info dicts (from create_info_files)
        output_dir: Root output directory
        num_sweeps: Number of sweeps to accumulate for each sample
    """
    output_dir = Path(output_dir)
    db_dir = output_dir / "gt_database"
    db_dir.mkdir(parents=True, exist_ok=True)

    dbinfos = {}  # detection_name -> list of dbinfo dicts

    for idx, info in enumerate(tqdm(infos, desc="Creating GT database")):
        sample_token = info["token"]
        sample = nusc.get("sample", sample_token)

        # Reconstruct transforms
        ref_from_car = np.array(info["ref_from_car"])
        car_from_global = np.array(info["car_from_global"])

        # Accumulate radar points from all 5 sensors
        all_radar_points = []
        for sensor_name in RADAR_SENSORS:
            sd_token = info["calibrations"][sensor_name]["sample_data_token"]
            sd = nusc.get("sample_data", sd_token)
            points = accumulate_radar_sweeps(
                nusc, sd, num_sweeps, ref_from_car, car_from_global
            )
            if points.shape[0] > 0:
                all_radar_points.append(points)

        if len(all_radar_points) == 0:
            continue

        radar_points = np.concatenate(all_radar_points, axis=0)  # (N, 21)

        # For each GT box, crop points inside
        for box_idx, gt_box in enumerate(info["gt_boxes"]):
            detection_name = gt_box["detection_name"]
            center = np.array(gt_box["center"])
            size = np.array(gt_box["size"])  # [width, length, height]
            rotation = Quaternion(gt_box["rotation"])

            # Get points in box frame
            # Translate points so box center is at origin
            points_xyz = radar_points[:, :3] - center[np.newaxis, :]

            # Rotate points into box-aligned frame (inverse of box rotation)
            rot_inv = rotation.inverse.rotation_matrix
            points_box_frame = (rot_inv @ points_xyz.T).T  # (N, 3)

            # Check which points are inside the box (half-extents)
            # nuScenes size convention: [width, length, height]
            half_w = size[0] / 2.0
            half_l = size[1] / 2.0
            half_h = size[2] / 2.0

            mask = (
                (np.abs(points_box_frame[:, 0]) <= half_w)
                & (np.abs(points_box_frame[:, 1]) <= half_l)
                & (np.abs(points_box_frame[:, 2]) <= half_h)
            )

            points_in_box = radar_points[mask]

            if points_in_box.shape[0] == 0:
                continue

            # Save cropped points as .bin file
            # Filename: {sample_idx}_{box_idx}_{detection_name}.bin
            filename = f"{idx}_{box_idx}_{detection_name}.bin"
            filepath = db_dir / filename
            points_in_box.astype(np.float32).tofile(str(filepath))

            # Store metadata
            dbinfo = {
                "name": detection_name,
                "path": str(Path("gt_database") / filename),
                "gt_idx": box_idx,
                "box3d_lidar": gt_box["center"] + gt_box["size"] + gt_box["rotation"],
                "num_points_in_gt": int(points_in_box.shape[0]),
                "difficulty": _get_difficulty(gt_box),
                "category_name": gt_box["category_name"],
                "sample_token": sample_token,
            }

            if detection_name not in dbinfos:
                dbinfos[detection_name] = []
            dbinfos[detection_name].append(dbinfo)

    # Save dbinfos
    dbinfos_path = output_dir / "dbinfos_train.pkl"
    with open(dbinfos_path, "wb") as f:
        pickle.dump(dbinfos, f)

    # Print statistics
    print(f"\nGT Database Statistics:")
    print(f"{'Category':<25} {'Count':<10}")
    print("-" * 35)
    total = 0
    for name in DETECTION_NAMES:
        count = len(dbinfos.get(name, []))
        if count > 0:
            print(f"{name:<25} {count:<10}")
            total += count
    print("-" * 35)
    print(f"{'Total':<25} {total:<10}")
    print(f"\nSaved GT database to {db_dir}")
    print(f"Saved dbinfos to {dbinfos_path}")


def _get_difficulty(gt_box):
    """
    Assign difficulty level based on number of radar points.

    Args:
        gt_box: GT box info dict

    Returns:
        Integer difficulty level (0=easy, 1=moderate, 2=hard)
    """
    num_radar_pts = gt_box.get("num_radar_pts", 0)
    if num_radar_pts >= 5:
        return 0  # easy
    elif num_radar_pts >= 2:
        return 1  # moderate
    else:
        return 2  # hard


def main():
    """Main entry point with argparse CLI."""
    parser = argparse.ArgumentParser(
        description="Prepare nuScenes radar data for PillarNet training."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to the nuScenes dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory for processed data.",
    )
    parser.add_argument(
        "--num-sweeps",
        type=int,
        default=6,
        help="Number of radar sweeps to accumulate (default: 6).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for processing (default: 4).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        choices=["v1.0-trainval", "v1.0-mini", "v1.0-test"],
        help="nuScenes dataset version (default: v1.0-trainval).",
    )
    parser.add_argument(
        "--create-gt-db",
        action="store_true",
        help="Whether to create the GT database for GT-sampling augmentation.",
    )

    args = parser.parse_args()

    # Validate paths
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"nuScenes data root: {data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Version: {args.version}")
    print(f"Number of sweeps: {args.num_sweeps}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Create GT database: {args.create_gt_db}")
    print()

    # Initialize nuScenes
    print("Loading nuScenes database...")
    nusc = NuScenes(version=args.version, dataroot=str(data_root), verbose=True)
    print()

    # Create info files
    print("Creating info files...")
    infos = create_info_files(nusc, str(output_dir), args.num_sweeps)
    print()

    # Create GT database (only for training data)
    if args.create_gt_db and "test" not in args.version:
        print("Creating GT database...")
        # Load train infos for GT database creation
        train_infos_path = output_dir / "train_infos.pkl"
        if train_infos_path.exists():
            with open(train_infos_path, "rb") as f:
                train_infos = pickle.load(f)
            create_gt_database(nusc, train_infos, str(output_dir), args.num_sweeps)
        else:
            print("Warning: train_infos.pkl not found, skipping GT database creation.")

    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
