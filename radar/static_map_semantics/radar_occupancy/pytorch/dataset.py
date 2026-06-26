"""
Radar Occupancy Grid Dataset — nuScenes radar data loading with multi-sweep
accumulation and ground truth occupancy generation from LiDAR.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion


class RadarOccupancyDataset(Dataset):
    """nuScenes radar dataset for occupancy grid prediction.

    Loads multi-sweep radar points, applies ego-motion compensation,
    creates pillar features, and generates ground truth occupancy from LiDAR.
    """

    def __init__(self, config, split="train", nusc=None):
        """
        Args:
            config: Full configuration dict
            split: 'train' or 'val'
            nusc: Optional pre-loaded NuScenes instance
        """
        self.config = config
        self.split = split
        self.dataset_cfg = config["dataset"]
        self.grid_cfg = config["grid"]
        self.pillar_cfg = config["model"]["pillar"]

        self.root = self.dataset_cfg["root"]
        self.num_sweeps = self.dataset_cfg["num_sweeps"]
        self.min_rcs = self.dataset_cfg["min_rcs"]
        self.radar_sensors = self.dataset_cfg["radar_sensors"]

        self.grid_size = self.grid_cfg["grid_size"]
        self.cell_size = self.grid_cfg["cell_size"]
        self.x_range = self.grid_cfg["x_range"]
        self.y_range = self.grid_cfg["y_range"]
        self.z_range = self.grid_cfg["z_range"]

        self.max_pillars = self.pillar_cfg["max_pillars"]
        self.max_points_per_pillar = self.pillar_cfg["max_points_per_pillar"]

        self.augment = split == "train" and "augmentation" in config.get("training", {})
        if self.augment:
            self.aug_cfg = config["training"]["augmentation"]

        if nusc is None:
            from nuscenes.nuscenes import NuScenes
            self.nusc = NuScenes(
                version=self.dataset_cfg["version"],
                dataroot=self.root,
                verbose=False
            )
        else:
            self.nusc = nusc

        self.samples = self._get_samples(split)

    def _get_samples(self, split):
        """Get sample tokens for the split."""
        scenes = self.nusc.scene
        n_train = self.dataset_cfg.get("train_scenes", 700)

        if split == "train":
            split_scenes = scenes[:n_train]
        else:
            split_scenes = scenes[n_train:]

        samples = []
        for scene in split_scenes:
            sample_token = scene["first_sample_token"]
            while sample_token:
                samples.append(sample_token)
                sample = self.nusc.get("sample", sample_token)
                sample_token = sample["next"]

        return samples

    def __len__(self):
        return len(self.samples)

    def _get_radar_points(self, sample_token):
        """Get multi-sweep radar points from all radar sensors in ego frame.

        Returns:
            points: (N, 6) [x, y, z, rcs, vr_comp, dt]
        """
        sample = self.nusc.get("sample", sample_token)
        all_points = []

        ego_pose_token = self.nusc.get(
            "sample_data",
            sample["data"][self.radar_sensors[0]]
        )["ego_pose_token"]
        ego_pose = self.nusc.get("ego_pose", ego_pose_token)
        ego_translation = np.array(ego_pose["translation"])
        ego_rotation = Quaternion(ego_pose["rotation"])

        for sensor in self.radar_sensors:
            sd_token = sample["data"][sensor]

            for sweep_idx in range(self.num_sweeps):
                sd = self.nusc.get("sample_data", sd_token)

                cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                sensor_translation = np.array(cs["translation"])
                sensor_rotation = Quaternion(cs["rotation"])

                sweep_ego = self.nusc.get("ego_pose", sd["ego_pose_token"])
                sweep_ego_translation = np.array(sweep_ego["translation"])
                sweep_ego_rotation = Quaternion(sweep_ego["rotation"])

                pc_path = os.path.join(self.root, sd["filename"])
                if not os.path.exists(pc_path):
                    break

                points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 18)

                valid = points[:, 14] == 0  # invalid_state == 0
                if self.min_rcs is not None:
                    valid &= points[:, 5] >= self.min_rcs
                points = points[valid]

                if len(points) == 0:
                    if sd["prev"]:
                        sd_token = sd["prev"]
                    continue

                xyz = points[:, :3]
                rcs = points[:, 5:6]
                vr_comp = np.sqrt(points[:, 8:9]**2 + points[:, 9:10]**2)

                xyz_hom = np.hstack([xyz, np.ones((len(xyz), 1))])

                rot_mat = sensor_rotation.rotation_matrix
                trans = sensor_translation
                xyz_ego_sweep = (rot_mat @ xyz[:, :3].T).T + trans

                rot_mat_ego = sweep_ego_rotation.rotation_matrix
                xyz_global = (rot_mat_ego @ xyz_ego_sweep.T).T + sweep_ego_translation

                rot_mat_cur = ego_rotation.rotation_matrix
                xyz_cur_ego = (rot_mat_cur.T @ (xyz_global - ego_translation).T).T

                dt = np.full((len(xyz_cur_ego), 1),
                           sweep_idx * (1.0 / 13.0))  # ~77ms per sweep

                point_features = np.hstack([
                    xyz_cur_ego, rcs, vr_comp, dt
                ])
                all_points.append(point_features)

                if sd["prev"]:
                    sd_token = sd["prev"]
                else:
                    break

        if all_points:
            return np.concatenate(all_points, axis=0).astype(np.float32)
        else:
            return np.zeros((0, 6), dtype=np.float32)

    def _get_lidar_occupancy_gt(self, sample_token):
        """Generate ground truth occupancy grid from LiDAR point cloud.

        Returns:
            occ_grid: (H, W) with 0=free, 1=occupied, 2=unknown
        """
        sample = self.nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_sd = self.nusc.get("sample_data", lidar_token)

        cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_rotation = Quaternion(cs["rotation"]).rotation_matrix
        lidar_translation = np.array(cs["translation"])

        pc_path = os.path.join(self.root, lidar_sd["filename"])
        points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 5)
        xyz = points[:, :3]

        xyz_ego = (lidar_rotation @ xyz.T).T + lidar_translation

        H, W = self.grid_size
        occ_grid = np.full((H, W), 2, dtype=np.int64)  # Unknown

        in_range = (
            (xyz_ego[:, 0] >= self.x_range[0]) & (xyz_ego[:, 0] < self.x_range[1]) &
            (xyz_ego[:, 1] >= self.y_range[0]) & (xyz_ego[:, 1] < self.y_range[1]) &
            (xyz_ego[:, 2] >= self.z_range[0]) & (xyz_ego[:, 2] < self.z_range[1])
        )
        xyz_valid = xyz_ego[in_range]

        gx = ((xyz_valid[:, 0] - self.x_range[0]) / self.cell_size).astype(int)
        gy = ((xyz_valid[:, 1] - self.y_range[0]) / self.cell_size).astype(int)
        gx = np.clip(gx, 0, H - 1)
        gy = np.clip(gy, 0, W - 1)
        occ_grid[gx, gy] = 1  # Occupied

        lidar_origin = lidar_translation[:2]
        origin_gx = int((lidar_origin[0] - self.x_range[0]) / self.cell_size)
        origin_gy = int((lidar_origin[1] - self.y_range[0]) / self.cell_size)

        max_range = min(
            abs(self.x_range[1] - self.x_range[0]),
            abs(self.y_range[1] - self.y_range[0])
        ) / 2

        for gxi, gyi in zip(gx[::10], gy[::10]):
            self._mark_free_ray(occ_grid, origin_gx, origin_gy, gxi, gyi)

        return occ_grid

    def _mark_free_ray(self, grid, x0, y0, x1, y1):
        """Bresenham's line to mark free cells along ray (excluding endpoint)."""
        H, W = grid.shape
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while (x, y) != (x1, y1):
            if 0 <= x < H and 0 <= y < W and grid[x, y] == 2:
                grid[x, y] = 0  # Free
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _create_pillars(self, points):
        """Convert points to pillar representation.

        Args:
            points: (N, 6) [x, y, z, rcs, vr, dt]

        Returns:
            pillar_features: (max_pillars, max_points, 9)
            pillar_indices: (max_pillars, 2) [grid_x, grid_y]
            num_pillars: int
        """
        H, W = self.grid_size

        if len(points) == 0:
            return (
                np.zeros((self.max_pillars, self.max_points_per_pillar, 9), dtype=np.float32),
                np.zeros((self.max_pillars, 2), dtype=np.int64),
                0
            )

        gx = ((points[:, 0] - self.x_range[0]) / self.cell_size).astype(int)
        gy = ((points[:, 1] - self.y_range[0]) / self.cell_size).astype(int)

        valid = (gx >= 0) & (gx < H) & (gy >= 0) & (gy < W)
        points = points[valid]
        gx = gx[valid]
        gy = gy[valid]

        pillar_ids = gx * W + gy
        unique_pillars, inverse = np.unique(pillar_ids, return_inverse=True)

        num_pillars = min(len(unique_pillars), self.max_pillars)
        pillar_features = np.zeros(
            (self.max_pillars, self.max_points_per_pillar, 9), dtype=np.float32
        )
        pillar_indices = np.zeros((self.max_pillars, 2), dtype=np.int64)

        for p_idx in range(num_pillars):
            mask = inverse == p_idx
            p_points = points[mask]

            if len(p_points) > self.max_points_per_pillar:
                choice = np.random.choice(len(p_points), self.max_points_per_pillar, replace=False)
                p_points = p_points[choice]

            n_pts = len(p_points)
            center_x = p_points[:, 0].mean()
            center_y = p_points[:, 1].mean()
            center_z = p_points[:, 2].mean()

            augmented = np.zeros((n_pts, 9), dtype=np.float32)
            augmented[:, :6] = p_points
            augmented[:, 6] = p_points[:, 0] - center_x
            augmented[:, 7] = p_points[:, 1] - center_y
            augmented[:, 8] = p_points[:, 2] - center_z

            pillar_features[p_idx, :n_pts] = augmented

            p_gx = gx[mask][0]
            p_gy = gy[mask][0]
            pillar_indices[p_idx] = [p_gx, p_gy]

        return pillar_features, pillar_indices, num_pillars

    def _augment(self, points, occ_grid):
        """Apply data augmentation."""
        if not self.augment:
            return points, occ_grid

        if self.aug_cfg.get("random_flip", False) and np.random.rand() > 0.5:
            points[:, 1] = -points[:, 1]
            occ_grid = np.flip(occ_grid, axis=1).copy()

        if "random_rotate" in self.aug_cfg:
            angle_range = self.aug_cfg["random_rotate"]
            angle = np.random.uniform(angle_range[0], angle_range[1])
            angle_rad = np.radians(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

            x_new = points[:, 0] * cos_a - points[:, 1] * sin_a
            y_new = points[:, 0] * sin_a + points[:, 1] * cos_a
            points[:, 0] = x_new
            points[:, 1] = y_new

        if "point_dropout" in self.aug_cfg:
            keep = np.random.rand(len(points)) > self.aug_cfg["point_dropout"]
            points = points[keep]

        return points, occ_grid

    def __getitem__(self, idx):
        sample_token = self.samples[idx]

        radar_points = self._get_radar_points(sample_token)
        occ_grid = self._get_lidar_occupancy_gt(sample_token)

        radar_points, occ_grid = self._augment(radar_points, occ_grid)

        pillar_features, pillar_indices, num_pillars = self._create_pillars(radar_points)

        return {
            "pillar_features": torch.from_numpy(pillar_features),
            "pillar_indices": torch.from_numpy(pillar_indices),
            "num_pillars": torch.tensor(num_pillars, dtype=torch.long),
            "occupancy_gt": torch.from_numpy(occ_grid),
            "sample_token": sample_token,
        }


class TemporalRadarOccupancyDataset(RadarOccupancyDataset):
    """Extended dataset that loads temporal sequences for TemporalPillarOccNet."""

    def __init__(self, config, split="train", nusc=None):
        super().__init__(config, split, nusc)
        self.num_frames = config["model"]["temporal"]["num_frames"]

    def __getitem__(self, idx):
        sample_token = self.samples[idx]
        sample = self.nusc.get("sample", sample_token)

        tokens = [sample_token]
        current = sample
        for _ in range(self.num_frames - 1):
            if current["prev"]:
                tokens.insert(0, current["prev"])
                current = self.nusc.get("sample", current["prev"])
            else:
                tokens.insert(0, tokens[0])

        pillar_features_seq = []
        pillar_indices_seq = []
        num_pillars_seq = []
        ego_transforms = []

        current_ego = self._get_ego_pose(tokens[-1])

        for t, token in enumerate(tokens):
            points = self._get_radar_points(token)
            pf, pi, np_ = self._create_pillars(points)
            pillar_features_seq.append(torch.from_numpy(pf))
            pillar_indices_seq.append(torch.from_numpy(pi))
            num_pillars_seq.append(torch.tensor(np_, dtype=torch.long))

            if t < len(tokens) - 1:
                past_ego = self._get_ego_pose(token)
                T = self._compute_relative_transform(past_ego, current_ego)
                ego_transforms.append(torch.from_numpy(T))

        occ_grid = self._get_lidar_occupancy_gt(sample_token)

        if not ego_transforms:
            ego_transforms = [torch.eye(4, dtype=torch.float32)]

        return {
            "pillar_features_seq": pillar_features_seq,
            "pillar_indices_seq": pillar_indices_seq,
            "num_pillars_seq": num_pillars_seq,
            "ego_transforms": torch.stack(ego_transforms),
            "occupancy_gt": torch.from_numpy(occ_grid),
            "sample_token": sample_token,
        }

    def _get_ego_pose(self, sample_token):
        """Get ego pose for a sample."""
        sample = self.nusc.get("sample", sample_token)
        sd = self.nusc.get("sample_data", sample["data"][self.radar_sensors[0]])
        ego = self.nusc.get("ego_pose", sd["ego_pose_token"])
        return {
            "translation": np.array(ego["translation"]),
            "rotation": Quaternion(ego["rotation"]),
        }

    def _compute_relative_transform(self, past_ego, current_ego):
        """Compute 4x4 transform from past ego frame to current ego frame."""
        T_past = np.eye(4, dtype=np.float32)
        T_past[:3, :3] = past_ego["rotation"].rotation_matrix
        T_past[:3, 3] = past_ego["translation"]

        T_current = np.eye(4, dtype=np.float32)
        T_current[:3, :3] = current_ego["rotation"].rotation_matrix
        T_current[:3, 3] = current_ego["translation"]

        T_relative = np.linalg.inv(T_current) @ T_past
        return T_relative.astype(np.float32)
