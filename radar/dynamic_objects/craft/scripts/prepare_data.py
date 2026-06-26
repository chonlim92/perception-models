#!/usr/bin/env python3
"""
prepare_data.py - Prepare nuScenes data for CRAFT model training/evaluation.

This script parses the nuScenes database JSON tables and creates preprocessed
info pickle files containing calibration matrices, file paths, radar point clouds
with ego-motion compensation, and 3D bounding box annotations.

Usage:
    python prepare_data.py --dataroot /data/nuscenes --version v1.0-trainval --out-dir ./data
    python prepare_data.py --dataroot /data/nuscenes --version v1.0-mini --out-dir ./data
    python prepare_data.py --dataroot /data/nuscenes --version v1.0-trainval \
        --out-dir ./data --num-sweeps 6 --workers 8

Output:
    - craft_infos_train.pkl: Training split info
    - craft_infos_val.pkl: Validation split info
    - (or craft_infos_mini_train.pkl / craft_infos_mini_val.pkl for mini version)
"""

import argparse
import json
import os
import pickle
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion

# =============================================================================
# nuScenes Official Train/Val Splits
# =============================================================================

# Official nuScenes train/val scene names
# Reference: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py

TRAIN_SCENES_V1_0 = [
    'scene-0001', 'scene-0002', 'scene-0004', 'scene-0005', 'scene-0006',
    'scene-0007', 'scene-0008', 'scene-0009', 'scene-0010', 'scene-0011',
    'scene-0019', 'scene-0020', 'scene-0021', 'scene-0022', 'scene-0023',
    'scene-0024', 'scene-0025', 'scene-0026', 'scene-0027', 'scene-0028',
    'scene-0029', 'scene-0030', 'scene-0031', 'scene-0032', 'scene-0033',
    'scene-0034', 'scene-0041', 'scene-0042', 'scene-0043', 'scene-0044',
    'scene-0045', 'scene-0046', 'scene-0047', 'scene-0048', 'scene-0049',
    'scene-0050', 'scene-0051', 'scene-0052', 'scene-0053', 'scene-0054',
    'scene-0055', 'scene-0056', 'scene-0057', 'scene-0058', 'scene-0059',
    'scene-0060', 'scene-0061', 'scene-0062', 'scene-0063', 'scene-0064',
    'scene-0065', 'scene-0066', 'scene-0067', 'scene-0068', 'scene-0069',
    'scene-0070', 'scene-0071', 'scene-0072', 'scene-0073', 'scene-0074',
    'scene-0075', 'scene-0076', 'scene-0077', 'scene-0078', 'scene-0079',
    'scene-0080', 'scene-0081', 'scene-0082', 'scene-0083', 'scene-0084',
    'scene-0085', 'scene-0086', 'scene-0087', 'scene-0088', 'scene-0089',
    'scene-0090', 'scene-0091', 'scene-0092', 'scene-0093', 'scene-0094',
    'scene-0095', 'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099',
    'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103', 'scene-0104',
    'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109',
    'scene-0110', 'scene-0111', 'scene-0112', 'scene-0113', 'scene-0114',
    'scene-0115', 'scene-0116', 'scene-0117', 'scene-0118', 'scene-0119',
    'scene-0120', 'scene-0121', 'scene-0122', 'scene-0123', 'scene-0124',
    'scene-0125', 'scene-0126', 'scene-0127', 'scene-0128', 'scene-0129',
    'scene-0130', 'scene-0131', 'scene-0132', 'scene-0133', 'scene-0134',
    'scene-0135', 'scene-0138', 'scene-0139', 'scene-0149', 'scene-0150',
    'scene-0151', 'scene-0152', 'scene-0154', 'scene-0155', 'scene-0157',
    'scene-0158', 'scene-0159', 'scene-0160', 'scene-0161', 'scene-0162',
    'scene-0163', 'scene-0164', 'scene-0165', 'scene-0166', 'scene-0167',
    'scene-0168', 'scene-0170', 'scene-0171', 'scene-0172', 'scene-0173',
    'scene-0174', 'scene-0175', 'scene-0176', 'scene-0177', 'scene-0178',
    'scene-0179', 'scene-0180', 'scene-0181', 'scene-0182', 'scene-0183',
    'scene-0184', 'scene-0185', 'scene-0187', 'scene-0188', 'scene-0190',
    'scene-0191', 'scene-0192', 'scene-0193', 'scene-0194', 'scene-0195',
    'scene-0196', 'scene-0199', 'scene-0200', 'scene-0202', 'scene-0203',
    'scene-0204', 'scene-0206', 'scene-0207', 'scene-0208', 'scene-0209',
    'scene-0210', 'scene-0211', 'scene-0212', 'scene-0213', 'scene-0214',
    'scene-0218', 'scene-0219', 'scene-0220', 'scene-0222', 'scene-0224',
    'scene-0225', 'scene-0226', 'scene-0227', 'scene-0228', 'scene-0229',
    'scene-0230', 'scene-0231', 'scene-0232', 'scene-0233', 'scene-0234',
    'scene-0235', 'scene-0236', 'scene-0237', 'scene-0238', 'scene-0239',
    'scene-0240', 'scene-0241', 'scene-0242', 'scene-0243', 'scene-0244',
    'scene-0245', 'scene-0246', 'scene-0247', 'scene-0248', 'scene-0249',
    'scene-0250', 'scene-0251', 'scene-0252', 'scene-0253', 'scene-0254',
    'scene-0255', 'scene-0256', 'scene-0257', 'scene-0258', 'scene-0259',
    'scene-0260', 'scene-0261', 'scene-0262', 'scene-0263', 'scene-0264',
    'scene-0283', 'scene-0284', 'scene-0285', 'scene-0286', 'scene-0287',
    'scene-0288', 'scene-0289', 'scene-0290', 'scene-0291', 'scene-0292',
    'scene-0293', 'scene-0294', 'scene-0295', 'scene-0296', 'scene-0297',
    'scene-0298', 'scene-0299', 'scene-0300', 'scene-0301', 'scene-0302',
    'scene-0303', 'scene-0304', 'scene-0305', 'scene-0306', 'scene-0315',
    'scene-0316', 'scene-0317', 'scene-0318', 'scene-0321', 'scene-0323',
    'scene-0324', 'scene-0328', 'scene-0347', 'scene-0348', 'scene-0349',
    'scene-0350', 'scene-0351', 'scene-0352', 'scene-0353', 'scene-0354',
    'scene-0355', 'scene-0356', 'scene-0357', 'scene-0358', 'scene-0359',
    'scene-0360', 'scene-0361', 'scene-0362', 'scene-0363', 'scene-0364',
    'scene-0365', 'scene-0366', 'scene-0367', 'scene-0368', 'scene-0369',
    'scene-0370', 'scene-0371', 'scene-0372', 'scene-0373', 'scene-0374',
    'scene-0375', 'scene-0376', 'scene-0377', 'scene-0378', 'scene-0379',
    'scene-0380', 'scene-0381', 'scene-0382', 'scene-0383', 'scene-0384',
    'scene-0385', 'scene-0386', 'scene-0388', 'scene-0389', 'scene-0390',
    'scene-0391', 'scene-0392', 'scene-0393', 'scene-0394', 'scene-0395',
    'scene-0396', 'scene-0397', 'scene-0398', 'scene-0399', 'scene-0400',
    'scene-0401', 'scene-0402', 'scene-0403', 'scene-0405', 'scene-0406',
    'scene-0407', 'scene-0408', 'scene-0410', 'scene-0411', 'scene-0412',
    'scene-0413', 'scene-0414', 'scene-0415', 'scene-0416', 'scene-0417',
    'scene-0418', 'scene-0419', 'scene-0420', 'scene-0421', 'scene-0422',
    'scene-0423', 'scene-0424', 'scene-0425', 'scene-0426', 'scene-0427',
    'scene-0428', 'scene-0429', 'scene-0430', 'scene-0431', 'scene-0432',
    'scene-0433', 'scene-0434', 'scene-0435', 'scene-0436', 'scene-0437',
    'scene-0438', 'scene-0439', 'scene-0440', 'scene-0441', 'scene-0442',
    'scene-0443', 'scene-0444', 'scene-0445', 'scene-0446', 'scene-0447',
    'scene-0448', 'scene-0449', 'scene-0450', 'scene-0451', 'scene-0452',
    'scene-0453', 'scene-0454', 'scene-0455', 'scene-0456', 'scene-0457',
    'scene-0458', 'scene-0459', 'scene-0461', 'scene-0462', 'scene-0463',
    'scene-0464', 'scene-0465', 'scene-0467', 'scene-0468', 'scene-0469',
    'scene-0471', 'scene-0472', 'scene-0474', 'scene-0475', 'scene-0476',
    'scene-0477', 'scene-0478', 'scene-0479', 'scene-0480', 'scene-0499',
    'scene-0500', 'scene-0501', 'scene-0502', 'scene-0504', 'scene-0505',
    'scene-0506', 'scene-0507', 'scene-0508', 'scene-0509', 'scene-0510',
    'scene-0511', 'scene-0512', 'scene-0513', 'scene-0514', 'scene-0515',
    'scene-0517', 'scene-0518', 'scene-0525', 'scene-0526', 'scene-0527',
    'scene-0528', 'scene-0529', 'scene-0530', 'scene-0531', 'scene-0532',
    'scene-0533', 'scene-0534', 'scene-0535', 'scene-0536', 'scene-0537',
    'scene-0538', 'scene-0539', 'scene-0541', 'scene-0542', 'scene-0543',
    'scene-0544', 'scene-0545', 'scene-0546', 'scene-0566', 'scene-0568',
    'scene-0570', 'scene-0571', 'scene-0572', 'scene-0573', 'scene-0574',
    'scene-0575', 'scene-0576', 'scene-0577', 'scene-0578', 'scene-0580',
    'scene-0582', 'scene-0583', 'scene-0584', 'scene-0585', 'scene-0586',
    'scene-0587', 'scene-0588', 'scene-0589', 'scene-0590', 'scene-0591',
    'scene-0592', 'scene-0593', 'scene-0594', 'scene-0595', 'scene-0596',
    'scene-0597', 'scene-0598', 'scene-0599', 'scene-0600', 'scene-0639',
    'scene-0640', 'scene-0641', 'scene-0642', 'scene-0643', 'scene-0644',
    'scene-0645', 'scene-0646', 'scene-0647', 'scene-0648', 'scene-0649',
    'scene-0650', 'scene-0651', 'scene-0652', 'scene-0653', 'scene-0654',
    'scene-0655', 'scene-0656', 'scene-0657', 'scene-0658', 'scene-0659',
    'scene-0660', 'scene-0661', 'scene-0662', 'scene-0663', 'scene-0664',
    'scene-0665', 'scene-0666', 'scene-0667', 'scene-0668', 'scene-0669',
    'scene-0670', 'scene-0671', 'scene-0672', 'scene-0673', 'scene-0674',
    'scene-0675', 'scene-0676', 'scene-0677', 'scene-0678', 'scene-0679',
    'scene-0681', 'scene-0683', 'scene-0684', 'scene-0685', 'scene-0686',
    'scene-0687', 'scene-0688', 'scene-0689', 'scene-0695', 'scene-0696',
    'scene-0697', 'scene-0698', 'scene-0700', 'scene-0701', 'scene-0703',
    'scene-0704', 'scene-0705', 'scene-0706', 'scene-0707', 'scene-0708',
    'scene-0709', 'scene-0710', 'scene-0711', 'scene-0712', 'scene-0713',
    'scene-0714', 'scene-0715', 'scene-0716', 'scene-0717', 'scene-0718',
    'scene-0719', 'scene-0726', 'scene-0727', 'scene-0728', 'scene-0730',
    'scene-0731', 'scene-0733', 'scene-0734', 'scene-0735', 'scene-0736',
    'scene-0737', 'scene-0738', 'scene-0786', 'scene-0787', 'scene-0789',
    'scene-0790', 'scene-0791', 'scene-0792', 'scene-0803', 'scene-0804',
    'scene-0805', 'scene-0806', 'scene-0808', 'scene-0809', 'scene-0810',
    'scene-0811', 'scene-0812', 'scene-0813', 'scene-0815', 'scene-0816',
    'scene-0817', 'scene-0819', 'scene-0820', 'scene-0821', 'scene-0822',
    'scene-0847', 'scene-0848', 'scene-0849', 'scene-0850', 'scene-0851',
    'scene-0852', 'scene-0853', 'scene-0854', 'scene-0855', 'scene-0856',
    'scene-0858', 'scene-0860', 'scene-0861', 'scene-0862', 'scene-0863',
    'scene-0864', 'scene-0865', 'scene-0866', 'scene-0868', 'scene-0869',
    'scene-0870', 'scene-0871', 'scene-0872', 'scene-0873', 'scene-0875',
    'scene-0876', 'scene-0877', 'scene-0878', 'scene-0880', 'scene-0882',
    'scene-0883', 'scene-0884', 'scene-0885', 'scene-0886', 'scene-0887',
    'scene-0888', 'scene-0889', 'scene-0890', 'scene-0891', 'scene-0892',
    'scene-0893', 'scene-0894', 'scene-0895', 'scene-0896', 'scene-0897',
    'scene-0898', 'scene-0899', 'scene-0900', 'scene-0901', 'scene-0902',
    'scene-0903', 'scene-0945', 'scene-0947', 'scene-0949', 'scene-0952',
    'scene-0953', 'scene-0955', 'scene-0956', 'scene-0957', 'scene-0958',
    'scene-0959', 'scene-0960', 'scene-0961', 'scene-0975', 'scene-0976',
    'scene-0977', 'scene-0978', 'scene-0979', 'scene-0980', 'scene-0981',
    'scene-0982', 'scene-0983', 'scene-0984', 'scene-0988', 'scene-0989',
    'scene-0990', 'scene-0991', 'scene-0992', 'scene-0994', 'scene-0995',
    'scene-0996', 'scene-0997', 'scene-0998', 'scene-0999', 'scene-1000',
    'scene-1001', 'scene-1002', 'scene-1003', 'scene-1004', 'scene-1005',
    'scene-1006', 'scene-1007', 'scene-1008', 'scene-1009', 'scene-1010',
    'scene-1011', 'scene-1012', 'scene-1013', 'scene-1014', 'scene-1015',
    'scene-1016', 'scene-1017', 'scene-1018', 'scene-1019', 'scene-1020',
    'scene-1021', 'scene-1022', 'scene-1023', 'scene-1024', 'scene-1025',
    'scene-1044', 'scene-1045', 'scene-1046', 'scene-1047', 'scene-1048',
    'scene-1049', 'scene-1050', 'scene-1051', 'scene-1052', 'scene-1053',
    'scene-1054', 'scene-1055', 'scene-1056', 'scene-1057', 'scene-1058',
    'scene-1074', 'scene-1075', 'scene-1076', 'scene-1077', 'scene-1078',
    'scene-1079', 'scene-1080', 'scene-1081', 'scene-1082', 'scene-1083',
    'scene-1084', 'scene-1085', 'scene-1086', 'scene-1087', 'scene-1088',
    'scene-1089', 'scene-1090', 'scene-1091', 'scene-1092', 'scene-1093',
    'scene-1094', 'scene-1095', 'scene-1096', 'scene-1097', 'scene-1098',
    'scene-1099', 'scene-1100', 'scene-1101', 'scene-1102', 'scene-1104',
    'scene-1105', 'scene-1106', 'scene-1107', 'scene-1108', 'scene-1109',
    'scene-1110',
]

VAL_SCENES_V1_0 = [
    'scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015',
    'scene-0016', 'scene-0017', 'scene-0018', 'scene-0035', 'scene-0036',
    'scene-0037', 'scene-0038', 'scene-0039', 'scene-0040', 'scene-0136',
    'scene-0137', 'scene-0140', 'scene-0141', 'scene-0142', 'scene-0143',
    'scene-0144', 'scene-0145', 'scene-0146', 'scene-0147', 'scene-0148',
    'scene-0149', 'scene-0153', 'scene-0154', 'scene-0156', 'scene-0186',
    'scene-0189', 'scene-0190', 'scene-0197', 'scene-0198', 'scene-0199',
    'scene-0200', 'scene-0201', 'scene-0202', 'scene-0203', 'scene-0204',
    'scene-0205', 'scene-0206', 'scene-0207', 'scene-0208', 'scene-0209',
    'scene-0214', 'scene-0215', 'scene-0216', 'scene-0217', 'scene-0218',
    'scene-0219', 'scene-0220', 'scene-0221', 'scene-0222', 'scene-0223',
    'scene-0265', 'scene-0266', 'scene-0267', 'scene-0268', 'scene-0269',
    'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274',
    'scene-0275', 'scene-0276', 'scene-0277', 'scene-0278', 'scene-0279',
    'scene-0280', 'scene-0281', 'scene-0282', 'scene-0307', 'scene-0308',
    'scene-0309', 'scene-0310', 'scene-0311', 'scene-0312', 'scene-0313',
    'scene-0314', 'scene-0315', 'scene-0316', 'scene-0317', 'scene-0318',
    'scene-0319', 'scene-0320', 'scene-0321', 'scene-0322', 'scene-0323',
    'scene-0324', 'scene-0325', 'scene-0326', 'scene-0327', 'scene-0328',
    'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
    'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521',
    'scene-0522', 'scene-0523', 'scene-0524', 'scene-0552', 'scene-0553',
    'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558',
    'scene-0559', 'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563',
    'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626', 'scene-0627',
    'scene-0628', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633',
    'scene-0634', 'scene-0635', 'scene-0636', 'scene-0637', 'scene-0638',
    'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778',
    'scene-0780', 'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784',
    'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797', 'scene-0798',
    'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905',
    'scene-0906', 'scene-0907', 'scene-0908', 'scene-0909', 'scene-0910',
    'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
    'scene-0916', 'scene-0917', 'scene-0918', 'scene-0919', 'scene-0920',
    'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924', 'scene-0925',
    'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930',
    'scene-0931', 'scene-0962', 'scene-0963', 'scene-0966', 'scene-0967',
    'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972',
]

MINI_TRAIN_SCENES = [
    'scene-0061', 'scene-0553', 'scene-0655', 'scene-0757',
    'scene-0796', 'scene-0916', 'scene-0996', 'scene-1077',
]

MINI_VAL_SCENES = [
    'scene-0103', 'scene-0916',
]

# =============================================================================
# nuScenes Sensor Configuration
# =============================================================================

CAMERAS = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_FRONT_LEFT',
]

RADARS = [
    'RADAR_FRONT',
    'RADAR_FRONT_LEFT',
    'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT',
    'RADAR_BACK_RIGHT',
]

# nuScenes detection classes used by CRAFT
DETECTION_CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

# Mapping from nuScenes category names to detection class names
CATEGORY_TO_DETECTION_CLASS = {
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'pedestrian',
    'human.pedestrian.stroller': 'pedestrian',
    'human.pedestrian.personal_mobility': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
}


# =============================================================================
# Data Loading Utilities
# =============================================================================

class NuScenesDatabase:
    """Lightweight nuScenes database parser that loads JSON tables directly."""

    def __init__(self, dataroot: str, version: str):
        """
        Initialize the database by loading all JSON tables.

        Args:
            dataroot: Path to the nuScenes dataset root directory.
            version: Dataset version string (e.g., 'v1.0-trainval', 'v1.0-mini').
        """
        self.dataroot = dataroot
        self.version = version
        self.table_root = os.path.join(dataroot, version)

        if not os.path.isdir(self.table_root):
            raise FileNotFoundError(
                f"Database table directory not found: {self.table_root}"
            )

        # Load all required tables
        print(f"Loading nuScenes tables from: {self.table_root}")
        self.scene = self._load_table('scene')
        self.sample = self._load_table('sample')
        self.sample_data = self._load_table('sample_data')
        self.ego_pose = self._load_table('ego_pose')
        self.calibrated_sensor = self._load_table('calibrated_sensor')
        self.sensor = self._load_table('sensor')
        self.log = self._load_table('log')

        # Optional tables (not present in test set)
        self.sample_annotation = self._load_table('sample_annotation', optional=True)
        self.instance = self._load_table('instance', optional=True)
        self.category = self._load_table('category', optional=True)
        self.attribute = self._load_table('attribute', optional=True)

        # Build token-to-record lookup dictionaries
        print("Building lookup indices...")
        self._scene_by_token = {r['token']: r for r in self.scene}
        self._sample_by_token = {r['token']: r for r in self.sample}
        self._sample_data_by_token = {r['token']: r for r in self.sample_data}
        self._ego_pose_by_token = {r['token']: r for r in self.ego_pose}
        self._calibrated_sensor_by_token = {r['token']: r for r in self.calibrated_sensor}
        self._sensor_by_token = {r['token']: r for r in self.sensor}
        self._log_by_token = {r['token']: r for r in self.log}

        if self.instance:
            self._instance_by_token = {r['token']: r for r in self.instance}
        if self.category:
            self._category_by_token = {r['token']: r for r in self.category}

        # Build sample_data index by sample_token and sensor channel
        self._sample_data_by_sample = defaultdict(list)
        for sd in self.sample_data:
            if sd['is_key_frame']:
                self._sample_data_by_sample[sd['sample_token']].append(sd)

        print(f"Loaded {len(self.scene)} scenes, {len(self.sample)} samples, "
              f"{len(self.sample_data)} sample_data records")

    def _load_table(self, table_name: str, optional: bool = False) -> List[Dict]:
        """Load a JSON table from the database directory."""
        filepath = os.path.join(self.table_root, f'{table_name}.json')
        if not os.path.isfile(filepath):
            if optional:
                print(f"  [Optional] {table_name}.json not found, skipping")
                return []
            raise FileNotFoundError(f"Required table not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"  Loaded {table_name}.json ({len(data)} records)")
        return data

    def get(self, table_name: str, token: str) -> Dict:
        """Get a record by its token from the specified table."""
        lookup = getattr(self, f'_{table_name}_by_token', None)
        if lookup is None:
            raise ValueError(f"Unknown table: {table_name}")
        record = lookup.get(token)
        if record is None:
            raise KeyError(f"Token {token} not found in {table_name}")
        return record

    def get_sample_data_for_sample(self, sample_token: str, channel: str) -> Optional[Dict]:
        """Get the key-frame sample_data record for a given sample and sensor channel."""
        for sd in self._sample_data_by_sample[sample_token]:
            cs = self.get('calibrated_sensor', sd['calibrated_sensor_token'])
            sensor = self.get('sensor', cs['sensor_token'])
            if sensor['channel'] == channel:
                return sd
        return None

    def get_scene_name(self, scene_token: str) -> str:
        """Get the scene name for a scene token."""
        return self.get('scene', scene_token)['name']


# =============================================================================
# Geometry Utilities
# =============================================================================

def quaternion_to_rotation_matrix(quaternion: List[float]) -> np.ndarray:
    """
    Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.

    Args:
        quaternion: Quaternion as [w, x, y, z].

    Returns:
        3x3 rotation matrix as numpy array.
    """
    q = Quaternion(quaternion)
    return q.rotation_matrix


def make_transform_matrix(translation: List[float], rotation: List[float]) -> np.ndarray:
    """
    Create a 4x4 homogeneous transformation matrix from translation and rotation.

    Args:
        translation: [x, y, z] translation vector.
        rotation: [w, x, y, z] quaternion.

    Returns:
        4x4 transformation matrix.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_rotation_matrix(rotation)
    T[:3, 3] = translation
    return T


def get_sensor_to_ego(calibrated_sensor: Dict) -> np.ndarray:
    """
    Get the sensor-to-ego transformation matrix from a calibrated_sensor record.

    Args:
        calibrated_sensor: nuScenes calibrated_sensor record.

    Returns:
        4x4 sensor-to-ego transformation matrix.
    """
    return make_transform_matrix(
        calibrated_sensor['translation'],
        calibrated_sensor['rotation']
    )


def get_ego_to_global(ego_pose: Dict) -> np.ndarray:
    """
    Get the ego-to-global transformation matrix from an ego_pose record.

    Args:
        ego_pose: nuScenes ego_pose record.

    Returns:
        4x4 ego-to-global transformation matrix.
    """
    return make_transform_matrix(
        ego_pose['translation'],
        ego_pose['rotation']
    )


def get_camera_intrinsic(calibrated_sensor: Dict) -> np.ndarray:
    """
    Get the 3x3 camera intrinsic matrix from a calibrated_sensor record.

    Args:
        calibrated_sensor: nuScenes calibrated_sensor record (must be a camera).

    Returns:
        3x3 camera intrinsic matrix.
    """
    return np.array(calibrated_sensor['camera_intrinsic'], dtype=np.float64)


# =============================================================================
# Radar Point Cloud Loading
# =============================================================================

def load_radar_pcd(filepath: str) -> np.ndarray:
    """
    Load a nuScenes radar PCD file.

    nuScenes radar PCD files store points with the following fields:
    x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, is_quality_valid,
    ambig_state, x_rms, y_rms, invalid_state, pdh0, vx_rms, vy_rms

    Args:
        filepath: Path to the .pcd file.

    Returns:
        Numpy array of shape (N, 18) with radar point attributes.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Radar PCD file not found: {filepath}")

    # Read the PCD header to determine format and point count
    with open(filepath, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line.startswith('DATA'):
                break

        # Parse header
        num_points = 0
        fields = []
        sizes = []
        types = []
        data_format = ''

        for line in header_lines:
            if line.startswith('FIELDS'):
                fields = line.split()[1:]
            elif line.startswith('SIZE'):
                sizes = [int(s) for s in line.split()[1:]]
            elif line.startswith('TYPE'):
                types = line.split()[1:]
            elif line.startswith('POINTS'):
                num_points = int(line.split()[1])
            elif line.startswith('DATA'):
                data_format = line.split()[1]

        if num_points == 0:
            return np.zeros((0, 18), dtype=np.float32)

        # Read binary data
        if data_format == 'binary':
            # Calculate point size from sizes
            point_size = sum(sizes)
            data = f.read(num_points * point_size)

            # Parse points
            points = np.zeros((num_points, len(fields)), dtype=np.float32)
            offset = 0
            for i in range(num_points):
                field_offset = 0
                for j, (size, type_char) in enumerate(zip(sizes, types)):
                    raw = data[offset + field_offset:offset + field_offset + size]
                    if type_char == 'F':
                        if size == 4:
                            points[i, j] = struct.unpack('f', raw)[0]
                        elif size == 8:
                            points[i, j] = struct.unpack('d', raw)[0]
                    elif type_char == 'I':
                        if size == 1:
                            points[i, j] = struct.unpack('B', raw)[0]
                        elif size == 2:
                            points[i, j] = struct.unpack('H', raw)[0]
                        elif size == 4:
                            points[i, j] = struct.unpack('I', raw)[0]
                    elif type_char == 'U':
                        if size == 1:
                            points[i, j] = struct.unpack('B', raw)[0]
                        elif size == 2:
                            points[i, j] = struct.unpack('H', raw)[0]
                        elif size == 4:
                            points[i, j] = struct.unpack('I', raw)[0]
                    field_offset += size
                offset += point_size

            return points

        elif data_format == 'binary_compressed':
            # Read compressed size and uncompressed size
            compressed_size = struct.unpack('I', f.read(4))[0]
            uncompressed_size = struct.unpack('I', f.read(4))[0]

            # Read and decompress
            import lzf
            compressed_data = f.read(compressed_size)
            data = lzf.decompress(compressed_data, uncompressed_size)

            # Column-major storage in compressed format
            points = np.zeros((num_points, len(fields)), dtype=np.float32)
            offset = 0
            for j, (size, type_char) in enumerate(zip(sizes, types)):
                col_data = data[offset:offset + num_points * size]
                if type_char == 'F' and size == 4:
                    points[:, j] = np.frombuffer(col_data, dtype=np.float32)
                elif type_char == 'F' and size == 8:
                    points[:, j] = np.frombuffer(col_data, dtype=np.float64).astype(np.float32)
                elif type_char in ('I', 'U') and size == 4:
                    points[:, j] = np.frombuffer(col_data, dtype=np.uint32).astype(np.float32)
                elif type_char in ('I', 'U') and size == 2:
                    points[:, j] = np.frombuffer(col_data, dtype=np.uint16).astype(np.float32)
                elif type_char in ('I', 'U') and size == 1:
                    points[:, j] = np.frombuffer(col_data, dtype=np.uint8).astype(np.float32)
                offset += num_points * size

            return points

        else:
            # ASCII format
            points = []
            for line in f:
                line = line.decode('ascii', errors='ignore').strip()
                if line:
                    values = [float(v) for v in line.split()]
                    points.append(values)
            if points:
                return np.array(points, dtype=np.float32)
            return np.zeros((0, 18), dtype=np.float32)


def accumulate_radar_sweeps(
    db: NuScenesDatabase,
    current_sd_token: str,
    num_sweeps: int,
    dataroot: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accumulate past radar sweeps with ego-motion compensation.

    For each past sweep, the radar points are transformed from:
        past_radar -> past_ego -> global -> current_ego -> current_radar

    This aligns all points to the current radar frame coordinate system.

    Args:
        db: NuScenes database instance.
        current_sd_token: Token of the current (key-frame) sample_data.
        num_sweeps: Number of sweeps to accumulate (including the current one).
        dataroot: Path to the nuScenes data root.

    Returns:
        Tuple of:
            - points: (N, 5) array with [x, y, z, vx_comp, vy_comp] in current
              radar frame.
            - timestamps: (N,) array of relative timestamps (seconds, current=0,
              past=negative).
    """
    current_sd = db.get('sample_data', current_sd_token)
    current_cs = db.get('calibrated_sensor', current_sd['calibrated_sensor_token'])
    current_ep = db.get('ego_pose', current_sd['ego_pose_token'])

    # Current frame transforms
    current_sensor2ego = get_sensor_to_ego(current_cs)
    current_ego2global = get_ego_to_global(current_ep)
    # Inverse: global -> current_ego -> current_sensor
    current_global2ego = np.linalg.inv(current_ego2global)
    current_ego2sensor = np.linalg.inv(current_sensor2ego)

    current_timestamp = current_sd['timestamp']

    all_points = []
    all_timestamps = []

    # Traverse the linked list of sample_data (prev pointers)
    sd_token = current_sd_token
    sweep_count = 0

    while sd_token != '' and sweep_count < num_sweeps:
        sd = db.get('sample_data', sd_token)
        cs = db.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ep = db.get('ego_pose', sd['ego_pose_token'])

        # Load the radar point cloud
        pcd_path = os.path.join(dataroot, sd['filename'])
        try:
            raw_points = load_radar_pcd(pcd_path)
        except (FileNotFoundError, Exception) as e:
            print(f"  Warning: Could not load {pcd_path}: {e}")
            sd_token = sd.get('prev', '')
            sweep_count += 1
            continue

        if raw_points.shape[0] == 0:
            sd_token = sd.get('prev', '')
            sweep_count += 1
            continue

        # Extract xyz and compensated velocities
        # Fields: x=0, y=1, z=2, ..., vx_comp=8, vy_comp=9
        xyz = raw_points[:, :3]  # (N, 3)
        vx_comp = raw_points[:, 8] if raw_points.shape[1] > 8 else np.zeros(len(xyz))
        vy_comp = raw_points[:, 9] if raw_points.shape[1] > 9 else np.zeros(len(xyz))

        # Transform points to the current radar frame
        # past_sensor -> past_ego -> global -> current_ego -> current_sensor
        past_sensor2ego = get_sensor_to_ego(cs)
        past_ego2global = get_ego_to_global(ep)

        # Homogeneous coordinates
        ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
        xyz_h = np.hstack([xyz, ones])  # (N, 4)

        # Chain the transforms
        transform = current_ego2sensor @ current_global2ego @ past_ego2global @ past_sensor2ego
        xyz_current = (transform @ xyz_h.T).T[:, :3]  # (N, 3)

        # Transform velocities (rotation only, no translation)
        rotation = transform[:3, :3]
        velocities = np.stack([vx_comp, vy_comp, np.zeros_like(vx_comp)], axis=1)  # (N, 3)
        velocities_current = (rotation @ velocities.T).T  # (N, 3)

        # Combine: [x, y, z, vx_comp, vy_comp]
        points = np.hstack([
            xyz_current,
            velocities_current[:, :2],  # Only vx, vy
        ]).astype(np.float32)

        # Relative timestamp (seconds)
        rel_time = (sd['timestamp'] - current_timestamp) / 1e6  # microseconds to seconds
        timestamps = np.full(len(points), rel_time, dtype=np.float32)

        all_points.append(points)
        all_timestamps.append(timestamps)

        # Move to previous sweep
        sd_token = sd.get('prev', '')
        sweep_count += 1

    if all_points:
        return np.vstack(all_points), np.concatenate(all_timestamps)
    else:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.float32)


# =============================================================================
# Info Generation
# =============================================================================

def get_annotations_for_sample(
    db: NuScenesDatabase,
    sample_token: str,
) -> List[Dict[str, Any]]:
    """
    Get all 3D bounding box annotations for a given sample.

    Args:
        db: NuScenes database instance.
        sample_token: Token of the sample.

    Returns:
        List of annotation dictionaries, each containing:
            - category_name: Full nuScenes category name
            - detection_name: Mapped detection class name (or None if not in classes)
            - translation: [x, y, z] center in global frame
            - size: [w, l, h] in meters
            - rotation: [w, x, y, z] quaternion
            - velocity: [vx, vy] in m/s (global frame)
            - num_lidar_pts: Number of lidar points in the box
            - num_radar_pts: Number of radar points in the box
            - instance_token: Instance token for tracking
    """
    if not db.sample_annotation:
        return []

    sample = db.get('sample', sample_token)
    annotations = []

    # Get all annotation tokens for this sample
    ann_tokens = sample.get('anns', [])

    for ann_token in ann_tokens:
        # Find the annotation record
        ann = None
        for a in db.sample_annotation:
            if a['token'] == ann_token:
                ann = a
                break

        if ann is None:
            continue

        # Get category name
        instance = db._instance_by_token.get(ann['instance_token'], {})
        category_token = instance.get('category_token', '')
        category = db._category_by_token.get(category_token, {})
        category_name = category.get('name', 'unknown')

        # Map to detection class
        detection_name = None
        for prefix, det_class in CATEGORY_TO_DETECTION_CLASS.items():
            if category_name.startswith(prefix):
                detection_name = det_class
                break

        # Compute velocity from annotation attributes
        # nuScenes provides velocity as part of the annotation
        velocity = [0.0, 0.0]
        if 'velocity' in ann:
            velocity = ann['velocity'][:2] if ann['velocity'] is not None else [0.0, 0.0]
        else:
            # Compute velocity from previous/next annotations if available
            velocity = _compute_velocity(db, ann)

        annotations.append({
            'category_name': category_name,
            'detection_name': detection_name,
            'translation': ann['translation'],
            'size': ann['size'],
            'rotation': ann['rotation'],
            'velocity': velocity,
            'num_lidar_pts': ann.get('num_lidar_pts', 0),
            'num_radar_pts': ann.get('num_radar_pts', 0),
            'instance_token': ann['instance_token'],
            'token': ann['token'],
        })

    return annotations


def _compute_velocity(db: NuScenesDatabase, annotation: Dict) -> List[float]:
    """
    Compute velocity for an annotation by differencing consecutive positions.

    Args:
        db: NuScenes database instance.
        annotation: The annotation record.

    Returns:
        [vx, vy] velocity in global frame (m/s).
    """
    prev_token = annotation.get('prev', '')
    next_token = annotation.get('next', '')

    if not prev_token and not next_token:
        return [0.0, 0.0]

    # Find timestamps via sample
    current_sample = db.get('sample', annotation['sample_token'])
    current_time = current_sample['timestamp'] / 1e6  # to seconds

    if prev_token:
        # Find prev annotation
        prev_ann = None
        for a in db.sample_annotation:
            if a['token'] == prev_token:
                prev_ann = a
                break
        if prev_ann:
            prev_sample = db.get('sample', prev_ann['sample_token'])
            prev_time = prev_sample['timestamp'] / 1e6
            dt = current_time - prev_time
            if abs(dt) > 1e-6:
                vx = (annotation['translation'][0] - prev_ann['translation'][0]) / dt
                vy = (annotation['translation'][1] - prev_ann['translation'][1]) / dt
                return [vx, vy]

    if next_token:
        # Find next annotation
        next_ann = None
        for a in db.sample_annotation:
            if a['token'] == next_token:
                next_ann = a
                break
        if next_ann:
            next_sample = db.get('sample', next_ann['sample_token'])
            next_time = next_sample['timestamp'] / 1e6
            dt = next_time - current_time
            if abs(dt) > 1e-6:
                vx = (next_ann['translation'][0] - annotation['translation'][0]) / dt
                vy = (next_ann['translation'][1] - annotation['translation'][1]) / dt
                return [vx, vy]

    return [0.0, 0.0]


def create_sample_info(
    db: NuScenesDatabase,
    sample_token: str,
    dataroot: str,
    num_sweeps: int = 6,
) -> Dict[str, Any]:
    """
    Create a complete info dictionary for one sample.

    This contains all information needed for training/inference:
    - Camera image paths and calibrations
    - Radar point cloud paths and calibrations
    - Ego pose
    - Precomputed radar-to-camera projection matrices
    - 3D bounding box annotations

    Args:
        db: NuScenes database instance.
        sample_token: Token of the sample to process.
        dataroot: Path to nuScenes data root.
        num_sweeps: Number of radar sweeps to accumulate.

    Returns:
        Dictionary containing all sample information.
    """
    sample = db.get('sample', sample_token)
    scene = db.get('scene', sample['scene_token'])

    info = {
        'sample_token': sample_token,
        'scene_token': sample['scene_token'],
        'scene_name': scene['name'],
        'timestamp': sample['timestamp'],
        'prev_sample_token': sample.get('prev', ''),
        'next_sample_token': sample.get('next', ''),
    }

    # =========================================================================
    # Camera data
    # =========================================================================
    camera_info = {}
    for cam_channel in CAMERAS:
        cam_sd = db.get_sample_data_for_sample(sample_token, cam_channel)
        if cam_sd is None:
            print(f"  Warning: No sample_data for {cam_channel} in sample {sample_token}")
            continue

        cam_cs = db.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        cam_ep = db.get('ego_pose', cam_sd['ego_pose_token'])

        # Transformation matrices
        sensor2ego = get_sensor_to_ego(cam_cs)
        ego2global = get_ego_to_global(cam_ep)
        intrinsic = get_camera_intrinsic(cam_cs)

        camera_info[cam_channel] = {
            'data_path': cam_sd['filename'],
            'sample_data_token': cam_sd['token'],
            'sensor2ego': sensor2ego.tolist(),
            'ego2global': ego2global.tolist(),
            'intrinsic': intrinsic.tolist(),
            'timestamp': cam_sd['timestamp'],
            'width': cam_sd.get('width', 1600),
            'height': cam_sd.get('height', 900),
        }

    info['cameras'] = camera_info

    # =========================================================================
    # Radar data
    # =========================================================================
    radar_info = {}
    for radar_channel in RADARS:
        radar_sd = db.get_sample_data_for_sample(sample_token, radar_channel)
        if radar_sd is None:
            print(f"  Warning: No sample_data for {radar_channel} in sample {sample_token}")
            continue

        radar_cs = db.get('calibrated_sensor', radar_sd['calibrated_sensor_token'])
        radar_ep = db.get('ego_pose', radar_sd['ego_pose_token'])

        # Transformation matrices
        sensor2ego = get_sensor_to_ego(radar_cs)
        ego2global = get_ego_to_global(radar_ep)

        radar_info[radar_channel] = {
            'data_path': radar_sd['filename'],
            'sample_data_token': radar_sd['token'],
            'sensor2ego': sensor2ego.tolist(),
            'ego2global': ego2global.tolist(),
            'timestamp': radar_sd['timestamp'],
            'num_sweeps': num_sweeps,
        }

    info['radars'] = radar_info

    # =========================================================================
    # Radar-to-Camera projection matrices
    # =========================================================================
    # Precompute the full projection matrix from each radar to each camera
    # radar_point_cam = K @ cam_ego2sensor @ radar_ego2global_inv @ radar_sensor2ego @ point_radar
    # Simplified: project_matrix = K @ T_cam_from_radar (3x4 submatrix of 4x4)

    radar_to_camera_projections = {}
    for radar_channel, r_info in radar_info.items():
        radar_sensor2ego = np.array(r_info['sensor2ego'])
        radar_ego2global = np.array(r_info['ego2global'])

        for cam_channel, c_info in camera_info.items():
            cam_sensor2ego = np.array(c_info['sensor2ego'])
            cam_ego2global = np.array(c_info['ego2global'])
            cam_intrinsic = np.array(c_info['intrinsic'])

            # Full transform: radar -> ego -> global -> ego -> camera
            cam_ego2sensor = np.linalg.inv(cam_sensor2ego)
            cam_global2ego = np.linalg.inv(cam_ego2global)

            # radar_sensor -> radar_ego -> global -> cam_ego -> cam_sensor
            T_cam_from_radar = (
                cam_ego2sensor @ cam_global2ego @ radar_ego2global @ radar_sensor2ego
            )

            # Projection matrix (3x4): K @ [R|t] where [R|t] is the 3x4 part of T
            projection = cam_intrinsic @ T_cam_from_radar[:3, :]

            key = f"{radar_channel}_to_{cam_channel}"
            radar_to_camera_projections[key] = projection.tolist()

    info['radar_to_camera_projections'] = radar_to_camera_projections

    # =========================================================================
    # Ego pose (for the reference timestamp - typically lidar timestamp)
    # =========================================================================
    # Use the front camera ego_pose as reference
    if 'CAM_FRONT' in camera_info:
        ref_ep_token = None
        cam_front_sd = db.get_sample_data_for_sample(sample_token, 'CAM_FRONT')
        if cam_front_sd:
            ref_ep = db.get('ego_pose', cam_front_sd['ego_pose_token'])
            info['ego2global'] = get_ego_to_global(ref_ep).tolist()
            info['ego_translation'] = ref_ep['translation']
            info['ego_rotation'] = ref_ep['rotation']

    # =========================================================================
    # Annotations (3D bounding boxes)
    # =========================================================================
    annotations = get_annotations_for_sample(db, sample_token)

    # Filter to detection classes only
    valid_annotations = []
    for ann in annotations:
        if ann['detection_name'] is not None:
            valid_annotations.append(ann)

    info['annotations'] = valid_annotations
    info['num_annotations'] = len(valid_annotations)

    return info


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def get_split_scenes(version: str) -> Tuple[List[str], List[str]]:
    """
    Get train and val scene lists based on the dataset version.

    Args:
        version: Dataset version string.

    Returns:
        Tuple of (train_scenes, val_scenes).
    """
    if 'mini' in version:
        return MINI_TRAIN_SCENES, MINI_VAL_SCENES
    else:
        return TRAIN_SCENES_V1_0, VAL_SCENES_V1_0


def process_dataset(
    dataroot: str,
    version: str,
    out_dir: str,
    num_sweeps: int = 6,
    workers: int = 1,
) -> None:
    """
    Process the entire dataset and generate info pickle files.

    Args:
        dataroot: Path to nuScenes dataset root.
        version: Dataset version (e.g., 'v1.0-trainval', 'v1.0-mini').
        out_dir: Output directory for pickle files.
        num_sweeps: Number of radar sweeps to accumulate per sample.
        workers: Number of parallel workers (currently single-threaded).
    """
    print("=" * 80)
    print("  CRAFT Data Preparation")
    print("=" * 80)
    print(f"\n  Data root:    {dataroot}")
    print(f"  Version:      {version}")
    print(f"  Output dir:   {out_dir}")
    print(f"  Num sweeps:   {num_sweeps}")
    print(f"  Workers:      {workers}")
    print()

    # Load database
    db = NuScenesDatabase(dataroot, version)

    # Get split scene lists
    train_scenes, val_scenes = get_split_scenes(version)

    # Build set of scene names for quick lookup
    train_scene_set = set(train_scenes)
    val_scene_set = set(val_scenes)

    # Categorize samples into train/val
    train_samples = []
    val_samples = []
    skipped_samples = []

    for sample in db.sample:
        scene = db.get('scene', sample['scene_token'])
        scene_name = scene['name']

        if scene_name in train_scene_set:
            train_samples.append(sample['token'])
        elif scene_name in val_scene_set:
            val_samples.append(sample['token'])
        else:
            skipped_samples.append(sample['token'])

    print(f"\nSplit statistics:")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples:   {len(val_samples)}")
    print(f"  Skipped:       {len(skipped_samples)}")
    print()

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Process train split
    print("-" * 80)
    print("Processing TRAIN split...")
    print("-" * 80)
    train_infos = []
    for i, sample_token in enumerate(train_samples):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing sample {i + 1}/{len(train_samples)}...")
        try:
            info = create_sample_info(db, sample_token, dataroot, num_sweeps)
            train_infos.append(info)
        except Exception as e:
            print(f"  Error processing sample {sample_token}: {e}")
            continue

    # Process val split
    print("-" * 80)
    print("Processing VAL split...")
    print("-" * 80)
    val_infos = []
    for i, sample_token in enumerate(val_samples):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing sample {i + 1}/{len(val_samples)}...")
        try:
            info = create_sample_info(db, sample_token, dataroot, num_sweeps)
            val_infos.append(info)
        except Exception as e:
            print(f"  Error processing sample {sample_token}: {e}")
            continue

    # Save pickle files
    prefix = "craft_infos_mini" if 'mini' in version else "craft_infos"

    train_pkl_path = os.path.join(out_dir, f"{prefix}_train.pkl")
    val_pkl_path = os.path.join(out_dir, f"{prefix}_val.pkl")

    print(f"\nSaving train infos to: {train_pkl_path}")
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train_infos, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saving val infos to: {val_pkl_path}")
    with open(val_pkl_path, 'wb') as f:
        pickle.dump(val_infos, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Print summary statistics
    print_statistics(train_infos, val_infos, prefix)


def print_statistics(
    train_infos: List[Dict],
    val_infos: List[Dict],
    prefix: str,
) -> None:
    """Print dataset statistics after processing."""
    print("\n" + "=" * 80)
    print("  DATASET STATISTICS")
    print("=" * 80)

    for split_name, infos in [("Train", train_infos), ("Val", val_infos)]:
        print(f"\n  {split_name} Split:")
        print(f"  {'─' * 40}")
        print(f"    Samples:     {len(infos)}")

        if not infos:
            continue

        # Count annotations by class
        class_counts = defaultdict(int)
        total_annotations = 0
        for info in infos:
            for ann in info.get('annotations', []):
                det_name = ann.get('detection_name')
                if det_name:
                    class_counts[det_name] += 1
                    total_annotations += 1

        print(f"    Annotations: {total_annotations}")
        print(f"\n    {'Class':<25} {'Count':>8} {'Percentage':>12}")
        print(f"    {'─' * 50}")
        for cls in DETECTION_CLASSES:
            count = class_counts.get(cls, 0)
            pct = 100.0 * count / total_annotations if total_annotations > 0 else 0
            print(f"    {cls:<25} {count:>8} {pct:>10.1f}%")

        # Camera and radar coverage
        num_cams = sum(1 for info in infos for _ in info.get('cameras', {}))
        num_radars = sum(1 for info in infos for _ in info.get('radars', {}))
        avg_cams = num_cams / len(infos) if infos else 0
        avg_radars = num_radars / len(infos) if infos else 0
        print(f"\n    Avg cameras/sample:  {avg_cams:.1f}")
        print(f"    Avg radars/sample:   {avg_radars:.1f}")

        # Unique scenes
        scenes = set(info.get('scene_name', '') for info in infos)
        print(f"    Unique scenes:       {len(scenes)}")

    print("\n" + "=" * 80)
    print("  Processing complete!")
    print("=" * 80 + "\n")


# =============================================================================
# Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare nuScenes data for CRAFT model training/evaluation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process mini dataset for quick testing
  python prepare_data.py --dataroot /data/nuscenes --version v1.0-mini --out-dir ./data

  # Process full trainval dataset
  python prepare_data.py --dataroot /data/nuscenes --version v1.0-trainval --out-dir ./data

  # Process with more radar sweeps
  python prepare_data.py --dataroot /data/nuscenes --version v1.0-trainval \\
      --out-dir ./data --num-sweeps 10
        """
    )

    parser.add_argument(
        '--dataroot',
        type=str,
        required=True,
        help='Path to the nuScenes dataset root directory'
    )
    parser.add_argument(
        '--version',
        type=str,
        required=True,
        choices=['v1.0-trainval', 'v1.0-mini', 'v1.0-test'],
        help='nuScenes dataset version'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        required=True,
        help='Output directory for generated pickle files'
    )
    parser.add_argument(
        '--num-sweeps',
        type=int,
        default=6,
        help='Number of radar sweeps to accumulate (default: 6)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_dataset(
        dataroot=args.dataroot,
        version=args.version,
        out_dir=args.out_dir,
        num_sweeps=args.num_sweeps,
        workers=args.workers,
    )
