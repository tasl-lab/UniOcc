# Copyright (c) 2025. All rights reserved.
# Licensed under the MIT License.

import logging
import os
import pickle
from functools import lru_cache
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class UniOcc(Dataset):
    """
    A generic dataset for occupancy forecasting tasks, supporting both past (observation)
    and future (forecast) frames.
    """

    def __init__(self, data_root, obs_len, fut_len, datasource_root=None):
        """
        Args:
            uniocc_path (str): Root directory holding the occupancy data. eg. "unified_datasets/NuScenes-via-Occ3D-2Hz-mini"
            obs_len (int): Number of past frames to use as input.
            fut_len (int): Number of future frames to forecast.
            datasource_path (str): [Only for camera images] Root directory to nuScenes, Waymo, OpV2V, etc. datasets.
        """
        self.data_root = data_root
        self.obs_len = obs_len
        self.fut_len = fut_len
        self.provide_camera = datasource_root is not None
        self.datasource_root = datasource_root

        if datasource_root is not None:
            assert os.path.exists(datasource_root), f"Datasource root {datasource_root} not found."

        # Load scene meta-info.
        with open(os.path.join(data_root, 'scene_infos.pkl'), 'rb') as f:
            self.scene_infos = pickle.load(f)

        self.instances = []
        for scene in self.scene_infos:
            scene_name = scene["scene_name"]
            scene_files = scene["occ_in_scene_paths"]

            # Verify scene length sufficiency.
            if len(scene_files) < (self.obs_len + self.fut_len):
                logging.warning(
                    f"Skipping scene {scene_name} - insufficient frames ({len(scene_files)})."
                )
                continue

            # Construct sliding-window segments.
            limit = len(scene_files) - (self.obs_len + self.fut_len) + 1
            for i in range(limit):
                sub_seq = scene_files[i : i + self.obs_len + self.fut_len]
                self.instances.append((scene_name, sub_seq))

        logging.info(f"Loaded {len(self.instances)} instances from {data_root}.")

    def __len__(self):
        return len(self.instances)

    def __pad_on_dim__(self, arr, pad_to, axis):
        """
        Pad an array to a specific length along a given axis.
        """
        pad_len = pad_to - arr.shape[axis]
        padding = [(0, 0)] * len(arr.shape)
        padding[axis] = (0, pad_len)
        return np.pad(arr, padding, mode="constant", constant_values=0)

    @lru_cache(maxsize=5000)
    def __load_file__(self, path):
        """
        Load and cache NumPy files to avoid repeated I/O overhead.
        """
        return np.load(path, allow_pickle=True)

    @lru_cache(maxsize=5000)
    def __load_image__(self, path):
        """
        Load and cache image files to avoid repeated I/O overhead.
        """
        return np.array(Image.open(path))

    def __getitem__(self, index):
        """
        Retrieve the observation and future sequences for a single training instance.
        """
        scene_name, occ_paths = self.instances[index]
        scene_tokens = [
            os.path.splitext(os.path.basename(path))[0] for path in occ_paths
        ]

        # Load occupancy labels
        occ_labels = [
            self.__load_file__(os.path.join(self.data_root, path))["occ_label"]
            for path in occ_paths
        ]
        occ_labels = np.array(occ_labels)

        # Load forward flows
        flows_forward = [
            self.__load_file__(os.path.join(self.data_root, path))["occ_flow_forward"]
            for path in occ_paths
        ]
        flows_forward = np.array(flows_forward)

        # Load backward flows
        flows_backward = [
            self.__load_file__(os.path.join(self.data_root, path))["occ_flow_backward"]
            for path in occ_paths
        ]
        flows_backward = np.array(flows_backward)

        # Load ego-to-world transformations
        ego_to_worlds = [
            self.__load_file__(os.path.join(self.data_root, path))[
                "ego_to_world_transformation"
            ]
            for path in occ_paths
        ]
        ego_to_worlds = np.array(ego_to_worlds)

        # Load annotations
        ann_dicts = [
            self.__load_file__(os.path.join(self.data_root, path))["annotations"]
            for path in occ_paths
        ]

        # Load camera FOV Mask
        cameras_fov_masks = [
            self.__load_file__(os.path.join(self.data_root, path))["occ_mask_camera"]
            for path in occ_paths
        ]

        cameras_image = []
        cameras_intrinsics = []
        cameras_extrinsics = []
        num_cameras = 0
        if self.provide_camera:
            # Load per-camera data
            cameras_infos = [
                self.__load_file__(os.path.join(self.data_root, path))["cameras"]
                for path in occ_paths
            ]
            num_cameras = len(cameras_infos[0])

            for t in range(len(occ_paths)):
                cameras_image.append([self.__load_image__(os.path.join(self.datasource_root, camera_info['filename']))
                                      for camera_info in cameras_infos[t]])
                cameras_intrinsics.append([camera_info['intrinsics'] for camera_info in cameras_infos[t]])
                cameras_extrinsics.append([camera_info['extrinsics'] for camera_info in cameras_infos[t]])

            # Stack over time
            cameras_image = np.array(cameras_image)
            cameras_intrinsics = np.array(cameras_intrinsics)
            cameras_extrinsics = np.array(cameras_extrinsics)

            # Pad to the max number of cameras (for example we assume 6 here).
            cameras_image = self.__pad_on_dim__(cameras_image, 6, axis=1)
            cameras_intrinsics = self.__pad_on_dim__(cameras_intrinsics, 6, axis=1)
            cameras_extrinsics = self.__pad_on_dim__(cameras_extrinsics, 6, axis=1)

        return {
            "scene_name":               scene_name,                         # str
            "scene_token":              scene_tokens[self.obs_len],         # str
            "instance_path":            occ_paths[self.obs_len],            # str
            "obs_occ_labels":           occ_labels[:self.obs_len],          # (obs_len, L, W, H)
            "fut_occ_labels":           occ_labels[self.obs_len:],          # (fut_len, L, W, H)
            "obs_flows_forward":        flows_forward[:self.obs_len],       # (obs_len, L, W, H, 3)
            "fut_flows_forward":        flows_forward[self.obs_len:],       # (fut_len, L, W, H, 3)
            "obs_flows_backward":       flows_backward[:self.obs_len],      # (obs_len, L, W, H, 3)
            "fut_flows_backward":       flows_backward[self.obs_len:],      # (fut_len, L, W, H, 3)
            "obs_ego_to_worlds":        ego_to_worlds[:self.obs_len],       # (obs_len, 4, 4)
            "fut_ego_to_worlds":        ego_to_worlds[self.obs_len:],       # (fut_len, 4, 4)
            "num_cameras":              num_cameras,                        # int
            "obs_cameras_image":        cameras_image[:self.obs_len],       # (obs_len, MAX_CAM, H, W, 3)  [0 ~ 255]
            "fut_cameras_image":        cameras_image[self.obs_len:],       # (fut_len, MAX_CAM, H, W, 3)  [0 ~ 255]
            "obs_cameras_fov_mask":     cameras_fov_masks[:self.obs_len],   # (obs_len, L, W, H)         Binary mask
            "fut_cameras_fov_mask":     cameras_fov_masks[self.obs_len:],   # (fut_len, L, W, H)         Binary mask
            "obs_cameras_intrinsics":   cameras_intrinsics[:self.obs_len],  # (obs_len, MAX_CAM, 3, 3)
            "fut_cameras_intrinsics":   cameras_intrinsics[self.obs_len:],  # (fut_len, MAX_CAM, 3, 3)
            "obs_cameras_extrinsics":   cameras_extrinsics[:self.obs_len],  # (obs_len, MAX_CAM, 4, 4)
            "fut_cameras_extrinsics":   cameras_extrinsics[self.obs_len:],  # (fut_len, MAX_CAM, 4, 4)
            "annotations":              ann_dicts                           # (obs_fut_len) of (num_annotations) dicts
        }

    @staticmethod
    def collate_fn(batch):
        """
        Merge a list of samples into a unified batch for efficient processing.
        """
        collated_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                collated_dict[key] = torch.tensor(np.stack([inst[key] for inst in batch]))
            else:  # List, Int
                collated_dict[key] = [inst[key] for inst in batch]
        return collated_dict


if __name__ == "__main__":
    # Example usage
    dataset_nusc_mini = UniOcc(
        data_root="datasets/NuScenes-via-Occ3D-2Hz-mini",
        obs_len=8,
        fut_len=12
    )

    dataset_carla_mini = UniOcc(
        data_root="datasets/Carla-2Hz-mini",
        obs_len=8,
        fut_len=12
    )

    # Concatenate multiple datasets
    train_set = torch.utils.data.ConcatDataset([dataset_nusc_mini, dataset_carla_mini])
    dataloader = DataLoader(train_set, batch_size=2, collate_fn=UniOcc.collate_fn)  # Batch size 2

    for batch_data in dataloader:
        print("Batch keys:", batch_data.keys())
        print("Observation occupancy shape:", batch_data["obs_occ_labels"].shape)
        print("Future occupancy shape:", batch_data["obs_occ_labels"].shape)
        break

    # To use the camera images, set the datasource_root argument.
    dataset_carla_mini = UniOcc(
        data_root="datasets/Carla-2Hz-mini",
        obs_len=8,
        fut_len=12,
        datasource_root="datasets/Carla-2Hz-mini"
    )

    dataloader = DataLoader(dataset_carla_mini, batch_size=2, collate_fn=UniOcc.collate_fn)  # Batch size 2

    for batch_data in dataloader:
        print("Batch keys:", batch_data.keys())
        print("Observation camera images shape:", batch_data["obs_cameras_image"].shape)
        print("Future camera images shape:", batch_data["fut_cameras_image"].shape)
        break