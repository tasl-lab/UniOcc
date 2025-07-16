# Copyright (c) 2025. All rights reserved.
# Licensed under the MIT License.
#
# This script contains the methods to find voxel-level flow,
# as described in the UniOcc paper https://arxiv.org/abs/2503.24381.

import numpy as np
from uniocc_utils import (
    OccFrameToEgoFrame,
    GetVoxelCoordinates
)

def ComputeFlowsForObjects(
    occ: np.ndarray,
    curr_annotations: dict,
    next_annotations: dict,
    curr_ego_to_next_ego: np.ndarray,
    labels_of_interest: list[int],
    grid_resolution: float = 0.4
) -> np.ndarray:
    """
    Computes voxel-level flow for each object across frames.

    Parameters:
      occ: (L, W, H) occupancy grid with semantic labels.
      curr_annotations: dict mapping obj_id to {'agent_to_ego': 4×4, 'size': [l,w,h]} at current frame.
      next_annotations: same mapping for the next frame.
      curr_ego_to_next_ego: 4×4 ego SE(3) transform from current to next frame in current ego frame.
      labels_of_interest: list of semantic labels to include (e.g., 1 for cars).
      grid_resolution: size (m) per voxel.

    Returns:
      flows: (L, W, H, 3) float32 flow field in num of voxels.
    """
    dynamic_flows = np.zeros((*occ.shape, 3), dtype=np.float32)
    dynamic_category_mask = np.isin(occ, labels_of_interest)

    for obj_id, curr_ann in curr_annotations.items():
        if obj_id not in next_annotations:
            continue
        next_ann = next_annotations[obj_id]
        curr_agent_to_curr_ego = curr_ann['agent_to_ego']
        next_agent_to_next_ego = next_ann['agent_to_ego']
        size = np.array(curr_ann['size'])

        agent_voxels, _ = GetVoxelCoordinates(
            np.array([-40, -40, -1.0]),  # Modify this if the grid range is different.
            np.array([40, 40, 5.4]),
            grid_resolution,
            curr_agent_to_curr_ego, size
        )
        if len(agent_voxels) == 0:
            continue
        agent_voxels = np.array(agent_voxels)
        agent_coords = OccFrameToEgoFrame(agent_voxels, grid_resolution=grid_resolution)
        agent_coords = np.concatenate(
            [agent_coords, np.ones((agent_coords.shape[0], 1))], axis=1)

        # Compute relative pose
        curr_agent_to_next_agent = np.linalg.inv(next_agent_to_next_ego) @ curr_ego_to_next_ego @ curr_agent_to_curr_ego
        curr_agent_move_next_agent = np.linalg.inv(curr_agent_to_next_agent)

        curr_ego_to_curr_agent = np.linalg.inv(curr_agent_to_curr_ego)
        ego_frame_movement_forward = curr_ego_to_next_ego @ curr_agent_to_curr_ego @ curr_agent_move_next_agent @ curr_ego_to_curr_agent
        agent_flow = (ego_frame_movement_forward @ agent_coords.T - agent_coords.T).T

        temp_flow = np.zeros_like(dynamic_flows)  # allocate flow for the current object
        temp_flow[agent_voxels[:, 0], agent_voxels[:, 1], agent_voxels[:, 2], :] = agent_flow[..., :3] / grid_resolution

        # Noise filter. Remove the flows on voxels that are associated with the agent but is not marked as dynamic.
        temp_flow *= dynamic_category_mask[:, :, :, np.newaxis]

        # overlay the object flow on the scene flow
        dynamic_flows += temp_flow

    return dynamic_flows

def ComputeFlowsForBackground(
    occ: np.ndarray,
    ego_transformation_to_next: np.ndarray,
    labels_of_interest: list[int],
    grid_resolution: float = 0.4
) -> np.ndarray:
    """
    Computes voxel-level flow for background (static) voxels (e.g., road, terrain).

    Parameters:
      occ: (L, W, H) semantic occupancy grid.
      ego_transformation_to_next: 4×4 SE(3) ego update transform.
      labels_of_interest: static labels to flow (e.g., 7 for road).

    Returns:
      flows: (L, W, H, 3) float32 flow field in num of voxels.
    """
    flows = np.zeros((*occ.shape, 3), dtype=np.float32)
    mask = np.isin(occ, labels_of_interest)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return flows

    ego_xyz = OccFrameToEgoFrame(coords, grid_resolution)
    homog = np.concatenate([ego_xyz, np.ones((len(coords), 1))], axis=1)

    transformed = (ego_transformation_to_next @ homog.T).T[:, :3]
    delta = (transformed - ego_xyz) / grid_resolution

    flows[mask, :] = delta

    return flows

def ComputeFlowsForOccupancyGrid(
    occ: np.ndarray,
    curr_annotations: dict,
    next_annotations: dict,
    ego_transformation_to_next: np.ndarray,
    grid_resolution: float = 0.4
) -> np.ndarray:
    """
    Computes full occupancy flow by combining object + background flows.

    Parameters:
      occ: occupancy grid.
      curr_annotations, next_annotations: object dicts for consecutive frames.
      ego_transformation_to_next: SE(3) ego transform.
      labels_of_interest: dynamic object labels.

    Returns:
      flows: (L, W, H, 3) float32 flow field in num of voxels.
    """
    object_flow = ComputeFlowsForObjects(
        occ, curr_annotations, next_annotations,
        ego_transformation_to_next, [1,2,3,4], grid_resolution
    )
    background_flow = ComputeFlowsForBackground(
        occ, ego_transformation_to_next, [0,5,6,7,8,9], grid_resolution
    )
    return object_flow + background_flow

if __name__ == "__main__":
    """
    Example usage: loads UniOcc-generated sample to test flow computation.
    """
    from uniocc_dataset import UniOcc
    # Adjust paths and scene index as needed.
    # In this example, we take t=0 as the current frame and t=1 as the next frame.
    dataset = UniOcc(data_root="datasets/nuScenes-via-Occ3D-2Hz-mini", obs_len=1, fut_len=1)
    sample = dataset[0]

    occ0 = sample['obs_occ_labels'][0]
    occ1 = sample['fut_occ_labels'][0]
    gt_forward_flow = sample['obs_flows_forward'][0]
    gt_backward_flow = sample['fut_flows_backward'][0]  # GT for backward flow is only available in the t>0 frames.
    ann0 = {ann['token']: {'agent_to_ego': ann['agent_to_ego'], 'size': ann['size']}
            for ann in sample['annotations'][0]}
    ann1 = {ann['token']: {'agent_to_ego': ann['agent_to_ego'], 'size': ann['size']}
            for ann in sample['annotations'][1]}
    ego_to_world_0 = sample['obs_ego_to_worlds'][0]
    ego_to_world_1 = sample['fut_ego_to_worlds'][0]
    ego_0_to_ego_1 = np.linalg.inv(ego_to_world_1) @ ego_to_world_0

    # Compute forward flow.
    forward_flow_grid = ComputeFlowsForOccupancyGrid(
        occ0, ann0, ann1, ego_0_to_ego_1, grid_resolution=0.4
    )
    assert np.allclose(forward_flow_grid, gt_forward_flow, atol=1e-3), "Computed forward flow does not match ground truth!"
    print("✅ Forward flow computed successfully!")

    # Compute backward flow.
    backward_flow_grid = ComputeFlowsForOccupancyGrid(
        occ1, ann1, ann0, np.linalg.inv(ego_0_to_ego_1), grid_resolution=0.4
    )
    assert np.allclose(backward_flow_grid, gt_backward_flow, atol=1e-3), "Computed backward flow does not match ground truth!"
    print("✅ Backward flow computed successfully!")

    # Optionally visualize result:
    from uniocc_viz import VisualizeOccFlow
    VisualizeOccFlow(occ0, forward_flow_grid, show_ego=True).run()

