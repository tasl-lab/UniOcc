# Copyright (c) 2025. All rights reserved.
# Licensed under the MIT License.
#
# This script demonstrates the Occupancy Space Tracking algorithms.

import os
import matplotlib.pyplot as plt
from uniocc_eval import *
from uniocc_viz import *
from uniocc_utils import *

# Use Roboto font for better visualization.
plt.rcParams['font.sans-serif'] = ['Roboto']
plt.rcParams['font.family'] = 'sans-serif'

#########################
#  MAIN DEMONSTRATION
#########################

if __name__ == "__main__":
    # Example main code with minimal demonstration usage
    DATA_ROOT = "datasets/NuScenes-via-Occ3D-2Hz-mini"
    SCENE_NAME = "scene-0061"
    NUM_FRAMES = len(os.listdir(f"{DATA_ROOT}/{SCENE_NAME}"))

    occs = [np.load(f"{DATA_ROOT}/{SCENE_NAME}/{i}.npz")['occ_label'] for i in range(NUM_FRAMES)]
    flows = [np.load(f"{DATA_ROOT}/{SCENE_NAME}/{i}.npz")['occ_flow_forward'] for i in range(NUM_FRAMES)]
    ego_to_worlds = [np.load(f"{DATA_ROOT}/{SCENE_NAME}/{i}.npz")['ego_to_world_transformation'] for i in range(NUM_FRAMES)]

    occs = np.array(occs)  # Shape (T, L, W, H)
    flows = np.array(flows)  # Shape (T, L, W, H, 3)
    ego_to_worlds = np.array(ego_to_worlds)  # Shape (T, 4, 4)

    ###########################################################################
    #### Step 1 (Localization): Given a series occupancy grids,
    #### we have to estimate the ego motion.
    ###########################################################################
    # Estimate ego motion from flows on static voxels
    ego_motion_transformations = EstimateEgoMotionFromFlows(occs, flows, [7, 8, 9])

    # Accumulate transformations to get the ego trajectory
    ego_cum_motion_transformations = AccumulateTransformations(ego_motion_transformations)

    # Adjust GT trajectories to originate from the first frame
    ego_initial_transform = ego_to_worlds[0]
    ego_initial_transform_inv = np.linalg.inv(ego_initial_transform)
    ego_to_worlds_normalized = np.array([ego_initial_transform_inv @ t for t in ego_to_worlds])
    gt_traj = ego_to_worlds_normalized[:, :2, 3]  # Extracting x, y coordinates from the transformation matrices

    # Adjust estimated trajectories to the same origin
    est_traj = ego_cum_motion_transformations[:, :2, 3]  # Extracting x, y coordinates from the estimated transformations

    # [DEBUG] Plot the ground truth and estimated trajectories.
    plt.figure(figsize=(10, 6))
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'r-', label='Ground Truth Trajectory')
    plt.plot(est_traj[:, 0], est_traj[:, 1], 'b--', label='Estimated Trajectory')
    plt.title("Ground Truth vs Estimated Trajectories")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()
    exit(0)  # Exit after plotting the trajectories for debugging.

    ###########################################################################
    #### Step 2 (Tracking): In this example, we track the cars in the occupancy grids.
    ###########################################################################
    # Extract the cars reprenting cars.
    car_occs = occs == 1  # '1' is the label for cars

    # Track objects in the occupancy grids
    object_trajectories, tracked_objects = TrackOccObjects(car_occs, flows, ego_cum_motion_transformations)

    # [DEBUG] Plot the tracked objects
    # for obj_id, traj in object_trajectories.items():
    #     traj = np.array(traj.values())
    #     if len(traj) < 10:
    #         continue
    #
    #     plt.plot(*zip(*traj), marker='o', label=f'Object {obj_id}')
    #
    # plt.title("Tracked Object Trajectories")
    # plt.xlabel("X Coordinate")
    # plt.ylabel("Y Coordinate")
    # plt.legend()
    # plt.show()
    # pass

    # Show the occupancies as consecutive frames with tracked objects.
    # Cars being tracked will be shown in the same color across frames.
    for t in range(occs.shape[0]):
        print(f"Visualizing occupancy frame {t} with tracked objects...")
        original_occ = occs[t]

        # Set non car voxels with lighter color.
        original_occ[original_occ == 2] = 14
        original_occ[original_occ == 3] = 14
        original_occ[original_occ == 4] = 14
        original_occ[original_occ == 5] = 17
        original_occ[original_occ == 6] = 18
        original_occ[original_occ == 7] = 12
        original_occ[original_occ == 8] = 18
        original_occ[original_occ == 9] = 12

        for obj_id, obj_voxels in tracked_objects.items():
            if t in obj_voxels:
                obj_mask = np.zeros_like(original_occ)
                obj_mask[tuple(obj_voxels[t].T)] = 1
                original_occ[obj_mask == 1] = obj_id % 9  # Set color.
                obj_coords_ego_curr = OccFrameToEgoFrame(obj_voxels[t]).mean(axis=0)
                print(f"Object {obj_id} at {obj_coords_ego_curr}.")

        viz = VisualizeOcc(original_occ, show_ego=True)
        RotateO3DCamera(viz, "figures/O3D_RearUpCamera.json")

        # Usage 1: Interact with mouse, close the Open3D window to continue to next frame.
        # viz.run()
        # del viz

        # Usage 2: Save as images.
        viz.poll_events()
        viz.update_renderer()
        viz.capture_screen_image(f'output/occ_frame_{t:04d}.png')
        viz.destroy_window()
        del viz