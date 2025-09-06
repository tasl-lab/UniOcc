# Copyright (c) 2025. All rights reserved.
# Licensed under the MIT License.
#
# This script provides the UniOcc utility functions for voxel manipulation,
# occupancy grid operations, and 3D IoU calculations.

import os
import pickle
from typing import List

import numpy as np
import shapely
from matplotlib import pyplot as plt

from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull


######################################
# VOXEL UTILITY AND CONVERSION METHODS
######################################

def GetVoxelCoordinates(
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    grid_resolution: float,
    transform_agent_to_ego: np.ndarray,
    size: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Calculate the voxel coordinates (occ space) occupied by a bounding box in a 3D grid.

    Parameters:
    - grid_min: np.array of shape (3,), the minimum bound of the grid in world space.
    - grid_max: np.array of shape (3,), the maximum bound of the grid in world space.
    - grid_resolution: Float, size of each voxel in real-world units.
    - transform_agent_to_ego: np.array of shape (4, 4) transformation of the bounding box center.
    - size: np.array of shape (3,), dimensions of the bounding box along its local axes. l, w, h

    Returns:
    - voxel_coordinates: [n, 3] np.array of voxel indices occupied by the bounding box
    """
    # Step 1: Define the 8 corners of the bounding box in its local frame
    dx, dy, dz = size / 2.0
    corners = np.array([
        [-dx, -dy, -dz], [-dx, -dy, dz], [-dx, dy, -dz], [-dx, dy, dz],
        [dx, -dy, -dz], [dx, -dy, dz], [dx, dy, -dz], [dx, dy, dz]
    ])

    # Step 2: Transform corners to ego frame
    corners_homo = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=-1)
    transformed_corners = (transform_agent_to_ego @ corners_homo.T).T
    transformed_corners = transformed_corners[:, :-1]
    occ_frame_corners = transformed_corners - grid_min

    # Step 3: Convert world space coordinates to voxel grid indices
    voxel_corners = np.floor(occ_frame_corners / grid_resolution).astype(int)

    # Step 4: Clip voxel indices to ensure they lie within the grid
    grid_shape = np.ceil((grid_max - grid_min) / grid_resolution).astype(int)
    min_corner = np.clip(np.min(voxel_corners, axis=0), 0, grid_shape - 1)
    max_corner = np.clip(np.max(voxel_corners, axis=0), 0, grid_shape - 1)

    # Step 5: Collect all voxel indices within the bounding box
    x_range = np.arange(min_corner[0], max_corner[0] + 1)
    y_range = np.arange(min_corner[1], max_corner[1] + 1)
    z_range = np.arange(min_corner[2], max_corner[2] + 1)

    h = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    h = np.array(h).T.reshape(-1, 3)

    # Step 6: Find the convex hull formed by the corners
    try:
        hull = ConvexHull(voxel_corners)
    except:
        return [], occ_frame_corners

    # Step 7: Find the voxels that lie within the hull
    interior_points = []
    for point in h:
        point_center = point + grid_resolution / 2.0
        if all([np.dot(eq[:-1], point_center) <= -eq[-1] for eq in hull.equations]):
            interior_points.append(np.array(point))

    return interior_points, occ_frame_corners

def VoxelToCorners(voxel_coords, resolution):
    """
    Convert voxel indices to 3D bounding-box corner coordinates.

    Parameters
    ----------
    voxel_coords : np.ndarray, shape (N, 3)
        Voxel indices within the occupancy grid.
    resolution : float
        Size (in meters) of each voxel.

    Returns
    -------
    corners : np.ndarray, shape (N, 8, 3)
        The 3D coordinates for the 8 corners of each voxel.
    """
    corners = np.zeros((voxel_coords.shape[0], 8, 3))
    for i, voxel in enumerate(voxel_coords):
        x, y, z = voxel
        base_xyz = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                             [0,0,1],[1,0,1],[0,1,1],[1,1,1]])
        corners[i] = (base_xyz + [x,y,z]) * resolution

    return corners

def OccFrameToEgoFrame(voxel_coords, grid_resolution=0.4,
                       center_ego=np.array([40, 40, 2.2])):
    """
    Convert voxel coordinates from occupancy space to ego frame.

    Parameters:
        - voxel_coordinates: [n, 3] np.array of voxel indices in occupancy space.
        - grid_resolution: Float, size of each voxel in real-world units.
        - center_ego: np.array of shape (3,), the center of the ego frame.
    """
    ego_coords = np.zeros_like(voxel_coords, dtype=np.float32)

    ego_coords[:, 0] = voxel_coords[:, 0] * grid_resolution - center_ego[0]
    ego_coords[:, 1] = voxel_coords[:, 1] * grid_resolution - center_ego[1]
    ego_coords[:, 2] = voxel_coords[:, 2] * grid_resolution - center_ego[2]

    return ego_coords

def EgoFrameToOccFrame(ego_coords, grid_resolution=0.4,
                       center_ego=np.array([40, 40, 2.2])):
     """
     Convert voxel coordinates from ego frame to occupancy space.

     Parameters:
          - voxel_coordinates: [n, 3] np.array of voxel indices in ego frame.
          - grid_resolution: Float, size of each voxel in real-world units.
          - center_ego: np.array of shape (3,), the center of the ego frame.
     """
     occ_coords = np.zeros_like(ego_coords, dtype=np.float32)

     occ_coords[:, 0] = (ego_coords[:, 0] + center_ego[0]) / grid_resolution
     occ_coords[:, 1] = (ego_coords[:, 1] + center_ego[1]) / grid_resolution
     occ_coords[:, 2] = (ego_coords[:, 2] + center_ego[2]) / grid_resolution

     return np.round(occ_coords).astype(int)

def AlignToCentroid(voxels):
    """
    Translate voxels so that their centroid is at the origin (0,0,0).
    """
    return voxels - np.mean(voxels, axis=0)

def RasterizeCoordsToGrid(voxels, shape):
    """
    Given voxel coordinates and a desired 3D shape,
    rasterize them into a binary occupancy grid.
    """
    grid = np.zeros(shape, dtype=np.uint8)
    vox_int = voxels.astype(np.int32)
    mask = (
        (vox_int[:,0] >= 0) & (vox_int[:,0] < shape[0]) &
        (vox_int[:,1] >= 0) & (vox_int[:,1] < shape[1]) &
        (vox_int[:,2] >= 0) & (vox_int[:,2] < shape[2])
    )
    valid_vox = vox_int[mask]
    grid[valid_vox[:,0], valid_vox[:,1], valid_vox[:,2]] = 1
    return grid


###################################################
#  3D IOU
###################################################
def Compute3DBBoxIoU(points1, points2):
    """
    Approximate 3D IoU based on 2D bounding rectangles
    plus vertical extents.

    Parameters
    ----------
    points1, points2 : np.ndarray, shape (N, 3)
        3D coordinates for each object's voxels.

    Returns
    -------
    float
        Approximate IoU between the 3D bounding volumes.
    """
    # Flatten to x-y
    xy1 = points1[:, :2]
    xy2 = points2[:, :2]
    h1  = points1[:, 2].max() - points1[:, 2].min()
    h2  = points2[:, 2].max() - points2[:, 2].min()

    mp1 = shapely.geometry.MultiPoint(xy1)
    mp2 = shapely.geometry.MultiPoint(xy2)

    rect1 = mp1.minimum_rotated_rectangle
    rect2 = mp2.minimum_rotated_rectangle

    intersect_area = rect1.intersection(rect2).area
    union_area     = rect1.union(rect2).area
    if union_area < 1e-5:
        return 0.0

    # Approximate volumes
    intersection_vol = intersect_area * min(h1, h2)
    union_vol        = union_area * max(h1, h2)
    return intersection_vol / union_vol


#################################
#  PCA-BASED VOXEL ALIGNMENT
#################################
def AlignWithPCA(voxels, reference_pca=None):
    """
    Align voxel distribution to principal axes. If 'reference_pca'
    is provided, ensure consistent orientation across frames.
    """
    pca = PCA(n_components=3)
    aligned = pca.fit_transform(voxels)

    if reference_pca is not None:
        # Attempt to keep consistent axis orientation
        sign = np.sign(np.diag(np.dot(pca.components_, reference_pca.components_.T)))
        aligned *= sign
    return aligned, pca

def ComputeGridIoU(a, b):
    """
    Binary IoU for two same-shaped occupancy grids.
    """
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    return intersection / union if union > 0 else 1.0


#####################################
#  SIMPLE CCL AND VOXEL SEGMENTATION
#####################################
def SegmentVoxels(frame, structure=np.ones((3, 3, 3)), min_num_voxels=32):
    """
    3D connected-component labeling with min/max voxel filtering.

    Parameters
    ----------
    frame : np.ndarray, shape (L, W, H)
        Occupancy grid to segment.

    structure : np.ndarray
        3D structuring element for connectivity.

    min_num_voxels : int
        Minimum number of voxels for a valid component.

    Returns
    -------
    labeled : np.ndarray, shape (L, W, H)
        A grid of the same shape where each connected component
        has a unique integer label.
    filtered_count : int
        Count of valid connected components after filtering.
    """
    labeled, num_objects = label(frame, structure=structure)
    after_filtered_count = 0
    new_label = 1
    for obj_id in range(1, num_objects+1):
        size = np.sum(labeled == obj_id)
        if size < min_num_voxels:
            labeled[labeled == obj_id] = 0
            continue

        labeled[labeled == obj_id] = new_label
        after_filtered_count += 1
        new_label += 1

    return labeled, after_filtered_count


#####################################
#  Ego Motion Estimation from Flows
#####################################
def __invert_rigid_transform__(R_points, t_points):
    """
    Invert a rigid transformation that transforms points (i.e., find camera motion).

    Args:
        R_points: (3, 3) rotation matrix that moves points from t to t+1
        t_points: (3,) translation vector that moves points from t to t+1

    Returns:
        R_cam: (3, 3) rotation matrix of the camera motion
        t_cam: (3,) translation vector of the camera motion
    """
    R_cam = R_points.T
    t_cam = -R_cam @ t_points
    return R_cam, t_cam


def __compute_point_motion__(points_src, motion_vectors):
    """
    Compute point (not camera) rotation R and translation t from 3D motion vectors.
    This function assumes the motions are low noise and apply Kabsch Algorithm.

    Args:
        points_src: (N, 3) array of 3D points at time t
        motion_vectors: (N, 3) array of motion vectors (p_dst - p_src)

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    assert points_src.shape == motion_vectors.shape
    N = points_src.shape[0]

    if N == 1:
        # Underdetermined: assume no rotation
        R_est = np.eye(3)
        t_est = motion_vectors[0]

        return R_est, t_est

    points_dst = points_src + motion_vectors

    centroid_src = np.mean(points_src, axis=0)
    centroid_dst = np.mean(points_dst, axis=0)

    src_centered = points_src - centroid_src
    dst_centered = points_dst - centroid_dst

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R_est = Vt.T @ U.T

    # Ensure proper rotation (det(R) = +1)
    if np.linalg.det(R_est) < 0:
        Vt[2, :] *= -1
        R_est = Vt.T @ U.T

    t_est = centroid_dst - R_est @ centroid_src

    return R_est, t_est

def __ransac_point_motion__(points_src, motion_vectors, threshold=0.01, max_iter=100):
    """
    Robust estimation of rigid transform from motion vectors using RANSAC.
    This function assumes that the motion vectors have noise.

    Args:
        points_src: (N, 3) array of 3D source points
        motion_vectors: (N, 3) array of 3D displacements
        threshold: inlier distance threshold
        max_iter: number of RANSAC iterations

    Returns:
        best_R: estimated rotation matrix
        best_t: estimated translation vector
    """
    assert points_src.shape == motion_vectors.shape
    N = points_src.shape[0]
    if N < 3:
        return __compute_point_motion__(points_src, motion_vectors)

    best_inliers = []
    best_R, best_t = None, None

    for _ in range(max_iter):
        idx = np.random.choice(N, 3, replace=False)
        R_trial, t_trial = __compute_point_motion__(
            points_src[idx], motion_vectors[idx]
        )

        # Apply to all source points
        transformed = (R_trial @ points_src.T).T + t_trial
        points_dst = points_src + motion_vectors
        errors = np.linalg.norm(transformed - points_dst, axis=1)
        inliers = np.where(errors < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R = R_trial
            best_t = t_trial

    if best_R is None:
        return __compute_point_motion__(points_src, motion_vectors)

    return __compute_point_motion__(
        points_src[best_inliers], motion_vectors[best_inliers]
    )


def EstimateEgoMotionFromFlows(occs,
                               flows,
                               static_object_labels : List[int],
                               grid_resolution=0.4):
    """
    Estimate ego motion from flow fields across occupancy grids.

    Parameters
    ----------
    occs : np.ndarray, shape (T, L, W, H)
        Occupancy grids across T timesteps.
    flows : np.ndarray, shape (T, L, W, H, 3)
        Flow fields at each frame.
    static_object_labels : List[int]
        List of labels for static objects.
    grid_resolution : float
        Size of each voxel in meters.

    Returns
    -------
    ego_motion : np.ndarray, shape (T, 4, 4)
        Estimated ego motion matrices for each timestep.
    """
    if static_object_labels is None:
        static_object_labels = [7, 8, 9] # UniOcc default labels for road, building, and terrain

    T = occs.shape[0]
    ego_motion = []

    for t in range(T):
        occ = occs[t]
        flow = flows[t]

        static_object_mask = np.isin(occ, static_object_labels)

        # If no static objects are present, skip this frame.
        if not np.any(static_object_mask):
            ego_motion.append(np.eye(4))

        # Get the flows on the static voxels.
        static_object_flows = flow[static_object_mask]

        # Get the coords of the static voxels.
        static_object_voxels = np.argwhere(static_object_mask)
        static_object_coords_meters = OccFrameToEgoFrame(static_object_voxels, grid_resolution=grid_resolution)

        # Convert flows to meters.
        static_object_flows_meters = static_object_flows * grid_resolution

        # Estimate ego translation and rotation using the flows. <-- This can certainly be done in a better way.
        R_est, t_est = __ransac_point_motion__(
            static_object_coords_meters, static_object_flows_meters
        )

        # Invert the rigid transformation to get the ego motion.
        R_est, t_est = __invert_rigid_transform__(R_est, t_est)

        # Create the 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R_est
        transform_matrix[:3, 3] = t_est
        ego_motion.append(transform_matrix)

    return np.array(ego_motion)

def AccumulateTransformations(transformations):
    """
    Accumulate a list of 4x4 transformation matrices into a cumsum form of transformations.

    Parameters
    ----------
    transformations : List[np.ndarray]
        List of 4x4 transformation matrices.

    Returns
    -------
    np.ndarray
        The accumulated transformation matrices.
    """
    T = len(transformations)
    cum_transformations = [np.eye(4)]

    for t in range(0, T - 1):
        prev_transformation = cum_transformations[len(cum_transformations) - 1]
        curr_transformation = prev_transformation @ transformations[t]

        cum_transformations.append(curr_transformation)

    return np.array(cum_transformations)


######################################
# Occupancy Space Bipartite Mathcing & Tracking
######################################
def BipartiteMatch(score_matrix):
    """
    Maximize total score by turning it into
    a min-cost assignment problem.

    Parameters
    ----------
    score_matrix : np.ndarray of shape (A, B)
        Score between each annotation and each prediction.

    Returns
    -------
    matched_pairs : list of (int, int)
        List of (ann_idx, pred_idx) matches.
    """
    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_pairs = list(zip(row_ind, col_ind))
    return matched_pairs

def TrackOccObjects(binary_occs,
                          flows,
                          ego_cum_motion_transformations,
                          threshold=2.0):
    """
    Track objects in binary occupancy grids using flow fields.

    Parameters
    ----------
    binary_occs : np.ndarray, shape (T, L, W, H)
        Binary occupancy grids across T timesteps.
    flows : np.ndarray, shape (T, L, W, H, 3)
        Flow fields at each frame.
    ego_cum_motion_transformations : np.ndarray, shape (T, 4, 4)
        Cumulative ego motion transformations since the first frame.
        It can come from ego_to_world, or from EstimateEgoMotionFromFlows (in which
        the world origin is ego's t=0s position).
    threshold : float
        Distance threshold in meters for centroid-based association. This
        value can be eye-balled from the small values in the cost_matrix below.

    Returns
    -------
    trajectories : dict
        {object_id: {timestep: centroid positions in global coordinates}}
    tracked_objects : dict
        {object_id: {timestep: object_voxels}}
    """

    T = binary_occs.shape[0]
    if T < 2:
        return {}, {}, {}

    # Accumulate per-frame object IDs
    all_id_grids = []
    object_trajectories = {}
    current_id = 1

    prev_centroids = []
    pred_centroids = []
    prev_ids = []

    # MAIN TRACKING LOOP
    for t in range(T):
        binary_occ = binary_occs[t]
        flow = flows[t]

        # Segment objects
        labeled, num_obj = SegmentVoxels(binary_occ)

        curr_centroids = []
        next_centroids = []

        for obj_idx in range(1, labeled.max()+1):
            obj_mask = (labeled == obj_idx)

            # Current centroid in (L, W, H) -> real coords
            curr_obj_voxels = np.argwhere(obj_mask)
            curr_obj_coords_ego_curr = OccFrameToEgoFrame(curr_obj_voxels).mean(axis=0)

            # Predict next centroid with flow
            obj_flow = flow[obj_mask]
            pred_obj_voxels = curr_obj_voxels + obj_flow
            pred_obj_coords_ego_next = OccFrameToEgoFrame(pred_obj_voxels).mean(axis=0)

            # Transform to global
            curr_obj_coords_ego_curr = np.array([*curr_obj_coords_ego_curr, 1.0])
            curr_obj_coords_world = ego_cum_motion_transformations[t].dot(curr_obj_coords_ego_curr)[:3]

            # Predict next centroid in world coordinates by compensating for ego motion.
            pred_obj_coords_ego_next = np.array([*pred_obj_coords_ego_next, 1.0])
            pred_obj_coords_world = ego_cum_motion_transformations[t+1].dot(pred_obj_coords_ego_next)[:3] if t < T - 1 else pred_obj_coords_ego_next[:3]

            # Save the results.
            curr_centroids.append(curr_obj_coords_world)
            next_centroids.append(pred_obj_coords_world)

        if len(pred_centroids) == 0:
            print(f"No previous predictions, initializing at frame {t}")

            # Assign new IDs
            these_ids = list(range(current_id, current_id + len(curr_centroids)))
            current_id += len(curr_centroids)
            # Initialize trajectories
            for idx, cid in enumerate(these_ids):
                object_trajectories[cid]= {t: curr_centroids[idx]}
            prev_centroids = curr_centroids.copy()
            pred_centroids = next_centroids.copy()
            prev_ids = these_ids

            id_grid = labeled.copy()
            all_id_grids.append(id_grid)
        else:
            # [DEBUG]
            # prev_dots = np.array(prev_centroids)
            # curr_dots = np.array(curr_centroids)
            # pred_dots = np.array(pred_centroids)
            # next_dots = np.array(next_centroids)
            # plt.plot(prev_dots[:, 0], prev_dots[:, 1], 'ro', label='Previous')
            # plt.plot(curr_dots[:, 0], curr_dots[:, 1], 'bo', label='Current')
            # plt.plot(pred_dots[:, 0], pred_dots[:, 1], 'y*', label='Predicted')
            # plt.plot(next_dots[:, 0], next_dots[:, 1], 'g*', label='Next Predicted')
            # plt.title(f"Frame {t}")
            # plt.xlabel("X")
            # plt.ylabel("Y")
            # plt.legend()
            # plt.show()

            # Edge case: if no objects detected in the current frame.
            if len(curr_centroids) == 0:
                all_id_grids.append(np.zeros_like(labeled))
                prev_ids = []
                continue

            # Hungarian-based association
            cost_matrix = cdist(np.array(pred_centroids), np.array(curr_centroids), 'euclidean')
            if np.any(np.isnan(cost_matrix)):
                print(f"NaN detected in cost matrix at frame {t}. Skipping frame.")
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Assign IDs
            matched_ids = {}  # Current idx->ID
            matched_t = {}  # Current idx->ID
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < threshold:
                    matched_t[c] = prev_ids[r]
                    matched_ids[c] = prev_ids[r]
                    object_trajectories[prev_ids[r]][t] = curr_centroids[c]

            # Assign new IDs to unmatched
            for i in range(len(curr_centroids)):
                if i not in matched_ids:
                    matched_t[i] = current_id
                    object_trajectories[current_id] = {t: curr_centroids[i]}
                    current_id += 1

            # Build ID grid
            id_grid = np.zeros_like(labeled)
            label_val = 1
            for obj_idx in range(1, labeled.max()+1):
                idx_mask = (labeled == obj_idx)
                if np.sum(idx_mask) < 1:
                    continue
                # Map BFS label to new ID
                if label_val - 1 not in matched_t:
                    print(f"Object {label_val - 1} not matched to ID {matched_t[label_val - 1]}")
                id_grid[idx_mask] = matched_t[label_val - 1]
                label_val += 1
            all_id_grids.append(id_grid)

            # Update for next iteration
            prev_centroids = curr_centroids.copy()
            pred_centroids = next_centroids.copy()
            prev_ids = matched_t.copy()

    # Get the voxels for the tracked objects.
    tracked_objects = {}
    for t, grid_ids in enumerate(all_id_grids):
        unique_ids = np.unique(grid_ids)
        for oid in unique_ids:
            if oid == 0:
                continue
            object_vox = np.argwhere(grid_ids == oid)
            if oid not in tracked_objects:
                tracked_objects[oid] = {}
            tracked_objects[oid][t] = object_vox

    return object_trajectories, tracked_objects


#######################################
# Testing Code.
#######################################
if __name__ == "__main__":
    # Test the RANSAC rigid transform estimation
    points = np.random.rand(20, 3)
    R_gt = R.from_euler('xyz', [10, 5, 2], degrees=True).as_matrix()
    t_gt = np.array([0.1, -0.2, 0.3])
    points_dst = (R_gt @ points.T).T + t_gt
    motion_vectors = points_dst - points

    # Estimate
    R_est, t_est = __compute_point_motion__(points, motion_vectors)
    print("R_est:\n", R_est)
    print("t_est:\n", t_est)