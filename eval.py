# Copyright (c) 2025. All rights reserved.
# Licensed under the MIT License.
#
# This script provides a reference implementation for occupancy forecasting
# without disclosing any proprietary details. It includes dataset utilities,
# GMM-based bounding-box analysis, and voxel-based evaluation methods.

import os
import pickle
import numpy as np
import shapely

from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from shapely.geometry import Polygon, Point
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from viz import VisualizeOcc


######################################
# VOXEL UTILITY AND CONVERSION METHODS
######################################

def __voxel_to_corners__(voxel_coords, resolution):
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

def __occ_frame_to_ego_frame__(voxel_coords, grid_resolution=0.4,
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

def __ego_frame_to_occ_frame__(ego_coords, grid_resolution=0.4,
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

###################################################
#  3D IOU / MATCHING UTILITIES & BIPARTITE MAPPING
###################################################

def __3d_bbox_iou__(points1, points2):
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

def __bipartite_match__(score_matrix):
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

#################################
#  PCA-BASED VOXEL ALIGNMENT
#################################

def __align_to_centroid__(voxels):
    """
    Translate voxels so that their centroid is at the origin (0,0,0).
    """
    return voxels - np.mean(voxels, axis=0)

def __align_with_pca__(voxels, reference_pca=None):
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

def __grid_iou__(a, b):
    """
    Binary IoU for two same-shaped occupancy grids.
    """
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    return intersection / union if union > 0 else 1.0

def __convert_voxels_to_grid__(voxels, shape):
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

#####################################
#  SIMPLE CCL AND VOXEL SEGMENTATION
#####################################
def __segment_voxels__(frame, structure=np.ones((3,3,3))):
    """
    3D connected-component labeling with min/max voxel filtering.

    Parameters
    ----------
    frame : np.ndarray, shape (L, W, H)
        Occupancy grid to segment.

    structure : np.ndarray
        3D structuring element for connectivity.

    Returns
    -------
    labeled : np.ndarray, shape (L, W, H)
        A grid of the same shape where each connected component
        has a unique integer label.
    filtered_count : int
        Count of valid connected components after filtering.
    """
    labeled, num_objects = label(frame, structure=structure)
    filtered_count = 0
    new_label = 1
    for obj_id in range(1, num_objects+1):
        size = np.sum(labeled == obj_id)

        labeled[labeled == obj_id] = new_label
        filtered_count += 1
        new_label += 1

    return labeled, filtered_count

def __geometric_centroid_occ__(voxels, resolution=0.4):
    """
    Compute the centroid (in real coordinates) for a set of voxel indices.
    """
    return np.mean(voxels, axis=0) * resolution
#######################################
# OBJECT LIKELIHOOD / GMM-BASED EVAL
#######################################
def FindGMMForCategory(category_id: int,
                       data_root: str,):
    with open(os.path.join(f'{data_root}/scene_infos.pkl'), 'rb') as f:
        scene_infos = pickle.load(f)

    # Gather bbox
    bboxes = []
    for scene_info in scene_infos:
        scene_files = scene_info['occ_in_scene_paths']

        for timestep in range(len(scene_files)):
            scene_file = scene_files[timestep]
            gt_data = np.load(os.path.join(data_root, scene_file), allow_pickle=True)

            for annotation in gt_data['annotations']:
                if annotation['category_id'] == category_id:
                    w, l, h = annotation['size']
                    bboxes.append(np.array((l, w, h)))

    bboxes = np.array(bboxes)

    # Find GMM
    def gmm_bic_score(estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)

    param_grid = {
        "n_components": range(1, 10),
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score, verbose=1
    )
    grid_search.fit(bboxes)

    n_components = grid_search.best_params_['n_components']
    covariance_type = grid_search.best_params_['covariance_type']

    print(f'Fitting GMM for category {category_id} with {len(bboxes)} samples')
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42
    )
    gmm.fit(bboxes)
    return gmm

def ComputeObjectLikelihoods(binary_occ: np.ndarray,
                             gmm_model: GaussianMixture,
                             resolution: float = 0.4):
    """
    Compute plausibility scores for segmented objects using a
    Gaussian Mixture Model (GMM) trained on bounding-box dimensions.

    Parameters
    ----------
    binary_occ : np.ndarray, shape (L, W, H)
        A 3D occupancy grid (monolabel).
    gmm_model : GaussianMixture
        A pretrained GMM for object dimension distribution.
    category_id : int
        Category identifier (e.g., 'Car', 'Pedestrian', etc.).
    resolution : float, optional
        The real-world size of each voxel in meters, by default 0.4.

    Returns
    -------
    probabilities : list of float
        The maximum GMM probability for each segmented object.
    likeli_count : int
        Count of objects that exceed a probability threshold of 0.5.
    total_count : int
        Total number of segmented objects in this grid (unfiltered).
    """
    # Example logic for demonstration: user-defined min/max voxel range
    labeled_grid, _ = __segment_voxels__(binary_occ)
    max_label = labeled_grid.max()

    probabilities = []
    likeli_count = 0
    total_count = 0

    for label_id in range(1, max_label + 1):
        mask = (labeled_grid == label_id)
        if not np.any(mask):
            continue
        points_3d = np.argwhere(mask) * resolution
        if points_3d.shape[0] < 3:
            continue
        total_count += 1

        # 2D bounding rectangle on x-y plane
        points_2d = points_3d[:, :2]
        multi_pts = shapely.geometry.MultiPoint(points_2d)
        rect_poly = multi_pts.minimum_rotated_rectangle

        if not isinstance(rect_poly, shapely.geometry.Polygon):
            continue

        x_coords, y_coords = rect_poly.exterior.coords.xy
        edges = [
            Point(x_coords[i], y_coords[i]).distance(
                Point(x_coords[i+1], y_coords[i+1])
            )
            for i in range(len(x_coords) - 1)
        ]

        length = max(edges)
        width  = min(edges)
        height = points_3d[:,2].max() - points_3d[:,2].min()

        dims = np.array([[length, width, height]])
        prob_array = gmm_model.predict_proba(dims)
        max_prob   = np.max(prob_array)
        probabilities.append(max_prob)

        if max_prob > 0.5:
            likeli_count += 1

    return probabilities, likeli_count, total_count

#################################
#  TEMPORAL SHAPE CONSISTENCY
#################################

def ComputeTemporalShapeConsistency(binary_occs,
                                    flows,
                                    ego_to_worlds,
                                    threshold=0.5):
    """
    Example method to measure temporal shape consistency by aligning
    predicted voxel sets across frames.

    Parameters
    ----------
    binary_occs : np.ndarray, shape (T, L, W, H)
        Binary occupancy grids across T timesteps.
    flows : np.ndarray, shape (T, L, W, H, 3)
        Flow fields at each frame.
    ego_to_worlds : np.ndarray, shape (T, 4, 4)
        Ego pose transformations for each frame.
    threshold : float
        Distance threshold for centroid-based association.

    Returns
    -------
    iou_scores : dict
        {object_id: mean iou across frames}
    trajectories : dict
        {object_id: list of centroid positions in global coordinates}
    tracked_objects : dict
        {object_id: list of voxel sets across frames}
    """
    # Full detail provided, implementing a simple shape-check pipeline

    T = binary_occs.shape[0]
    if T < 2:
        return {}, {}, {}

    # Accumulate per-frame object IDs
    all_id_grids = []
    object_trajectories = {}
    current_id = 1

    prev_centroids = []
    prev_ids = []

    # MAIN TRACKING LOOP
    for t in range(T):
        frame = binary_occs[t]
        if t < T-1:
            frame_flow = flows[t]
        else:
            frame_flow = np.zeros_like(frame)[..., None].repeat(3, axis=-1)

        # Segment objects
        labeled, num_obj = __segment_voxels__(frame)

        centroids_t = []
        predicted_tplus1 = []

        for obj_idx in range(1, labeled.max()+1):
            obj_mask = (labeled == obj_idx)
            if np.sum(obj_mask) < 3:
                continue

            # Current centroid in (L, W, H) -> real coords
            voxels_curr = np.argwhere(obj_mask)
            cent_ego_curr = __geometric_centroid_occ__(__occ_frame_to_ego_frame__(voxels_curr))

            # Predict next centroid if we have flow
            flows = frame_flow[obj_mask]
            next_voxels = voxels_curr + flows
            cent_ego_next = __geometric_centroid_occ__(__occ_frame_to_ego_frame__(next_voxels))

            # Transform to global
            cent_ego_curr_4d = np.array([*cent_ego_curr, 1.0])
            cent_world_curr = ego_to_worlds[t].dot(cent_ego_curr_4d)[:3]

            cent_ego_next_4d = np.array([*cent_ego_next, 1.0])
            if t < T-1:
                cent_world_next = ego_to_worlds[t + 1].dot(cent_ego_next_4d)[:3]
            else:
                cent_world_next = cent_world_curr

            centroids_t.append(cent_world_curr)
            predicted_tplus1.append(cent_world_next)

        if t == 0:
            # Assign new IDs
            these_ids = list(range(current_id, current_id + len(centroids_t)))
            current_id += len(centroids_t)
            # Initialize trajectories
            for idx, cid in enumerate(these_ids):
                object_trajectories[cid] = [centroids_t[idx]]
            prev_centroids = predicted_tplus1
            prev_ids = these_ids

            id_grid = labeled.copy()
            all_id_grids.append(id_grid)
        else:
            prev_dots = np.array(prev_centroids)
            curr_dots = np.array(centroids_t)

            # plt.plot(prev_dots[:, 0], prev_dots[:, 1], 'ro', label='Previous')
            # plt.plot(curr_dots[:, 0], curr_dots[:, 1], 'bo', label='Current')
            # plt.title(f"Frame {t}")
            # plt.xlabel("X")
            # plt.ylabel("Y")
            # plt.legend()
            # plt.show()

            # Hungarian-based association
            if len(centroids_t) == 0:
                all_id_grids.append(np.zeros_like(labeled))
                prev_centroids = []
                prev_ids = []
                continue

            cost_matrix = cdist(np.array(prev_centroids), np.array(centroids_t), 'euclidean')
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Assign IDs
            matched_ids = {}  # Current->Prev
            matched_t = np.zeros(len(centroids_t), dtype=int)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < threshold:
                    matched_t[c] = prev_ids[r]
                    object_trajectories[prev_ids[r]].append(centroids_t[c])
                    matched_ids[c] = prev_ids[r]

            # Assign new IDs to unmatched
            for i in range(len(centroids_t)):
                if i not in matched_ids:
                    matched_t[i] = current_id
                    object_trajectories[current_id] = [centroids_t[i]]
                    current_id += 1

            # Build ID grid
            id_grid = np.zeros_like(labeled)
            label_val = 1
            for obj_idx in range(1, labeled.max()+1):
                idx_mask = (labeled == obj_idx)
                if np.sum(idx_mask) < 1:
                    continue
                # Map BFS label to new ID
                id_grid[idx_mask] = matched_t[label_val - 1]
                label_val += 1
            all_id_grids.append(id_grid)

            # Update for next iteration
            prev_centroids = predicted_tplus1
            prev_ids = matched_t.tolist()

    # SHAPE IOU
    tracked_objects = {}
    for t, grid_ids in enumerate(all_id_grids):
        unique_ids = np.unique(grid_ids)
        for oid in unique_ids:
            if oid == 0:
                continue
            object_vox = np.argwhere(grid_ids == oid)
            if oid not in tracked_objects:
                tracked_objects[oid] = []
            tracked_objects[oid].append(object_vox)


    iou_scores = {}
    for oid, vox_list in tracked_objects.items():
        if len(vox_list) < 2:
            continue

        aligned_shapes = []
        reference_pca = None

        # Align shapes with PCA
        for vox_block in vox_list:
            if len(vox_block) < 3:
                continue
            shifted = __align_to_centroid__(vox_block)
            rotated, pca = __align_with_pca__(shifted, reference_pca)
            if reference_pca is None:
                reference_pca = pca
            aligned_shapes.append(rotated)

        # IoU across frames
        ious = []
        for i in range(len(aligned_shapes) - 1):
            shapeA = aligned_shapes[i]
            shapeB = aligned_shapes[i+1]
            gridA = __convert_voxels_to_grid__(shapeA, shape=(binary_occs.shape[1], binary_occs.shape[2], binary_occs.shape[3]))
            gridB = __convert_voxels_to_grid__(shapeB, shape=(binary_occs.shape[1], binary_occs.shape[2], binary_occs.shape[3]))
            ious.append(__grid_iou__(gridA, gridB))
        if len(ious) > 0:
            iou_scores[oid] = np.mean(ious)

    return iou_scores, object_trajectories, tracked_objects

######################################
#  STATIC BACKGROUND CONSISTENCY
######################################

def ComputeStaticConsistency(binary_occs, ego_to_worlds):
    """
    Evaluate how consistently the static background
    is preserved across consecutive frames.

    Parameters
    ----------
    binary_occs : np.ndarray, shape (T, L, W, H)
        Monolabel occupancy grids (0/1) at T timesteps.
    ego_to_worlds : np.ndarray, shape (T, 4, 4)
        Global pose transforms for each frame.

    Returns
    -------
    list of float
        Consistency scores for each consecutive pair of frames.
    """
    T, L, W, H = binary_occs.shape

    consistencies = []
    for t in range(T - 1):
        curr = binary_occs[t]
        next = binary_occs[t + 1]

        curr_T = ego_to_worlds[t]
        next_T = ego_to_worlds[t + 1]
        curr_to_next_T = np.linalg.inv(ego_to_worlds[t + 1]) @ ego_to_worlds[t]

        # Step 1: Current frame meshgrid
        xx = np.arange(L)
        yy = np.arange(W)
        zz = np.arange(H)
        xx, yy, zz = np.meshgrid(xx, yy, zz, indexing='ij')
        mesh_coords = np.stack([xx, yy, zz], axis=-1)
        mesh_coords = mesh_coords.reshape(-1, 3)

        # Step 2: convert to ego frame
        mesh_coords_ego = __occ_frame_to_ego_frame__(mesh_coords)

        # Step 3: advance the meshgrid to the next frame
        mesh_coords_ego = np.concatenate([mesh_coords_ego, np.ones((len(mesh_coords_ego), 1))], axis=-1)
        mesh_coords_ego_next = (curr_to_next_T @ mesh_coords_ego.T).T

        # Step 4: Floor to int
        mesh_coords_ego_next = __ego_frame_to_occ_frame__(mesh_coords_ego_next)

        # Step 5: Filter out of bounds
        in_range_mesh_coords = mesh_coords_ego_next[
            (mesh_coords_ego_next[..., 0] >= 0) & (mesh_coords_ego_next[..., 0] < L) &
            (mesh_coords_ego_next[..., 1] >= 0) & (mesh_coords_ego_next[..., 1] < W) &
            (mesh_coords_ego_next[..., 2] >= 0) & (mesh_coords_ego_next[..., 2] < H)]

        # Step 6: find a FOV mask
        fov_mask = np.zeros_like(next)
        fov_mask[in_range_mesh_coords[:, 0], in_range_mesh_coords[:, 1], in_range_mesh_coords[:, 2]] = 1
        next_static = next == 1

        # Step 7: current frame voxels
        static_voxels_coords = np.argwhere(curr == 1)
        static_voxels_coords_ego = __occ_frame_to_ego_frame__(static_voxels_coords)

        # Step 8: advance the voxels to the next frame
        static_voxels_coords_ego = np.concatenate(
            [static_voxels_coords_ego, np.ones((len(static_voxels_coords_ego), 1))], axis=-1)
        static_voxels_coords_ego_next = (curr_to_next_T @ static_voxels_coords_ego.T).T

        # Step 9: Floor to int
        static_voxels_coords_ego_next = __ego_frame_to_occ_frame__(static_voxels_coords_ego_next)

        # Step 10: Filter out of bounds
        in_range_static_voxels_coords = static_voxels_coords_ego_next[
            (static_voxels_coords_ego_next[:, 0] >= 0) & (static_voxels_coords_ego_next[:, 0] < L) &
            (static_voxels_coords_ego_next[:, 1] >= 0) & (static_voxels_coords_ego_next[:, 1] < W) &
            (static_voxels_coords_ego_next[:, 2] >= 0) & (static_voxels_coords_ego_next[:, 2] < H)]

        # Step 11: rasterize the voxels
        scene_movement_mask = np.zeros_like(next)
        scene_movement_mask[
            in_range_static_voxels_coords[:, 0], in_range_static_voxels_coords[:, 1], in_range_static_voxels_coords[
                                                                                      :, 2]] = 1

        # Step 12: compute the intersection
        intersection = np.sum(fov_mask & next_static & scene_movement_mask)

        # Step 13: compute the union
        union = np.sum((fov_mask & next_static) | scene_movement_mask)

        # Step 14: compute the static consistency
        if union == 0:
            return [1.0]
        static_consistency = intersection / union
        consistencies.append(static_consistency)

    return consistencies

##############################################
#  SIMPLE BINARY IOU FOR MONOLABEL GRIDS
##############################################
def ComputeIoU(occ_gt, occ_pred, free_label=10):
    """
    Compute IoU for two occupancy grids.

    Parameters
    ----------
    occ_gt : np.ndarray     Ground truth occupancy.
    occ_pred : np.ndarray   Predicted occupancy.
    free_label: int         Label of the voxels that should not be counted in the IoU. (usually 10 for free space)

    Returns
    -------
    float
        Intersection-over-union for the single-class occupancy.
    """
    intersection = np.sum(np.logical_and(occ_gt != free_label, occ_pred != free_label))
    union        = np.sum(np.logical_or (occ_gt != free_label, occ_pred != free_label))
    return intersection/union if union != 0 else 1.0

def ComputeIoUForCategory(occ_gt, occ_pred, category_id):
    """
    Similar to above but just compute IoU for a specific category in the occupancy grid.

    Parameters
    ----------
    occ_gt : np.ndarray     Ground truth occupancy.
    occ_pred : np.ndarray   Predicted occupancy.
    category_id: int        The category ID to compute IoU for.

    Returns
    -------
    float
        Intersection-over-union for the specified category.
    """
    intersection = np.sum(np.logical_and(occ_gt == category_id, occ_pred == category_id))
    union        = np.sum(np.logical_or (occ_gt == category_id, occ_pred == category_id))
    return intersection/union if union != 0 else 1.0

#########################
#  MAIN DEMONSTRATION
#########################

if __name__ == "__main__":
    # Example main code with minimal demonstration usage
    DATA_ROOT = "datasets/NuScenes-via-Occ3D-2Hz-mini"
    DATA_PATH_0 = "datasets/NuScenes-via-Occ3D-2Hz-mini/scene-0061/0.npz"
    DATA_PATH_1 = "datasets/NuScenes-via-Occ3D-2Hz-mini/scene-0061/1.npz"

    # Load occ labels
    data0 = np.load(DATA_PATH_0)
    data1 = np.load(DATA_PATH_1)
    occ_label_0 = data0['occ_label']
    occ_label_1 = data1['occ_label']
    flow_0 = data0['occ_flow_forward']
    flow_1 = data1['occ_flow_forward']
    ego_to_world0 = data0['ego_to_world_transformation']
    ego_to_world1 = data1['ego_to_world_transformation']

    # Visualize the occupancy grids
    # VisualizeOcc(occ_label_0).run()
    # VisualizeOcc(occ_label_1).run()

    # Compute IoU
    print("IoU between two frames: ", ComputeIoU(occ_label_0, occ_label_1))
    print("IoU for Cars between two frames: ", ComputeIoUForCategory(occ_label_0, occ_label_1, category_id=1))

    # Temporal consistency for static background (roads, labeled as 7)
    road_occ_0 = data0['occ_label'] == 7
    road_occ_1 = data1['occ_label'] == 7
    road_occs = np.stack([road_occ_0, road_occ_1], axis=0)
    ego_to_worlds = np.stack([ego_to_world0, ego_to_world1], axis=0)
    static_consistency = ComputeStaticConsistency(road_occs, ego_to_worlds)
    print("Static consistency between two frames: ", static_consistency)

    # Temporal consistency for cars (labeled as 1)
    car_occ_0 = data0['occ_label'] == 1
    car_occ_1 = data1['occ_label'] == 1
    car_occs = np.stack([car_occ_0, car_occ_1], axis=0)
    flows = np.stack([flow_0, flow_1], axis=0)
    ego_to_worlds = np.stack([ego_to_world0, ego_to_world1], axis=0)
    iou_scores, object_trajectories, tracked_objects = ComputeTemporalShapeConsistency(car_occs, flows, ego_to_worlds)
    print("IoU scores for tracked objects: ", iou_scores)
    print("Object trajectories: ", object_trajectories)

    # Build and test with GMM model
    gmm_car = FindGMMForCategory(category_id=1, data_root=DATA_ROOT)
    scores = ComputeObjectLikelihoods(car_occ_0, gmm_car)
    print("Object likelihood scores: ", scores[0])



