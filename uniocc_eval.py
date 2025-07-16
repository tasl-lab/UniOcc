# Copyright (c) 2025. All rights reserved.
# Licensed under the MIT License.
#
# This script provides a reference implementation for occupancy forecasting
# without disclosing any proprietary details. It includes:
#   - GMM-based shape likelihood evaluation
#   - Temporal shape consistency evaluation
#   - Static background consistency evaluation
#   - Standard IoU & mIoU evaluation

from matplotlib import pyplot as plt
from shapely import Point
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

from uniocc_viz import *
from uniocc_utils import *

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
    labeled_grid, _ = SegmentVoxels(binary_occ)
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

def ComputeTemporalShapeConsistencyByTracking(binary_occs,
                                              flows,
                                              ego_to_worlds,
                                              threshold=2.0):
    """
    Example method to measure temporal shape consistency by aligning
    predicted voxel sets across frames. This function also performs
    occupancy space tracking.

    Parameters
    ----------
    binary_occs : np.ndarray, shape (T, L, W, H)
        Binary occupancy grids across T timesteps.
    flows : np.ndarray, shape (T, L, W, H, 3)
        Flow fields at each frame.
    ego_to_worlds : np.ndarray, shape (T, 4, 4)
        Ego pose transformations for each frame.
        Note: this field can be originated from the ego's pose at t=0,
        in which case we esimate it using EstimateEgoMotionFromFlows().
    threshold : float
        Distance threshold in meters for centroid-based association. This
        value can be eye-balled from the small values in the cost_matrix below.

    Returns
    -------
    iou_scores : dict
        {object_id: mean iou across frames}
    trajectories : dict
        {object_id: {timestep: centroid positions in global coordinates}}
    tracked_objects : dict
        {object_id: {timestep: object_voxels}}
    """
    object_trajectories, tracked_objects = (
        TrackOccObjects(binary_occs, flows, ego_to_worlds, threshold=threshold))

    iou_scores = {}
    for oid, vox_list in tracked_objects.items():
        if len(vox_list) < 2:
            continue

        aligned_shapes = []
        reference_pca = None

        # Align shapes with PCA
        for vox_block in vox_list.items():
            if len(vox_block) < 3:
                continue
            shifted = AlignToCentroid(vox_block)
            rotated, pca = AlignWithPCA(shifted, reference_pca)
            if reference_pca is None:
                reference_pca = pca
            aligned_shapes.append(rotated)

        # IoU across frames
        ious = []
        for i in range(len(aligned_shapes) - 1):
            shapeA = aligned_shapes[i]
            shapeB = aligned_shapes[i+1]
            gridA = RasterizeCoordsToGrid(shapeA, shape=(binary_occs.shape[1], binary_occs.shape[2], binary_occs.shape[3]))
            gridB = RasterizeCoordsToGrid(shapeB, shape=(binary_occs.shape[1], binary_occs.shape[2], binary_occs.shape[3]))
            ious.append(ComputeGridIoU(gridA, gridB))
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
        Note: this field can be originated from the ego's pose at t=0,
        in which case we esimate it using EstimateEgoMotionFromFlows().

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
        mesh_coords_ego = OccFrameToEgoFrame(mesh_coords)

        # Step 3: advance the meshgrid to the next frame
        mesh_coords_ego = np.concatenate([mesh_coords_ego, np.ones((len(mesh_coords_ego), 1))], axis=-1)
        mesh_coords_ego_next = (curr_to_next_T @ mesh_coords_ego.T).T

        # Step 4: Floor to int
        mesh_coords_ego_next = EgoFrameToOccFrame(mesh_coords_ego_next)

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
        static_voxels_coords_ego = OccFrameToEgoFrame(static_voxels_coords)

        # Step 8: advance the voxels to the next frame
        static_voxels_coords_ego = np.concatenate(
            [static_voxels_coords_ego, np.ones((len(static_voxels_coords_ego), 1))], axis=-1)
        static_voxels_coords_ego_next = (curr_to_next_T @ static_voxels_coords_ego.T).T

        # Step 9: Floor to int
        static_voxels_coords_ego_next = EgoFrameToOccFrame(static_voxels_coords_ego_next)

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
    iou_scores, object_trajectories, tracked_objects = ComputeTemporalShapeConsistencyByTracking(car_occs, flows, ego_to_worlds)
    print("IoU scores for tracked objects: ", iou_scores)
    print("Object trajectories: ", object_trajectories)

    # Build and test with GMM model
    gmm_car = FindGMMForCategory(category_id=1, data_root=DATA_ROOT)
    scores = ComputeObjectLikelihoods(car_occ_0, gmm_car)
    print("Object likelihood scores: ", scores[0])



