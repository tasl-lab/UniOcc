# Copyright (c) 2025. All rights reserved.
# Licensed under the MIT License.
#
# This script contains the visualization utilities for UniOcc.
# Note: when the open3d window shows, move your mouse around,
# press p to save camera location.

import open3d as o3d
import numpy as np
import math
import argparse

# Constants
FREE_LABEL = 10
LABEL_ROAD = 7
VOXEL_SIZE = [0.4, 0.4, 0.4]
TGT_POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]

# Strong Color map for visualization
COLOR_MAP = np.array([
    [255, 255, 255, 255],   # 0 undefined       white
    [0, 150, 245, 255],     # 1 car             blue
    [255, 192, 203, 255],   # 2 bicycle         pink
    [200, 180, 0, 255],     # 3 motorcycle      dark orange
    [255, 0, 0, 255],       # 4 pedestrian      red
    [255, 240, 150, 255],   # 5 traffic_cone    light yellow
    [0, 175, 0, 255],       # 6 vegetation      green
    [255, 0, 255, 255],     # 7 road            dark pink
    [0, 175, 0, 255],       # 8 terrain         green
    [230, 230, 250, 255],   # 9 building        white
    [0, 0, 0, 0],           # 10 free           black
    [128, 128, 128, 255],   # 11
    [211, 211, 211, 255],   # 12 reserved       gray
    [120, 200, 255, 255],   # 13 reserved       light blue
    [255, 220, 230, 255],   # 14 reserved       light pink
    [120, 200, 255, 255],   # 15 reserved       light orange
    [255, 100, 100, 255],   # 16 reserved       light red
    [255, 245, 190, 255],   # 17 reserved       light yellow
    [100, 220, 100, 255],   # 18 reserved       light green
    [255, 100, 255, 255],   # 19 reserved       light magenta
], dtype=np.float32)


def __voxel_to_points__(voxel: np.ndarray,
                        occupancy_mask: np.ndarray,
                        voxel_size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert voxel data into a point cloud representation.

    Args:
        voxel (np.ndarray): The 3D voxel array (e.g., labels or intensities).
        occupancy_mask (np.ndarray): A boolean mask indicating which voxels are 'occupied'.
        voxel_size (tuple): The size of each voxel in (x, y, z) dimensions.

    Returns:
        tuple:
            - points (np.ndarray): The 3D coordinates of occupied voxels, shape (N, 3).
            - voxel_values (np.ndarray): The values at the occupied locations in 'voxel'.
            - occupancy_indices (np.ndarray): The [x, y, z] indices for the occupied voxels.
    """
    occupied_indices = np.where(occupancy_mask)
    # Convert indices to continuous coordinates by multiplying by voxel size
    points = np.column_stack((
        occupied_indices[0] * voxel_size[0],
        occupied_indices[1] * voxel_size[1],
        occupied_indices[2] * voxel_size[2]
    ))
    return points, voxel[occupied_indices], occupied_indices


def __voxel_profile__(points: np.ndarray,
                      voxel_size: tuple[float, float, float]) -> np.ndarray:
    """
    Generate 3D bounding box profiles for each point/voxel center.

    Args:
        points (np.ndarray): The array of 3D points (N, 3).
        voxel_size (tuple): The (x, y, z) size of each voxel.

    Returns:
        np.ndarray: An array of shape (N, 7) containing
                    [center_x, center_y, center_z, w, l, h, yaw].
    """
    # centers: x, y, z with z adjusted to align with the bottom of the voxel
    centers = np.column_stack([
        points[:, 0],
        points[:, 1],
        points[:, 2] - voxel_size[2] / 2.0
    ])

    # width, length, height arrays
    w_l_h = np.column_stack([
        np.full(centers.shape[0], voxel_size[0]),
        np.full(centers.shape[0], voxel_size[1]),
        np.full(centers.shape[0], voxel_size[2])
    ])

    # yaw is zero for all
    yaw = np.zeros((centers.shape[0], 1))

    return np.hstack((centers, w_l_h, yaw))


def __rotz__(angle: float) -> np.ndarray:
    """
    Compute a rotation matrix for rotation about the Z-axis by 'angle'.

    Args:
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def __compute_box_3d__(center: np.ndarray,
                       size: np.ndarray,
                       heading_angle: np.ndarray) -> np.ndarray:
    """
    Compute 3D bounding box corners given center, box size, and heading angle.

    Args:
        center (np.ndarray): Shape (N, 3). Center coordinates of boxes.
        size (np.ndarray): Shape (N, 3). [width, length, height].
        heading_angle (np.ndarray): Shape (N, 1). Heading angles in radians.

    Returns:
        np.ndarray: Shape (N, 8, 3). Corner points for each box.
    """
    # size: [w, l, h] -> reorder or rename for clarity
    h, w, l = size[:, 2], size[:, 0], size[:, 1]

    # Adjust heading angle to match desired orientation
    heading_angle = -heading_angle - math.pi / 2.0

    # Raise center z by half height to align with bottom
    center[:, 2] += h / 2.0

    # Half-sizes for x/y/z corners
    l_half = (l / 2.0)[:, None]
    w_half = (w / 2.0)[:, None]
    h_half = (h / 2.0)[:, None]

    # Corner offsets in local coordinates
    x_corners = np.hstack([-l_half, l_half, l_half, -l_half,
                           -l_half, l_half, l_half, -l_half])[..., None]
    y_corners = np.hstack([ w_half, w_half, -w_half, -w_half,
                            w_half, w_half, -w_half, -w_half])[..., None]
    z_corners = np.hstack([ h_half, h_half, h_half, h_half,
                           -h_half, -h_half, -h_half, -h_half])[..., None]

    # Combine local offsets
    corners_3d = np.concatenate([x_corners, y_corners, z_corners], axis=2)

    # Translate to world coordinates
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]

    return corners_3d


def __generate_ego_car__() -> np.ndarray:
    """
    Generate a voxelized representation of the ego car in local space.

    Returns:
        np.ndarray: Shape (N, 3). 3D coordinates of ego car voxel points.
    """
    ego_range = [-1.8, -0.8, -0.8, 1.8, 0.8, 0.8]
    ego_voxel_size = [0.1, 0.1, 0.1]

    # Compute voxel dimensions
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])

    # Create a grid of voxel indices
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)

    # Convert indices to 3D points
    ego_point_x = ((ego_xyz[:, 0:1] + 0.5) / ego_xdim) * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = ((ego_xyz[:, 1:2] + 0.5) / ego_ydim) * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = ((ego_xyz[:, 2:3] + 0.5) / ego_zdim) * (ego_range[5] - ego_range[2]) + ego_range[2]
    return np.hstack((ego_point_y, ego_point_x, ego_point_z))


def __place_ego_car_at_position__(ego_points: np.ndarray,
                                  map_center: np.ndarray) -> np.ndarray:
    """
    Move ego car points so that it is placed at 'map_center'.

    Args:
        ego_points (np.ndarray): Shape (N, 3). Original ego car points.
        map_center (np.ndarray): Shape (3,). Desired center point in space.

    Returns:
        np.ndarray: Translated ego car points.
    """
    # Center the ego car around (0,0,0)
    ego_points[:, 0] -= np.mean(ego_points[:, 0])
    ego_points[:, 1] -= np.mean(ego_points[:, 1])
    ego_points[:, 2] -= np.mean(ego_points[:, 2])

    # Then shift it to map_center
    ego_points[:, 0] += map_center[0]
    ego_points[:, 1] += map_center[1]
    ego_points[:, 2] += map_center[2]
    return ego_points


def FillRoadInOcc(occ: np.ndarray) -> np.ndarray:
    """
    Fill the lowest 'road' slice in an occupancy grid to ensure continuous road labeling.

    Args:
        occ (np.ndarray): A 3D occupancy array (possibly with labeled voxels).

    Returns:
        np.ndarray: The modified occupancy array with the road label filled in its lowest slice.
    """
    road_mask = (occ == LABEL_ROAD)
    road_level = np.nonzero(road_mask)[2].min()
    occ[:, :, road_level] = LABEL_ROAD
    return occ


def CreateOccHandle(occ: np.ndarray,
                    free_label: int,
                    voxelize: bool = True) -> o3d.visualization.VisualizerWithKeyCallback:
    """
    Create a 3D visualizer handle to display the occupancy grid data.

    Args:
        occ (np.ndarray): The 3D occupancy array.
        free_label (int): The label used for 'free/empty' voxels.
        voxelize (bool): If True, also create bounding boxes around each occupied voxel.

    Returns:
        o3d.visualization.VisualizerWithKeyCallback: The Open3D visualizer handle.
    """
    voxel_mask = (occ != free_label)

    # Convert voxel to points
    colors = COLOR_MAP / 255.0
    points, labels, _ = __voxel_to_points__(occ, voxel_mask, VOXEL_SIZE)
    color_indices = labels % len(colors)
    point_colors = colors[color_indices]

    # Prepare bounding boxes (if voxelize is True)
    bboxes = __voxel_profile__(points, VOXEL_SIZE)
    bbox_corners = __compute_box_3d__(bboxes[:, 0:3],
                                   bboxes[:, 3:6],
                                   bboxes[:, 6:7])
    corner_indices_base = np.arange(0, bbox_corners.shape[0] * 8, 8)
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]).reshape((1, 12, 2))
    edges = np.tile(edges, (bbox_corners.shape[0], 1, 1))
    edges += corner_indices_base[:, None, None]

    # Create visualizer
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window()

    # Set background color
    render_opts = visualizer.get_render_option()
    render_opts.background_color = np.array([1.0, 1.0, 1.0])

    # Create and add the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors[:, :3])
    visualizer.add_geometry(pcd)

    # Add a coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.6, origin=[0, 0, 0])
    visualizer.add_geometry(coord_frame)

    # Optionally add bounding boxes
    if voxelize:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox_corners.reshape(-1, 3))
        line_set.lines = o3d.utility.Vector2iVector(edges.reshape(-1, 2))
        line_set.paint_uniform_color((0.5, 0.5, 0.5))
        visualizer.add_geometry(line_set)

    return visualizer


def AddFlowToVisHandle(vis_handle: o3d.visualization.VisualizerWithKeyCallback,
                       flow: np.ndarray,
                       resolution: float = 0.4) -> None:
    """
    Add flow vectors to an existing Open3D visualizer.

    Args:
        vis_handle (o3d.visualization.VisualizerWithKeyCallback): The Open3D visualizer.
        flow (np.ndarray): Shape (X, Y, Z, 3). Flow vectors for each voxel.
        resolution (float): Scale factor for the flow vectors.
    """
    # Indices where flow is not zero
    nonzero_indices = np.argwhere(np.any(flow != (0, 0, 0), axis=-1))
    nonzero_vectors = flow[nonzero_indices[:, 0],
                           nonzero_indices[:, 1],
                           nonzero_indices[:, 2]]

    start_points = nonzero_indices * np.array(VOXEL_SIZE)
    end_points = start_points + nonzero_vectors * resolution

    # Create line segments
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack((start_points, end_points)))

    num_lines = start_points.shape[0]
    lines = np.column_stack((np.arange(num_lines),
                             np.arange(num_lines, 2 * num_lines)))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1.0, 0.0, 0.0])

    vis_handle.add_geometry(line_set)


def AddCenterEgoToVisHandle(occ: np.ndarray,
                            vis_handle: o3d.visualization.VisualizerWithKeyCallback) -> None:
    """
    Add the ego car model at the center of the occupancy grid to the visualizer.

    Args:
        occ (np.ndarray): The 3D occupancy array.
        vis_handle (o3d.visualization.VisualizerWithKeyCallback): The visualizer.
    """
    map_center = np.array(occ.shape) * np.array(VOXEL_SIZE) / 2.0
    map_center[2] -= 1.2  # Adjust for 'floor' offset

    ego_points = __generate_ego_car__()
    ego_points = __place_ego_car_at_position__(ego_points, map_center)

    ego_pcd = o3d.geometry.PointCloud()
    ego_pcd.points = o3d.utility.Vector3dVector(ego_points)
    vis_handle.add_geometry(ego_pcd)

def VisualizeOcc(occ: np.ndarray,
                 free_label: int = FREE_LABEL,
                 show_ego: bool = False) -> o3d.visualization.VisualizerWithKeyCallback:
    """
    Visualize a 3D occupancy grid.

    Args:
        occ (np.ndarray): The 3D occupancy array.
        free_label (int): The label corresponding to 'free/empty' voxels.
        show_ego (bool): If True, also place the ego car at the center.

    Returns:
        o3d.visualization.VisualizerWithKeyCallback: The Open3D visualizer.
    """
    vis_handle = CreateOccHandle(occ, free_label)

    if show_ego:
        AddCenterEgoToVisHandle(occ, vis_handle)

    vis_handle.poll_events()
    vis_handle.update_renderer()

    return vis_handle


def VisualizeOccFlow(occ: np.ndarray,
                     flow: np.ndarray,
                     free_label: int = FREE_LABEL,
                     show_ego: bool = False) -> o3d.visualization.VisualizerWithKeyCallback:
    """
    Visualize a 3D occupancy grid along with flow vectors.

    Args:
        occ (np.ndarray): The 3D occupancy array.
        flow (np.ndarray): The 4D flow array (X, Y, Z, 3).
        free_label (int): The label corresponding to 'free/empty' voxels.
        show_ego (bool): If True, also place the ego car at the center.

    Returns:
        o3d.visualization.VisualizerWithKeyCallback: The Open3D visualizer.
    """
    vis_handle = CreateOccHandle(occ, free_label)
    AddFlowToVisHandle(vis_handle, flow)

    if show_ego:
        AddCenterEgoToVisHandle(occ, vis_handle)

    vis_handle.poll_events()
    vis_handle.update_renderer()

    return vis_handle


def VisualizeOccFlowFile(file_path: str, free_label: int = FREE_LABEL) -> o3d.visualization.VisualizerWithKeyCallback:
    """
    Load occupancy and flow data from a file, then visualize them in 3D.

    Args:
        file_path (str): Path to the .npz file containing 'occ_label' and 'occ_flow_forward'.
        free_label (int): The label corresponding to 'free/empty' voxels.

    Returns:
        o3d.visualization.VisualizerWithKeyCallback: The Open3D visualizer.
    """
    data = np.load(file_path, allow_pickle=True)
    voxel_label = data['occ_label']
    occ_flow = data['occ_flow_forward']

    # new_label = np.ones_like(voxel_label) * free_label
    # new_label[:, :, 2:14] = voxel_label[:, :, 4:]


    vis_handle = CreateOccHandle(voxel_label, free_label)
    AddFlowToVisHandle(vis_handle, occ_flow)
    AddCenterEgoToVisHandle(voxel_label, vis_handle)

    vis_handle.poll_events()
    vis_handle.update_renderer()
    return vis_handle

def RotateO3DCamera(viz_handle, camera_json_path):
    ctr = viz_handle.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters(camera_json_path)
    ctr.convert_from_pinhole_camera_parameters(parameters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize 3D Occupancy and Flow data.")
    parser.add_argument('--file_path', type=str, default='example/COHFF-val5/2021_08_18_19_48_05/1045/000068.npz',
                        help="Path to the .npz file containing 'occ_label' and 'occ_flow_forward'.")
    args = parser.parse_args()

    print("Press p to save camera position.")

    VisualizeOccFlowFile(args.file_path).run()
