# **UniOcc**: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2503.24381-<COLOR>.svg)](https://arxiv.org/abs/2503.24381)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/tasl-lab/uniocc)

Alternative: [![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?logo=googledrive&logoColor=fff)](https://drive.google.com/drive/folders/18TSklDPPW1IwXvfTb6DtSNLhVud5-8Pw?usp=sharing)
[![Baidu](https://img.shields.io/badge/Baidu-Pan-2932E1?logo=Baidu&logoColor=white)](https://pan.baidu.com/s/17Pk2ni8BwwU4T2fRmVROeA?pwd=kdfj)

> Autonomous Driving researchers, have you ever been bothered by the fact that popular datasets all have their different
> formats, and standardizing them is a pain? Have you ever been frustrated by the difficulty of just understanding
> the file semantics? This challenge is even worse in the occupancy domain. But, **UniOcc is here to help.**

**UniOcc** is a unified framework for occupancy forecasting, single-frame occupancy prediction, and occupancy flow 
estimation in autonomous driving. By integrating multiple real-world (nuScenes, Waymo) and 
synthetic (CARLA, OpenCOOD) datasets, UniOcc enables multi-domain training, seamless cross-dataset evaluation, 
and robust benchmarking across diverse driving environments.

[Yuping Wang<sup>1,2</sup>*](https://www.linkedin.com/in/yuping-wang-5a7178185/),
[Xiangyu Huang<sup>3</sup>*](https://www.linkedin.com/in/xiangyu-huang-606089292),
[Xiaokang Sun<sup>1</sup>*](https://scholar.google.com/citations?user=2sWnAjQAAAAJ&hl=en),
[Mingxuan Yan<sup>1</sup>](https://waterhyacinthinnanhu.github.io/),
[Shuo Xing<sup>4</sup>](https://shuoxing98.github.io/),
[Zhengzhong Tu<sup>4</sup>](https://vztu.github.io/),
[Jiachen Li<sup>1</sup>](https://jiachenli94.github.io/)

<sup>1</sup>University of California, Riverside; <sup>2</sup>University of Michigan; <sup>3</sup>University of Wisconsin-Madison; <sup>4</sup>Texas A&M University

---

## Supported Tasks

- **Occupancy Forecasting**: Predict future 3D occupancy grids over time given historical occupancies or camera inputs.
- **Occupancy Prediction**: Generate detailed 3D occupancy grids from camera inputs.
- **Flow Estimation**: Provides forward and backward voxel-level flow fields for more accurate motion modeling and object tracking.  
- **Multi-Domain Dataset Integration**: Supports major autonomous driving datasets (nuScenes, Waymo, CARLA, etc.) with consistent annotation and evaluation pipelines.  
- **Ground-Truth-Free Metrics**: Beyond standard IoU, introduces shape and dimension plausibility checks for generative or multi-modal tasks.  
- **Cooperative Autonomous Driving**: Enables multi-agent occupancy fusion and forecasting, leveraging viewpoint diversity from multiple vehicles.  

---

## Pre-requisites

We simplify our benchmark so you only need:

- Python 3.9 or higher
  ```shell
  pip install torch torchvision pillow tqdm numpy open3d
  ```
- Huggingface
  ```shell
  pip install "huggingface_hub[cli]"
  ```
  
You **do not** need:
- nuscenes-devkit
- waymo-open-dataset
- tensorflow

---

## Dataset Download

The UniOcc dataset is available on HuggingFace. The size of each dataset is as follows:

| Dataset Name                         | Number of Scenes | Training Instances | Size (GB) |
|--------------------------------------|-----------------:|-------------------:|----------:|
| NuScenes-via-Occ3D-2Hz-mini          |               10 |                404 |       0.6 |
| NuScenes-via-OpenOccupancy-2Hz-mini  |                ~ |                  ~ |       0.4 |
| NuScenes-via-SurroundOcc-2Hz-mini    |                ~ |                  ~ |       0.4 |
| NuScenes-via-OpenOccupancy-2Hz-val   |              150 |              6,019 |       6.2 |
| NuScenes-via-Occ3D-2Hz-val           |                ~ |                  ~ |       9.1 |
| NuScenes-via-SurroundOcc-2Hz-val     |                ~ |                  ~ |       6.2 |
| NuScenes-via-Occ3D-2Hz-train         |              700 |             28,130 |      41.2 |
| NuScenes-via-OpenOccupancy-2Hz-train |                ~ |                  ~ |      28.3 |   
| NuScenes-via-SurroundOcc-2Hz-train   |                ~ |                  ~ |      28.1 |
| Waymo-via-Occ3D-2Hz-mini             |               10 |                397 |      0.84 |
| Waymo-via-Occ3D-2Hz-val              |              200 |               8069 |      15.4 |   
| Waymo-via-Occ3D-2Hz-train            |              798 |             31,880 |      59.5 |
| Waymo-via-Occ3D-10Hz-mini            |               10 |              1,967 |       4.0 |
| Waymo-via-Occ3D-10Hz-val             |              200 |             39,987 |      74.4 |   
| Waymo-via-Occ3D-10Hz-train           |              798 |            158,081 |     286.6 |
| Carla-2Hz-mini                       |                2 |                840 |       1.0 |
| Carla-2Hz-val                        |                4 |              2,500 |       2.9 |   
| Carla-2Hz-train                      |               11 |              8,400 |       9.3 |
| Carla-10Hz-mini                      |                2 |              4,200 |       5.0 |
| Carla-10Hz-val                       |                4 |             12,500 |      15.0 |   
| Carla-10Hz-train                     |               11 |             42,200 |      46.5 | 
| OPV2V-10Hz-val          |   9 |                  8035  |       23.5    |  
| OPV2V-10Hz-train        |   43 |              18676      |       49.8    |
| OPV2V-10Hz-test         |   16 |              3629      |       9.6    |


To download each dataset, use the following command (recommend you to download only the folders you need):

```shell
huggingface-cli download tasl-lab/uniocc --include "NuScenes-via-Occ3D-2Hz-mini*" --repo-type dataset --local-dir ./datasets
huggingface-cli download tasl-lab/uniocc --include "Carla-2Hz-train*" --repo-type dataset --local-dir ./datasets
...
```

---
## Contents

Inside each dataset, you will find the following files:

```
datasets
├── NuScenes-via-Occ3D-2Hz-mini
│   ├── scene_infos.pkl
│   ├── scene_001           <-- Scene Name
│   │   ├── 1.npz           <-- Time Step
│   │   ├── 2.npz
│   │   ├── ...
│   ├── scene_002
│   ...
├── OpenCOOD-via-OpV2V-10Hz-val
│   ├── scene_infos.pkl
│   ├── scene_001           <-- Scene Name
│   │   ├── 1061            <-- CAV ID
│   │   │   │   ├── 1.npz   <-- Time Step
│   │   │   │   ├── 2.npz
│   │   │   │   ├── ...
│   │   │   ├── scene_002
│   ...
```

- `scene_infos.pkl`: A list of dictionaries, each containing the scene name, start and end frame, and other metadata.
- `scene_XXX`: A directory containing the data for a single scenario.
- `YYY.npz`: A NumPy file containing the following data for a single time step.
  - `occ_label`: A 3D occupancy grid (L x W x H) with semantic labels.
  - `occ_mask_camera`: A 3D grid (L x W x H) with binary values with `1` indicating the voxel is in the camera FOV and `0` otherwise.
  - `occ_flow_forward`: A 3D flow field (L x W x H x 3) with voxel flow vectors pointing to each voxel's next frame coordinate. In the last frame, flow is 0. The unit of the flow is num_voxels.
  - `occ_flow_backward`: A 3D flow field (L x W x H x 3) with voxel flow vectors pointing to each voxel's previous frame coordinate. In the first frame, flow is 0. The unit of the flow is num_voxels.
  - `ego_to_world_transformation`: A 4x4 transformation matrix from the ego vehicle to the world coordinate system.
  - `cameras`: A list of camera objects with intrinsic and extrinsic parameters.
    - `name`: The camera name (i.e. CAM_FRONT in nuScenes).
    - `filename`: The **relative path** to the camera image from the original datasource (i.e. nuScenes).
    - `intrinsics`: A 3x3 intrinsic matrix.
    - `extrinsics`: A 4x4 extrinsic matrix from the camera to the ego vehicle's LiDAR.
  - `annotations`: A list of objects with bounding boxes and class labels.
    - `token`: The object token, consistent with their original datasource.
    - `agent_to_ego`: A 4x4 transformation matrix from the object to the ego vehicle.
    - `agent_to_world`: A 4x4 transformation matrix from the object to the world coordinate system.
    - `size`: The size of the agent's bounding box in meters. (Length, Width, Height)
    - `category_id`: The object category (i.e. `1` for car, `4` for pedestrian, etc.)

  
<img src="figures/flow.png" alt="Alt Text" style="width:80%; height:auto;">

> Note: we provide the flow annotation to both dynmaic voxels (agents) and static voxels (envrionments) in the scene.

---
## Visualizing the Dataset

You can visualize the dataset using the provided `viz.py` script. For example:

```shell
python uniocc_viz.py --file_path datasets/NuScenes-via-Occ3D-2Hz-mini/scene-0061/0.npz
```

In this script, we also provide the API to visualize any 3D occupancy grid, with or without a flow field.

---
## Usage

### Without Camera Images

If you only need the occupancy data, you can use the provided `uniocc_dataset.py` script to load the dataset.

```python
from uniocc_dataset import UniOcc

dataset_carla_mini = UniOcc(
    data_root="datasets/Carla-2Hz-mini",
    obs_len=8,
    fut_len=12
)

dataset_nusc_mini = UniOcc(
    data_root="datasets/NuScenes-via-Occ3D-2Hz-mini",
    obs_len=8,
    fut_len=12
)

dataset = torch.utils.data.ConcatDataset([dataset_carla_mini, dataset_nusc_mini])
```

### With Camera Images

If you want to use the camera images from nuScenes, Waymo or OpV2V, it is necessary to download them from the original dataset.
- [nuScenes](https://www.nuscenes.org/download)
- [Waymo Open Dataset v1](https://waymo.com/open/download)
  - Convert to KITTI format using [this tool](https://github.com/caizhongang/waymo_kitti_converter)
- [OpV2V](https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V)

You can then provide the root directory to the dataloader to load the camera images.

```python
from uniocc_dataset import UniOcc

dataset_carla_mini = UniOcc(
    data_root="datasets/Carla-2Hz-mini",
    obs_len=8,
    fut_len=12,
    datasource_root="datasets/Carla-2Hz-mini"
)

dataset_nusc_mini = UniOcc(
    data_root="datasets/NuScenes-via-Occ3D-2Hz-mini",
    obs_len=8,
    fut_len=12,
    datasource_root="<YOUR_NUSCENES_ROOT>"  # e.g. <YOUR_NUSCENES_ROOT>/sweeps/CAM_FRONT
)

dataset_waymo_mini = UniOcc(
    data_root="datasets/Waymo-via-Occ3D-2Hz-mini",
    obs_len=8,
    fut_len=12,
    datasource_root="<YOUR_KITTI_WAYMO_ROOT>"  # e.g. <YOUR_KITTI_WAYMO_ROOT>/training/image_0
)

dataset = torch.utils.data.ConcatDataset([dataset_carla_mini, dataset_nusc_mini, dataset_waymo_mini])

```
---

## Occupancy Space Localization, Segmentation, Voxel Alignment, Tracking

In `uniocc_utils.py`, we provide a set of utility functions for occupancy space localization, segmentation, voxel alignment, and tracking. These functions are designed to work with the voxelized occupancy grids and can be used for various tasks such as object tracking, segmentation, and motion estimation.

| **Function**                                | **Description**                                                                                                                              |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `VoxelToCorners`                            | Converts voxel indices to 3D bounding-box corner coordinates for spatial visualization and geometry computations.                            |
| `OccFrameToEgoFrame` / `EgoFrameToOccFrame` | Transforms voxel coordinates between occupancy grid space and the ego-centric coordinate frame using voxel resolution and ego center offset. |
| `AlignToCentroid`                           | Recenters voxel coordinates by subtracting their centroid, aligning the shape around the origin.                                             |
| `RasterizeCoordsToGrid`                     | Converts a list of voxel coordinates into a binary 3D occupancy grid of specified dimensions.                                                |
| `Compute3DBBoxIoU`                          | Approximates 3D IoU by computing overlap between 2D rotated bounding boxes and comparing height extents.                                     |
| `AlignWithPCA`                              | Rotates voxel point clouds to align with principal axes using PCA; supports alignment with a reference PCA basis.                            |
| `ComputeGridIoU`                            | Calculates voxel-wise binary IoU between two occupancy grids of identical shape.                                                             |
| `SegmentVoxels`                             | Performs 3D connected-component labeling (CCL) on an occupancy grid, with filtering by minimum voxel count.                                  |
| `EstimateEgoMotionFromFlows`                | Estimates ego-motion from voxel flow fields over time by extracting static voxels and applying RANSAC-based rigid transform fitting.         |
| `AccumulateTransformations`                 | Composes a sequence of frame-to-frame transformation matrices into global poses over time.                                                   |
| `TrackOccObjects`                           | Tracks objects across frames using voxel flows and estimated ego-motion, returning per-object trajectories and voxel groupings.              |
| `BipartiteMatch`                            | Solves the optimal assignment problem (Hungarian algorithm) to associate predicted and reference objects based on a cost/score matrix.       |

## Visualization API

In `uniocc_viz.py`, we provide a set of visualization functions to render occupancy grids, flow fields, and camera images in 3D using Open3D. These functions can be used to visualize occupancy grids, flow vectors, and the ego vehicle model in a 3D scene.

| **Function**                    | **Description**                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `__voxel_to_points__`           | Converts a 3D voxel array with a boolean occupancy mask into 3D point cloud coordinates, their values, and indices. |
| `__voxel_profile__`             | Creates bounding box profiles for each voxel as `[x, y, z, w, l, h, yaw]` for rendering.                            |
| `__rotz__`                      | Computes a Z-axis rotation matrix given an angle in radians.                                                        |
| `__compute_box_3d__`            | Calculates 8-corner 3D box coordinates from voxel centers, dimensions, and yaw angles.                              |
| `__generate_ego_car__`          | Produces a voxelized point cloud representation of the ego vehicle centered at the origin.                          |
| `__place_ego_car_at_position__` | Translates ego vehicle voxels to a specified center location in the scene.                                          |
| `FillRoadInOcc`                 | Ensures the bottom-most slice of the occupancy grid contains labeled road voxels for consistent visualization.      |
| `CreateOccHandle`               | Builds a full Open3D visualizer, rendering occupancy grids with color and optional bounding boxes.                  |
| `AddFlowToVisHandle`            | Draws flow vectors on the Open3D visualizer as red line segments for motion inspection.                             |
| `AddCenterEgoToVisHandle`       | Adds the ego vehicle model to the visualizer at the center of the occupancy scene.                                  |
| `VisualizeOcc`                  | Creates a visualizer for a static occupancy grid and optionally adds the ego vehicle.                               |
| `VisualizeOccFlow`              | Visualizes both occupancy and voxel-level flow vectors in 3D, with optional ego car rendering.                      |
| `VisualizeOccFlowFile`          | Loads `.npz` files containing occupancy and flow data, and visualizes them together.                                |
| `RotateO3DCamera`               | Loads Open3D camera parameters from a JSON file and applies them to the current visualizer view.                    |



## 📏 Evaluation 

Additional dependencies:
```shell
pip install shapely matplotlib scikit-learn pickle
```

Demo (needs a sample dataset in `datasets/`):
```shell
python uniocc_eval.py
```

We provide these evaluation APIs, as described in our paper.


| Function | Description                                                                                                                                                         |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `FindGMMForCategory` | Fits the best Gaussian-Mixture Model (GMM) to all length-width-height triples of a chosen class, creating a “realism” prior for that object category.               |
| `ComputeObjectLikelihoods` | Segments each object in a binary occupancy grid and scores its bounding-box dimensions against the pretrained GMM, returning plausibility probabilities and counts. |
| `ComputeTemporalShapeConsistencyByTracking` | Tracks every object across frames using voxel flows, aligns shapes, and reports the mean IoU of consecutive shapes, higher = smoother temporal geometry.            |
| `ComputeStaticConsistency` | Warps static voxels from frame *t* to *t + 1* via ego motion and measures how well they overlap, giving an IoU-style score for background stability.                |
| `ComputeIoU` | Computes the standard intersection-over-union between two mono-label occupancy grids while ignoring a specified “free-space” label.                                 |
| `ComputeIoUForCategory` | Same as `ComputeIoU`, but restricted to voxels of a single semantic class, enabling per-category performance evaluation.                                            |

## Checklist

- [x] Release non-cooperative datasets
- [x] Release cooperative dataset
- [x] Release the dataset API
- [x] Release the visualization script
- [x] Release the evaluation scripts
- [x] Release the occupancy segmentation, localization, tracking scripts
- [x] Release data generation scripts


---
## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{wang2025uniocc,
  title={Uniocc: A unified benchmark for occupancy forecasting and prediction in autonomous driving},
  author={Wang, Yuping and Huang, Xiangyu and Sun, Xiaokang and Yan, Mingxuan and Xing, Shuo and Tu, Zhengzhong and Li, Jiachen},
  journal={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
