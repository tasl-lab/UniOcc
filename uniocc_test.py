# Copyright (c) 2025. All rights reserved.
# Licensed under the MIT License.
#
# This script contains the unit tests for the RANSAC-based camera motion estimation


from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
import numpy as np
from uniocc_utils import __invert_rigid_transform__, __ransac_point_motion__

def test_identity_motion():
    # Generate 10 random 3D points
    points_src = np.random.rand(10, 3)

    # No motion (zero vectors), i.e., points_dst = points_src
    motion_vectors = np.zeros_like(points_src)

    # Estimate camera motion
    R_cam, t_cam = __invert_rigid_transform__(
        *__ransac_point_motion__(points_src, motion_vectors))

    # Camera didn't move → expect identity rotation and zero translation
    assert_allclose(R_cam, np.eye(3), atol=1e-6)
    assert_allclose(t_cam, np.zeros(3), atol=1e-6)
    print("✅ test_identity_motion passed")

def test_pure_translation():
    # Ground-truth camera motion: no rotation, known translation
    t_cam_gt = np.array([0.5, -1.0, 0.25])
    R_cam_gt = np.eye(3)

    # Corresponding point transform: inverse translation, no rotation
    t_point = -t_cam_gt
    R_point = np.eye(3)

    # Simulate 20 source points
    points_src = np.random.rand(20, 3)
    # Apply point transform to get destination points
    points_dst = points_src @ R_point.T + t_point
    # Motion vectors = points_dst - points_src
    motion_vectors = points_dst - points_src

    # Estimate camera motion
    R_cam, t_cam = __invert_rigid_transform__(
        *__ransac_point_motion__(points_src, motion_vectors))

    # Expect identity rotation and the ground-truth translation
    assert_allclose(R_cam, R_cam_gt, atol=1e-6)
    assert_allclose(t_cam, t_cam_gt, atol=1e-6)
    print("✅ test_pure_translation passed")

def test_pure_rotation():
    # Define camera rotation: 30° around Z, 15° around Y, 10° around X
    R_cam_gt = Rotation.from_euler('ZYX', [30, 15, 10], degrees=True)
    t_cam_gt = np.zeros(3)  # No translation

    # Invert the camera rotation to get point rotation
    R_point = R_cam_gt.inv()

    # Generate 15 source points
    points_src = np.random.rand(15, 3)
    # Apply point rotation to get destination points
    points_dst = R_point.apply(points_src)
    motion_vectors = points_dst - points_src

    # Estimate camera motion
    R_cam_est, t_cam_est = __invert_rigid_transform__(
        *__ransac_point_motion__(points_src, motion_vectors))

    # Compare estimated rotation to ground truth
    assert_allclose(R_cam_est, R_cam_gt.as_matrix(), atol=1e-6)
    # Expect near-zero translation
    assert_allclose(t_cam_est, np.zeros(3), atol=1e-6)
    print("✅ test_pure_rotation passed")

def test_se3_motion():
    # Define ground-truth camera rotation and translation
    R_cam_gt = Rotation.from_euler('XYZ', [10, -20, 45], degrees=True)
    t_cam_gt = np.array([-0.2, 0.4, 0.6])

    # Invert to get point motion
    R_point = R_cam_gt.inv().as_matrix()
    t_point = -R_point @ t_cam_gt

    # Create 30 random 3D points
    points_src = np.random.randn(30, 3)
    points_dst = (R_point @ points_src.T).T + t_point
    motion_vectors = points_dst - points_src

    # Estimate camera motion
    R_cam_est, t_cam_est = __invert_rigid_transform__(
        *__ransac_point_motion__(points_src, motion_vectors))

    # Expect estimated values to match ground truth
    assert_allclose(R_cam_est, R_cam_gt.as_matrix(), atol=1e-6)
    assert_allclose(t_cam_est, t_cam_gt, atol=1e-6)
    print("✅ test_se3_motion passed")

def test_single_point():
    # One point in space
    point_src = np.array([[1.0, 2.0, 3.0]])
    motion_vector = np.array([[0.2, -0.1, 0.5]])

    # Camera translation should be negative of motion vector
    expected_R = np.eye(3)
    expected_t = -motion_vector[0]

    # Estimate motion
    R_cam, t_cam = __invert_rigid_transform__(
        *__ransac_point_motion__(point_src, motion_vector))

    assert_allclose(R_cam, expected_R, atol=1e-6)
    assert_allclose(t_cam, expected_t, atol=1e-6)
    print("✅ test_single_point passed")

def test_ransac_with_outliers():
    np.random.seed(42)

    # Define clean camera motion
    R_cam_gt = Rotation.from_euler('XYZ', [15, -10, 20], degrees=True)
    t_cam_gt = np.array([0.3, -0.5, 0.2])
    R_point = R_cam_gt.inv().as_matrix()
    t_point = -R_point @ t_cam_gt

    # Inliers (correct motion)
    points_in = np.random.uniform(-1, 1, (50, 3))
    points_dst_in = (R_point @ points_in.T).T + t_point
    motion_in = points_dst_in - points_in

    # Outliers (random noise)
    points_out = np.random.uniform(0, 0, (10, 3))
    motion_out = np.random.uniform(0, 0, (10, 3))

    # Combine
    points_all = np.vstack([points_in, points_out])
    motions_all = np.vstack([motion_in, motion_out])

    # Estimate with RANSAC
    R_cam_est, t_cam_est = __invert_rigid_transform__(
        *__ransac_point_motion__(points_all, motions_all))

    # Should match ground-truth motion (tolerate minor deviation)
    assert_allclose(R_cam_est, R_cam_gt.as_matrix(), atol=5e-2)
    assert_allclose(t_cam_est, t_cam_gt, atol=5e-2)
    print("✅ test_ransac_with_outliers passed")

if __name__ == "__main__":
    test_identity_motion()
    test_pure_translation()
    test_pure_rotation()
    test_se3_motion()
    test_single_point()
    test_ransac_with_outliers()




