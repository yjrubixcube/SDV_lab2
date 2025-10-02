#!/usr/bin/env python3
"""
LiDAR-Camera Calibration Homework: PnP Exercise

This homework helps you understand the complete calibration process using
the Perspective-n-Point (PnP) algorithm with OpenCV.

You need to implement three functions:
1. undistort_points() - Remove camera distortion from image points
2. solve_pnp() - Solve the PnP problem to find camera pose
3. create_transform_matrix() - Create 4x4 transformation matrix from PnP result

Scoring: 0-100 points based on transformation matrix accuracy
- Translation error: 50% weight
- Rotation error: 50% weight
"""

import csv
import json
import numpy as np
import cv2
import yaml


# ============================================================================
# Grading Configuration
# ============================================================================
MAX_TRANSLATION_ERROR = 0.10  # meters
MAX_ROTATION_ERROR = 5.0  # degrees
TRANSLATION_POINTS = 25  # points for translation component
ROTATION_POINTS = 25  # points for rotation component
TOTAL_POINTS = TRANSLATION_POINTS + ROTATION_POINTS  # 50 points total


def load_point_pairs(csv_file):
    """
    Load 3D-2D point correspondences from CSV file.

    Args:
        csv_file: Path to CSV file with columns [x_3d, y_3d, z_3d, x_2d, y_2d]

    Returns:
        object_points: Nx3 numpy array of 3D points in world coordinates (meters)
        image_points: Nx2 numpy array of 2D points in image coordinates (pixels)
    """
    object_points = []
    image_points = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 3D point in world coordinates (LiDAR frame)
            object_points.append(
                [float(row["x_3d"]), float(row["y_3d"]), float(row["z_3d"])]
            )
            # 2D point in image coordinates (pixels)
            image_points.append([float(row["x_2d"]), float(row["y_2d"])])

    return np.array(object_points, dtype=np.float32), np.array(
        image_points, dtype=np.float32
    )


def load_camera_info(yaml_file):
    """
    Load camera intrinsic parameters from YAML file.

    Args:
        yaml_file: Path to camera_info.yaml file

    Returns:
        camera_matrix: 3x3 camera intrinsic matrix K
        dist_coeffs: 1x5 distortion coefficients [k1, k2, p1, p2, k3]
    """
    with open(yaml_file, "r") as f:
        camera_info = yaml.safe_load(f)

    # Extract camera matrix K (3x3)
    camera_matrix = np.array(
        camera_info["camera_matrix"]["data"], dtype=np.float32
    ).reshape(3, 3)

    # Extract distortion coefficients (1x5)
    dist_coeffs = np.array(
        camera_info["distortion_coefficients"]["data"], dtype=np.float32
    )

    return camera_matrix, dist_coeffs


def load_ground_truth(json_file):
    """
    Load ground truth transformation matrix for grading.

    Args:
        json_file: Path to ground_truth_transform.json

    Returns:
        transform_matrix: 4x4 transformation matrix
    """
    with open(json_file, "r") as f:
        transform_matrix = json.load(f)

    return np.array(transform_matrix, dtype=np.float32)


# ============================================================================
# TODO 1: Implement undistort_points()
# ============================================================================


def undistort_points(image_points, camera_matrix, dist_coeffs):
    """
    Undistort 2D image points using camera intrinsics and distortion coefficients.

    Camera lenses introduce distortion (radial and tangential). Before solving PnP,
    we need to remove this distortion to get the ideal pinhole camera projection.

    Args:
        image_points: Nx2 array of distorted image points (pixels)
        camera_matrix: 3x3 camera intrinsic matrix K
        dist_coeffs: 1x5 distortion coefficients [k1, k2, p1, p2, k3]

    Returns:
        undistorted_points: Nx2 array of undistorted image points (normalized coordinates)

    Hint:
        Use cv2.undistortPoints() function from OpenCV.
        Documentation: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga55c716492470bfe86b0ee9bf3a1f0f7e

        Note: cv2.undistortPoints() returns normalized coordinates (divided by camera matrix).
        You may need to reshape the output to Nx2 format.

    Expected behavior:
        Input: [[100.5, 200.3], [150.2, 250.8], ...]  (distorted pixels)
        Output: [[0.05, 0.12], [0.08, 0.15], ...]  (undistorted normalized coordinates)
    """
    # TODO: Implement this function
    # Your code here (approximately 1-2 lines)
    
    # print(image_points.shape)
    reshaped_points = np.reshape(image_points, (len(image_points), 1, 2))
    # print(reshaped_points.shape)

    results = cv2.undistortPoints(reshaped_points, camera_matrix, dist_coeffs)

    # print(results.shape)
    results = np.reshape(results, (len(results), 2))
    # print(results.shape)
    return results
    raise NotImplementedError("TODO 1: Implement undistort_points()")


# ============================================================================
# TODO 2: Implement solve_pnp()
# ============================================================================


def solve_pnp(object_points, image_points, camera_matrix):
    """
    Solve the Perspective-n-Point problem to find camera pose.

    The PnP algorithm estimates the camera pose (rotation and translation) given:
    - N >= 4 known 3D points in world coordinates
    - Corresponding 2D projections in image coordinates
    - Camera intrinsic parameters

    Mathematical relationship: s * p = K * (R * P + t)
    Where we solve for R (rotation) and t (translation).

    Args:
        object_points: Nx3 array of 3D points in world coordinates (meters)
        image_points: Nx2 array of undistorted 2D points (normalized coordinates)
        camera_matrix: 3x3 camera intrinsic matrix K

    Returns:
        success: Boolean indicating if PnP solver succeeded
        rvec: 3x1 rotation vector (axis-angle representation)
        tvec: 3x1 translation vector (meters)

    Hint:
        Use cv2.solvePnP() function from OpenCV.
        Documentation: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d

        Parameters to use:
        - objectPoints: object_points (3D world points)
        - imagePoints: image_points (2D image points, already undistorted)
        - cameraMatrix: camera_matrix (K matrix)
        - distCoeffs: np.zeros(5) - zero distortion since points are already undistorted
        - flags: cv2.SOLVEPNP_ITERATIVE - iterative refinement method

    Expected behavior:
        Input: 16 point correspondences, camera matrix
        Output: success=True, rvec=[3x1], tvec=[3x1]
    """
    # TODO: Implement this function
    # Your code here (approximately 3-5 lines)

    # print(object_points.shape)
    # print(image_points.shape)
    # print(camera_matrix)

    retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)

    return retval, rvec, tvec

    raise NotImplementedError("TODO 2: Implement solve_pnp()")


# ============================================================================
# TODO 3: Implement create_transform_matrix() [30 points]
# ============================================================================


def create_transform_matrix(rvec, tvec):
    """
    Create 4x4 homogeneous transformation matrix from rotation vector and translation vector.

    The transformation matrix has the form:
        [R11  R12  R13  tx]
        [R21  R22  R23  ty]
        [R31  R32  R33  tz]
        [ 0    0    0    1]

    Where the top-left 3x3 block is the rotation matrix R,
    and the top-right 3x1 block is the translation vector t.

    Args:
        rvec: 3x1 or 1x3 rotation vector (axis-angle representation from solvePnP)
        tvec: 3x1 or 1x3 translation vector (from solvePnP)

    Returns:
        transform_matrix: 4x4 homogeneous transformation matrix

    Hint:
        1. Convert rotation vector to rotation matrix using cv2.Rodrigues()
        2. Create a 4x4 identity matrix
        3. Set the top-left 3x3 block to the rotation matrix
        4. Set the top-right 3x1 block to the translation vector
        5. The bottom row should remain [0, 0, 0, 1]

    Expected behavior:
        Input: rvec=[3x1], tvec=[3x1]
        Output: 4x4 matrix with rotation and translation combined
    """
    # TODO: Implement this function
    # Your code here (approximately 5-8 lines)

    rot_mat, jacobian = cv2.Rodrigues(rvec)

    transform_mat = np.zeros((4, 4))

    transform_mat[:3, :3] = rot_mat

    transform_mat[:3, 3] = tvec.reshape(3)

    transform_mat[3, 3] = 1

    return transform_mat

    raise NotImplementedError("TODO 3: Implement create_transform_matrix()")


# ============================================================================
# Grading Function (PROVIDED - DO NOT MODIFY)
# ============================================================================


def compute_translation_error(T_student, T_gt):
    """
    Compute translation error between two transformation matrices.

    Args:
        T_student: 4x4 student transformation matrix
        T_gt: 4x4 ground truth transformation matrix

    Returns:
        error: Euclidean distance between translation vectors (meters)
    """
    t_student = T_student[:3, 3]
    t_gt = T_gt[:3, 3]
    return np.linalg.norm(t_student - t_gt)


def compute_rotation_error(T_student, T_gt):
    """
    Compute rotation error between two transformation matrices.

    The error is the rotation angle needed to align the two rotation matrices.
    Uses the formula: angle = arccos((trace(R1^T * R2) - 1) / 2)

    Args:
        T_student: 4x4 student transformation matrix
        T_gt: 4x4 ground truth transformation matrix

    Returns:
        error: Angular difference in degrees
    """
    R_student = T_student[:3, :3]
    R_gt = T_gt[:3, :3]

    # Compute relative rotation: R_diff = R_student^T * R_gt
    R_diff = R_student.T @ R_gt

    # Rotation angle from trace: angle = arccos((trace(R) - 1) / 2)
    trace = np.trace(R_diff)

    # Clamp to valid range for arccos
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Compute angle in degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def grade_solution(T_student, T_gt):
    """
    Grade the student's solution by comparing transformation matrices.

    Scoring uses global configuration:
    - Translation error: 50% weight
    - Rotation error: 50% weight

    The score decreases linearly with error based on MAX_TRANSLATION_ERROR
    and MAX_ROTATION_ERROR thresholds.

    Args:
        T_student: 4x4 student transformation matrix
        T_gt: 4x4 ground truth transformation matrix

    Returns:
        score: Total score out of TOTAL_POINTS
    """
    # Compute errors
    trans_error = compute_translation_error(T_student, T_gt)
    rot_error = compute_rotation_error(T_student, T_gt)

    # Translation score
    trans_score = max(0, TRANSLATION_POINTS * (1 - trans_error / MAX_TRANSLATION_ERROR))

    # Rotation score
    rot_score = max(0, ROTATION_POINTS * (1 - rot_error / MAX_ROTATION_ERROR))

    # Total score
    total_score = trans_score + rot_score

    print("\n" + "=" * 50)
    print("GRADING RESULTS")
    print("=" * 50)
    print(f"Translation error: {trans_error:.4f} m")
    print(f"  Score: {trans_score:.1f}/{TRANSLATION_POINTS}")
    print(f"\nRotation error: {rot_error:.2f}°")
    print(f"  Score: {rot_score:.1f}/{ROTATION_POINTS}")
    print(f"\nTotal Score: {total_score:.1f}/{TOTAL_POINTS}")
    print("=" * 50)

    return total_score


# ============================================================================
# Main Function (PROVIDED - DO NOT MODIFY)
# ============================================================================


def main():
    """
    Main function to run the calibration homework.

    This function:
    1. Loads test data (point pairs, camera info, ground truth)
    2. Calls your implemented functions
    3. Grades your solution
    """
    print("=" * 50)
    print("LiDAR-Camera Calibration Homework: PnP Exercise")
    print("=" * 50)

    # Step 0: Load data
    print("\nLoading test data...")
    object_points, image_points_distorted = load_point_pairs("point_pairs.csv")
    camera_matrix, dist_coeffs = load_camera_info("camera_info.yaml")
    gt_transform = load_ground_truth("ground_truth_transform.json")

    print(f"Loaded {len(object_points)} point correspondences")
    print(f"Camera matrix K:\n{camera_matrix}")
    print(f"Distortion coefficients: {dist_coeffs}")

    # Step 1: Undistort image points
    print("\n" + "-" * 50)
    print("Step 1: Undistorting image points...")
    print("-" * 50)
    try:
        image_points_undistorted = undistort_points(
            image_points_distorted, camera_matrix, dist_coeffs
        )
        print(f"Undistorted {len(image_points_undistorted)} points")
        print(f"Sample undistorted point: {image_points_undistorted[0]}")
    except NotImplementedError as e:
        print(f"ERROR: {e}")
        return

    # Step 2: Solve PnP
    print("\n" + "-" * 50)
    print("Step 2: Solving PnP problem...")
    print("-" * 50)
    try:
        success, rvec, tvec = solve_pnp(
            object_points, image_points_undistorted, camera_matrix
        )

        if not success:
            print("ERROR: PnP solver failed!")
            return

        print("PnP solved successfully!")
        print(f"Rotation vector (axis-angle): {rvec.flatten()}")
        print(f"Translation vector: {tvec.flatten()}")

    except NotImplementedError as e:
        print(f"ERROR: {e}")
        return

    # Step 3: Create transformation matrix
    print("\n" + "-" * 50)
    print("Step 3: Creating 4x4 transformation matrix...")
    print("-" * 50)
    try:
        transform_matrix = create_transform_matrix(rvec, tvec)
        print(f"Transformation matrix:\n{transform_matrix}")

    except NotImplementedError as e:
        print(f"ERROR: {e}")
        return

    # Grade the solution
    grade_solution(transform_matrix, gt_transform)

    print("\nHomework complete! Submit your calibration_homework.py file.")


if __name__ == "__main__":
    main()
