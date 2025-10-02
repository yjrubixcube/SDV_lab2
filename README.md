# LiDAR-Camera Calibration: PnP Exercise

## Overview

This homework helps you understand the complete LiDAR-camera extrinsic calibration process using the Perspective-n-Point (PnP) algorithm. You will work with real data extracted from the LCTK calibration system.

## Background

### The Calibration Problem

LiDAR-camera calibration finds the transformation between two sensors:
- **LiDAR** detects the calibration board in 3D space (X, Y, Z coordinates)
- **Camera** detects ArUco markers on the board in 2D image space (pixel coordinates)

The goal is to find the rotation **R** and translation **t** that relate these coordinate systems.

### The PnP Algorithm

Given:
- N ≥ 4 known 3D points in world coordinates (object points)
- Corresponding 2D projections in image coordinates (image points)
- Camera intrinsic parameters (focal length, principal point, distortion)

Solve for camera pose (R, t) using the equation:
```
s * p = K * (R * P + t)
```

Where:
- **P**: 3D world point
- **p**: 2D image point (homogeneous coordinates)
- **K**: camera intrinsic matrix
- **R**: rotation matrix
- **t**: translation vector
- **s**: scale factor

### Camera Distortion

Real cameras have lens distortion (radial and tangential). Before solving PnP, we must:
1. Undistort the 2D image points using camera distortion coefficients
2. Use undistorted points for PnP solving

## Files Provided

1. **`calibration_homework.py`** - Template file with TODOs (you edit this)
2. **`point_pairs.csv`** - 3D-2D point correspondences from real calibration
3. **`camera_info.yaml`** - Camera intrinsic matrix and distortion coefficients
4. **`ground_truth_transform.json`** - Reference transformation for grading

## Tasks

You need to implement three functions in `solver.py`:

### TODO 1: Implement `undistort_points()`

Undistort 2D image points using camera intrinsics and distortion coefficients.

**Hint**: Use `cv2.undistortPoints()` with the camera matrix and distortion coefficients.

**Resources**:
- [OpenCV undistortPoints documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga55c716492470bfe86b0ee9bf3a1f0f7e)

### TODO 2: Implement `solve_pnp()`

Solve the PnP problem to find camera pose (rotation vector and translation vector).

**Hint**: Use `cv2.solvePnP()` with:
- 3D object points (already in world coordinates)
- 2D undistorted image points
- Camera intrinsic matrix
- Zero distortion (since points are already undistorted)
- Method: `cv2.SOLVEPNP_ITERATIVE`

**Resources**:
- [OpenCV solvePnP documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)

### TODO 3: Implement `create_transform_matrix()`

Create a 4x4 homogeneous transformation matrix from rotation vector and translation vector.

**Matrix Format**:
```
[R11  R12  R13  tx]
[R21  R22  R23  ty]
[R31  R32  R33  tz]
[ 0    0    0    1]
```

Where the top-left 3×3 block is the rotation matrix R, and the top-right 3×1 block is the translation vector t.

**Hint**:
1. Convert rotation vector to rotation matrix using `cv2.Rodrigues()`
2. Create a 4×4 identity matrix
3. Set the top-left 3×3 block to the rotation matrix
4. Set the top-right 3×1 block to the translation vector

## Running Your Code

```bash
python3 solve.py
```

Expected output:
```
Loading test data...
Loaded 16 point correspondences
Camera matrix K:
[[1164.62    0.    950.12]
 [   0.   1161.10  538.55]
 [   0.      0.      1.  ]]

Step 1: Undistorting image points...
Undistorted 16 points

Step 2: Solving PnP...
PnP solved successfully!
Rotation vector: [ 0.123 -0.456  0.789]
Translation: [ 1.234 -0.567  2.890]

Step 3: Creating 4x4 transformation matrix...
Transformation matrix:
[[ 0.0089 -0.9997  0.0247  0.125]
 [ 0.7071  0.0212  0.7069 -0.053]
 [-0.7071 -0.0125  0.7069  0.087]
 [ 0.0000  0.0000  0.0000  1.000]]

==================================================
GRADING RESULTS
==================================================
Translation error: 0.0023 m
  Score: 49.4/50

Rotation error: 0.12°
  Score: 49.4/50

Total Score: 98.8/100
==================================================
```

## Grading Criteria

Your solution will be automatically graded based on transformation matrix accuracy:

**Scoring (0-100 points)**:
- **Translation error: 50% weight**
  - Linear scale: 0m error = 50 points, 0.20m error = 0 points
  - Formula: `score = max(0, 50 * (1 - error/0.20))`

- **Rotation error: 50% weight**
  - Linear scale: 0° error = 50 points, 10° error = 0 points
  - Formula: `score = max(0, 50 * (1 - error/10.0))`

**Total Score** = Translation Score + Rotation Score
