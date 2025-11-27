## Overview

This research develops a deep learning-powered robotic solution to address labor shortages in strawberry greenhouse farms by creating a digital twin environment for automated harvesting operations.

## Key Contributions

### 1. Digital Twin Development

- Created a complete digital twin model of the SILAL strawberry greenhouse farm in Abu Dhabi, UAE
- Implemented in ROS-Gazebo simulation environment
- Includes realistic 3D models of greenhouse structure, strawberry plants, and environmental conditions

### 2. MARTA Robot Platform

- **MARTA** (Mobile Autonomous Robot with Telescopic Arm)
- Differential drive mobile base with telescopic arm mechanism
- Simpler to control than articulated 6-DOF manipulators
- Equipped with:
    - Intel RealSense D435i RGB-D camera
    - 9-DOF IMU
    - RPLidar A1 for obstacle detection
    - Two-finger gripper

### 3. YOLOv9-GLEAN Detection Model

- Novel architecture combining YOLOv9 with GLEAN (Generalized Efficient Layer Aggregation Networks)
- Super-resolution capabilities for enhanced detection accuracy
- Trained on hybrid dataset:
    - 650 real images from SILAL greenhouse
    - 216 synthetic CAD-generated images
    - Total: 1,158 images (ripe and unripe strawberries)

### 4. Performance Results

- **Precision:** 0.996
- **Recall:** 0.991
- **mAP@50-95:** 0.99
- **Model Size:** 212 MB
- **FPS:** 38

Outperforms state-of-the-art models:

- YOLOv8m-seg: 0.983 precision, 0.980 recall
- YOLOv5m-seg: 0.981 precision, 0.979 recall
- LS-YOLOv8: 0.987 precision, 0.984 recall

### 5. Visual Servoing and Grasping

- ROS-MoveIt integration for motion planning
- Hand-to-eye camera calibration using Aruco markers
- TracIK inverse kinematics solver
- OMPL (Open Motion Planning Library) for collision-free path generation
- Automatic scanning and grasping workflow

## Technical Architecture

### Simulation Framework

- **Environment:** ROS-Gazebo
- **Robot Description:** URDF (Universal Robot Description Format)
- **Physics Engine:** ODE (Open Dynamics Engine)
- **Control System:** ros_control with PID controllers
- **Motion Planning:** MoveIt with OMPL

### Detection Pipeline

1. RGB-D camera captures images during scanning
2. YOLOv9-GLEAN processes images for strawberry detection
3. Depth estimation from RGB-D camera
4. Pose estimation for detected strawberries
5. Motion planning generates collision-free trajectory
6. Gripper approaches and grasps target strawberry

### Network Architecture

- **Backbone:** RepNCSPELAN blocks
- **Neck:** SPPELAN (Spatial Pyramid Pooling Efficient Layer Aggregation)
- **Head:** Three detection/segmentation blocks for multi-scale objects
- **Input:** 640×640×3 images
- **Activation:** Sigmoid function
- **Loss:** Categorical Cross-Entropy

## Key Innovations

1. **Programmable Gradient Information (PGI):** Minimizes data loss during deep network transmission
2. **Hybrid Dataset Training:** Combines real and synthetic images for robust detection
3. **Digital Twin Validation:** Tests algorithms in virtual environment before real-world deployment
4. **Telescopic Arm Design:** Simpler control compared to articulated arms while maintaining effective reach

## Methodology

### Dataset Augmentation

- Rotation, flipping, and blurring techniques
- Roboflow labeling tool for annotation
- Addresses lighting variations in greenhouse environment

### Camera Calibration

- Pinhole camera model
- Hand-to-eye calibration: AX = XB problem
- Aruco marker-based calibration in Gazebo

### Strawberry Searching Algorithm

1. Robot positioned at home position
2. Gripper-mounted camera scans 500mm vertically
3. Detection of ripe strawberries during scan
4. Selection based on confidence value or minimum path
5. Grasping action performed
6. Return to home position and repeat

## Hardware Configuration

**Training Platform:**

- CPU: Intel Core i9-14900KF @ 3.20 GHz
- RAM: 128 GB
- GPU: NVIDIA GeForce RTX 3090
- Storage: 2 TB HDD
- OS: Ubuntu
- Framework: PyTorch 2.1.2+cu118

**Training Parameters:**

- Epochs: 300
- Batch size: 32
- Initial learning rate: 0.001
- Learning rate momentum: 0.90
- Weight decay: 0.0005

## Applications and Benefits

- Addresses labor shortages in agriculture
- Enables 24/7 operation in controlled environments
- Reduces harvest timing errors
- Improves fruit quality through consistent harvesting
- Provides objective ripeness assessment
- Enables offline programming and validation

## Future Work

- Real-world implementation and validation
- Multi-robot coordination for enhanced productivity
- Explainable AI (XAI) integration for model transparency
- Data-driven decision making approaches
- Human-robot collaboration exploration
- Extended application to other crops

## Conclusion

This research successfully demonstrates the feasibility of automated strawberry harvesting using digital twin technology, advanced deep learning detection, and robotic manipulation. The YOLOv9-GLEAN model achieves state-of-the-art performance, and the integrated system provides a complete framework for developing and testing agricultural robotics solutions in simulation before real-world deployment.