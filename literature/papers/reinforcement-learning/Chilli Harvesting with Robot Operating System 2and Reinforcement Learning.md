This paper presents an autonomous robotic system for red chilli harvesting that integrates Robot Operating System 2 (ROS 2) framework with Reinforcement Learning (RL) agents to address labor-intensive and inefficient traditional harvesting methods.

## Problem Statement

Traditional chilli harvesting faces several challenges: high labor costs, operational delays, inconsistent quality due to unskilled workers, and crop damage that reduces market value. The authors identify a gap in agricultural automation specifically for red chilli harvesting requiring precise detection, plucking, and handling capabilities.

## System Architecture

**Hardware Components:**

- Six-degree-of-freedom robotic arm with high-torque servo motors
- Soft-material gripper designed for fragile crop handling
- RGB and depth cameras for visual perception
- Mobile base for field navigation
- Embedded computing platform (NVIDIA Jetson)
- Soil monitoring sensors (moisture, pH, nutrients)

**Software Framework:**

- ROS 2 modular architecture enabling component integration
- Deep Q-Network (DQN) implementation for RL agent
- Computer vision algorithms (OpenCV) for image processing
- SLAM for localization and mapping
- TensorFlow/PyTorch for neural network training

## Methodology

**Vision System:** The system employs RGB cameras for color-based ripeness detection and depth cameras for 3D spatial mapping. Image processing pipeline includes noise filtering, segmentation algorithms, and contour detection. Sensor fusion combines RGB and depth data to reduce false positives and ensure targets are within the manipulator's workspace.

**Reinforcement Learning Agent:** Trained in Gazebo simulation environment with varying field layouts, plant heights, and lighting conditions. The state space incorporates camera data, chilli positioning, ripeness assessment, and arm orientation. The reward structure incentivizes successful plucking (+10), precise cutting (+5), and efficient placement (+3), while penalizing crop damage (-10), misidentification (-5), and missed targets (-7).

**Control System:** Implements PID control for arm positioning with real-time adjustments based on sensor feedback. A dedicated obstacle avoidance node uses depth camera data integrated into local cost maps. The controller coordinates all subsystems including wheel rotation, arm movement, gripper actuation, and camera control.

**Fleet Management:** Master-slave architecture where a central node coordinates multiple robots through task auctioning and dynamic reallocation. Robots share status information (position, battery, workload) via ROS 2 topics. The system employs Teb Local Planner and RVO algorithms for decentralized multi-robot coordination and collision avoidance.

## Results

**Detection Performance:**

- Average detection accuracy: 95% for ripe chilli identification
- Harvesting success rate: >90% without crop damage
- System effectively distinguished between ripe/unripe chillies and non-target objects

**Operational Efficiency:**

- Fleet management reduced overall harvesting time through efficient work distribution
- RL agent demonstrated adaptability to varying environmental conditions (lighting, soil quality, terrain)
- Real-time decision-making enabled dynamic response to field changes

**System Validation:** Testing conducted in Gazebo simulation with various environmental scenarios and small-scale physical prototype. The system successfully navigated shortest paths, avoided obstacles, and operated under different lighting conditions.

## Technical Contributions

1. Integration of DQN-based RL with ROS 2 for agricultural manipulation tasks
2. Multi-sensor fusion approach combining RGB and depth data for robust detection
3. Scalable fleet management architecture for multi-robot coordination
4. Soft gripper design optimized for delicate crop handling
5. Incorporation of soil monitoring for informed decision-making

## Limitations and Future Work

The authors acknowledge challenges in:

- Adapting the RL model to realistic field conditions
- Ensuring sensor precision across varying environments
- Optimizing end-effector performance
- Efficiently managing fleet coordination and soil condition monitoring

Future directions include:

- Extending system capabilities to diverse crop types
- Integrating advanced sensors (thermal cameras) for enhanced crop health assessment
- Using sensor data for predictive scheduling of harvesting operations
- Improving generalization through expanded training datasets

## Significance

This work advances agricultural automation by demonstrating a practical integration of modern robotics technologies (ROS 2, deep RL, computer vision) for precision agriculture. The modular architecture and simulation-based validation approach provide a framework for developing autonomous systems for labor-intensive agricultural tasks, contributing to sustainable farming practices and operational efficiency improvements.