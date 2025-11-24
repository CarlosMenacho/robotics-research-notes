
# Abstract 
- sim2real transfer, particulary for manipulation policies relying on RGB images, remains a critical challenge
- There's a significant domain shift (syntetic -> real-world) visual data

# Introduction
- Sim2Real is able to transfer of control policies learned in sim envs to real-world
- Notably, all of these methods rely on perception modalities like depth, tactile sensing, or point cloud inputs, having a low discrepancy  with sim and real
- **modalities that can be simulated well, can be transfered well**
- RGB is rarely used in robotics learning applications
- RGB images are easy to acquire in real-world environments with cameras and align closely with human perception 
- There's a difference between image-sim and real image. "vision Sim2Real an out-of-domain generalization problem"

# Related works
- Simulators, are very inportant for scaling up robot learning due to several advantages: parallelization, cost, time efficiency, and safety.
- Previous attempts in this domain have primary focused on navigation
- Gaussian Splating is the state-of-the-art rendering technique that models scenes using 3D Gaussian primitives 
- offer photorealistic representation of complex geometries
- NeRF and its derivatives (pointclouds) structure of Gaussian Splats enables easier manipulation

# Preliminary

## 1. Rigid body transformations in Gaussian Splatting

- High-quality rendering is needed for getting translation and rotations
- Each 3D Gaussian can be transformed using Homogeneus transformation matrix
$$
\begin{equation}
\begin{split}
\mu' &= R\mu + t \\
\Sigma' &= R\Sigma R^T
\end{split}
\end{equation}
$$
- applying these transformations updates the position and orientation of 3D Gaussian Splatting

# Method

- if each rigid body in the Gaussian Splat representation can be segmented, and the transformation in simulator is identified, then it is feasible to render rigid body in novel poses
- Generating rigid bodies, interacting with robot in sim we can generate photorealistic renderings
- Instead using mesh primitives, utilize Gaussian Splats

## Problem statement 
- $S_{real}$ as te Gaussian Splat of the real world scene using RGB
- $S_{obj}^k$ as the Splat of the $k-$th object in the scjene
- Its goal is to use $S_{real}$ for generate photorealistic renderings $I^{sim}$ 
- Then collect demonstration using the expert $\epsilon$ for training RGB-based images policies
- $\epsilon$ generates $\tau_\epsilon$ trajectories consisting of state-action pairs $\{ (s_1, a_1), ..., (s_T, a_T)\}$ for a full episode
- State is defined by $s_t = (q_t, x_t^1, ... , x_t^n)$ where $q_t \in \mathbb{R}^3$  denotes the robot's joint angles
- $s_t^k = (p_t^k, R_t^k)$  represents the position and orientation of the $k-$th object in the scene
- $a_t= (p_t^e, R_t^e)$ refers to the end effector's position and orientation
- The policy relies solely on real-world RGB images $I^{real}$ at test time
## Definitions of Coordinate Frames and Transformations

- $\mathcal{F}_{real}$ Real-world coordinate frame and primary reference frame
- $\mathcal{F}_{sim}$ simulator coordinate frame and real world robot frame $\mathcal{F}_{robot}$ are aligned with $\mathcal{F}_{real}$ 
- $\mathcal{F}_{splat}$ splat coordinate frame. Represents the frame of the base of the robot in the Gaussian Splat of the scene $\mathcal{S}_{real}$ 
- $T_{\mathcal{F}_{robot}}^{\mathcal{F}_{splat}}$ different frame of the robot's base
## Robot Splat Models

1. Alignment of Gaussian Splat robot Frame to the Simulator Frame
	- manually segment out the 3D Gaussians associated with the robot 
2. Segmentation of the Robot Links
	- To associate 3D Gaussians with the respective links in $\mathcal{S}_{real}$ , ground truth bboxes of the robot's links
	- This method allows us to isolate the 3D Gaussians corresponding to each link in the real- world scene 
3. Forward Kinematics transformation
	- They uses forward kinematics from PyBullet

## Object Splat Models

- Given the position $p_t^k \in s_t$  and orientation $R_t^k \in s_t$ can be used to calculate the transformation of object $T_{fk}^{k-obj}$ from its original simulator frame

--- 
- Uses KNN to classify based on simulator point cloud gripper, 

## Policy training and Deployment

- They employ Diffusion Policy (state-of-the-art for behavior cloning)
- It mitigates the vision Sim2Real gap discrepancies
	- simulated scenes lack shadows
	- rigid body assumptions like cables 
	- Image augmentation are used to address these above issues