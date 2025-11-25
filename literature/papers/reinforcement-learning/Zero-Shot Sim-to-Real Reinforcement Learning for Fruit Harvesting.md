---

---

- Sim2Real pipeline for autonomous strawberry picking
- leverages custom Mujoco  simulation that integrates domain randomization techniques
- Low-level control with high-level perception and decision making
- Contributions:
	- Fruit Gym, strawberries picking tasks
	- Domain randomization techniques 
	- RL training pipeline employing Dormant Ratio Minimization DRM

---
- Agricultural tasks such as fruit harvesting require robots to operate under these variable conditions 
- Commercial robots achieve 90% of success in strawberry harvesting, some issues: dense clusters, occluded fruit, failed grasping and missed detections.
- Since we aim to enable fruit picking in the field, it is possible that training in one location with this approach would not generalize to other locations. 
- The authors uses Mujoco Gymnasium simulation environment using domain randomization

---
- Discrepancy between sim data and real data can produce poor performance on physical robots
- Randomization help training to learn across the diverse conditions the robot could encounter on real farms
- RL is an emerging area in agricultural robotics
- end-to-end sim2real RL has yet to be explored

--- 
# Simulation Environment

- Sim2Real pipeline comprises three main components: 
	- Simulation environment 
	- low-level control module based on Cartesian impedance
	- high-level policy learning framework
- Environment: based on Mujoco physic engine 
	- Action Space: 7-dimensional vector representing the change and grasp commands
	- Observation Space: includes end-effector pose, velocity, gripper position and gripper state vector. Also includes two wrist cameras to provide RGB images
	- Domain Randomization, lightning (pos, intensity, and headlight properties), camera parameters (pos and orientation noise) and object position
- Low-level Control via impedance Controller
	- This architecture has two control layers is employed: high-level RL policy at low frequency ~ 20Hz, with real-time impedance controller tracking actions at 1KHz
$$
F= K_p \cdot e + K_d \cdot \dot{e} + F_{ff} + F_{cor} 
	$$
- Controller's objective formula
	- where $e=p-p_{ref}$ is the error between the measured pose p and the reference pose $p_{ref}$ 
	- $F_{ff}$ is the feed forward force
	- $F_{cor}$ accounts for Coriolis effects

--- 
# Policy Learning

- Dormant Ratio Minimization DRM is employed 
- DRM builds upon DrQv2. And mitigates the common issue of network inactivity
	- Dormant-Ratio-Guided Perturbation, pediodically perturbing the network
	- Awaken Exploration Scheduler, Adapting the exploration noise dynamically when dormant ratio is hight
	- Dormant-Ratio-Guided Exploration, Adjusting the value target by incorporating a dormant-ratio-dependent parameter

## Reward function
- Grasp reward
- End effector-target Proximity reward
- Displacement penalty, moving the fruit away from their initial position
- Energy penalty, taking large action
- Smoothness penalty: changing direction quickly

# Real robot setup

-  Franka Panda robot using ROS2 implementation of the Cartesian impedance controller

# Experiments

- Authors compare DRM with DrQv2 baseline for 2 million timesteps
- Success rate was measured to evaluate its policy in 180 trials (sim and real envs)