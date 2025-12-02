This paper introduces Sim2Fruit, a ROS2 package that integrates plant segmentation algorithms with reinforcement learning agents to train robotic arm systems for autonomous fruit harvesting through sim-to-real transfer learning. The system addresses the persistent challenge of enabling accurate autonomous plant harvesting by combining 3D point cloud processing with RL-based control.

## Problem Statement

Current agricultural robotic solutions face three critical limitations:

1. Vision systems falter due to insufficient light variability, data occlusion, and real-time processing constraints
2. Manipulators and end-effectors struggle to adapt to diverse crop geometries
3. Lack of domain generalization across different orchard layouts, fruit geometries, and sensing conditions

Additionally, no existing literature fully incorporates RL and vision-based end-effector control within the same integrated environment, primarily due to challenges in real-time system-level coordination and synchronization.

## Technical Architecture

**ROS2-Based Modular System:** The Sim2Fruit package comprises six fundamental nodes orchestrating the complete pipeline:

1. **Gazebo Simulator Node**: Hosts the virtual environment and publishes simulated sensor data
2. **Plants Node**: Provides service calls to load plant models (from Pheno4D dataset of tomato plants) into Gazebo
3. **Robot State Node**: Publishes joint states of the robotic arm from URDF specifications (MyCobot Robotic Arm 320 M5)
4. **Vision Model Node**: Processes simulated sensor data or raw PCD and publishes segmented point clouds
5. **Environment Node**: Subscribes to segmented PCD and robot joint states, providing observation states for RL training
6. **Robot Controller Node**: Translates RL agent decisions into arm control commands and updates joint states

## Vision System Implementation

**3D Deep Learning Frameworks:**

- Integrates pre-trained PointNet and PointNet++ architectures for plant organelle segmentation
- Processes raw point cloud data to identify plant anatomy (stem, leaves, fruit)
- Supports configuration of additional pre-trained vision models
- Provides automated encoding of plant state and health metrics
- Informs reward schema by enabling positive reinforcement for fruit-directed actions

**Data Acquisition:** Point cloud data obtained through either simulated sensors in Gazebo via sensor plugins or explicitly defined plant models, with each point receiving predicted segmentation labels.

## Reinforcement Learning Framework

**Current Implementation (Proof-of-Concept):**

- Policy network with randomly initialized weights serving as a toy policy
- Action space includes: fruit picking (extending and grasping), plant pruning, pest control, and idle state
- Utilizes off-the-shelf single-agent RL algorithms from RLlib, specifically Proximal Policy Optimization (PPO)
- Feedback loop enables iterative learning through observation-action-reward cycles

**Training Algorithm:** The proposed single-agent model-free RL algorithm follows these steps:

1. Initialize neural network parameters and replay buffer
2. Reset Sim2Fruit environment with randomized plant/fruit positions
3. Obtain initial observation state (joint states, fruit pose)
4. Sample actions from policy network with action-masking for invalid moves
5. Execute actions in simulation, receiving rewards and observing next state
6. Store transitions in replay buffer
7. Update policy parameters via chosen MFRL update rule
8. Continue until convergence, advancing time by episode length

**Expected Learning Outcomes:** Understanding joint states to properly execute picking, pruning, pest control operations, or maintain idle state based on observation inputs.

## Methodology and Data Pipeline

**Simulation Environment:**

- Gazebo robotics simulator provides physics-accurate virtual environment
- Plant models from Pheno4D dataset offer clear structures for segmentation and manipulation
- URDF specifications ensure accurate representation of robotic arm dynamics
- Simulated sensors provide real-time feedback for training

**Information Flow:**

1. Robotic arm and plant models housed in Gazebo environment
2. Simulated sensors provide vision model with PCD
3. Vision model segments plant organelle
4. Segmented PCD fed as observation state into RL agent
5. RL agent selects action based on policy network
6. Action executed in simulation
7. Process repeats with updated observation
8. After training convergence, learned weights deployed to physical robotic arm

## Future Research Directions

**RL Algorithm Development:**

- Comparison between model-free RL (MFRL) and model-based RL (MBRL) approaches
- MBRL implementation using "world models" to generate synthetic training data
- Evaluation of sample efficiency and convergence rates across different algorithms
- Integration of LSTM architecture for temporal context from past observations
- Implementation of action masking to prevent infeasible actions

**Enhanced Vision Integration:**

- Using PointNet to inform and estimate approach poses for grasping
- Encoding contextual knowledge such as estimated arm poses into RL observations
- Integration of additional sensor modalities beyond RGB and depth

**Simulation Realism:**

- Integration with GazeboPlants plugin for realistic plant motion modeling
- Development of configurable plant models with variable characteristics
- Domain randomization through modified plant parameters (height, mass, structure, spacing)
- Environment randomization including lighting conditions, wind speeds, and obstacles

**Generalization Improvements:**

- Domain randomization proven to enable more robust, generalizable RL solutions
- Development of diverse simulation scenarios to improve sim-to-real transfer
- Addressing complications in data collection identified in vision model research

## Expected Performance Metrics

While not yet fully implemented with converging policies, the system is designed to evaluate:

- Harvest success rate across different harvesting conditions
- Task completion time for picking operations
- Accuracy of fruit segmentation by vision models
- Minimization of plant damage during harvesting
- Generalization across diverse environmental conditions

## Technical Contributions

1. **Integrated Pipeline**: First comprehensive ROS2 package combining simulation, vision models, and RL for agricultural robotics
2. **Real-time Coordination**: Addresses synchronization challenges between vision processing and robotic control
3. **Digital Twin Framework**: Enables safe, cost-effective testing of decision-making algorithms
4. **Modular Architecture**: Facilitates integration of diverse vision models and RL algorithms
5. **Sim-to-Real Transfer**: Provides pathway from simulated training to physical deployment
6. **Open-Source Implementation**: Code available at github.com/ronydahdal/sim2fruit

## Challenges and Considerations

**Current Limitations:**

- Toy policy network with random weights serves only as proof-of-concept
- No integrated reward model logic yet implemented
- Limited plant model diversity in current implementation
- Requires development of comprehensive evaluation metrics

**Integration Challenges:**

- Real-time system-level coordination complexity
- Synchronization between multiple ROS2 nodes
- Accurate URDF modeling for realistic simulation
- Computational requirements for training convergence

## Significance

Sim2Fruit represents an initial effort toward bridging the gap between vision-based perception and RL-based control in agricultural robotics. By providing an integrated testbed for training harvesting policies in simulation, the system enables:

1. Safe exploration of control strategies without risk to physical equipment or crops
2. Rapid iteration and testing of different RL algorithms and vision models
3. Scalable training through simulated environments generating extensive data samples
4. Foundation for future research in model-based RL for precision agriculture
5. Reduction in real-world data collection requirements through sim-to-real transfer

The modular ROS2 architecture ensures extensibility for future enhancements while maintaining compatibility with existing robotics infrastructure. This work establishes a framework for advancing autonomous agricultural systems through the integration of deep learning-based perception and reinforcement learning-based control.