This research compares different visual and spatial observation representations for reinforcement learning policies in real-world robotic manipulation, specifically focusing on vacuum gripper tasks. The study evaluates 2D RGB images, depth images, and 3D voxel grid representations for the challenging task of picking up parcels of various sizes, shapes, and weights.

## Key Contributions

1. **Comprehensive comparison** of visual (2D images) versus spatial (depth images and 3D voxel grids) observations in real-world RL
2. **Real-world validation** of spatial encoders on an actual robotic system (previous work only in simulation)
3. **Novel application** of voxel-based representations for vacuum gripper control using sample-efficient RL

## Problem Setting: Vacuum Gripping

**Task:** "Last-inch manipulation" - final phase of grasping boxes with a vacuum gripper

**Challenges:**

- Vacuum grippers require precise positioning on smooth, flat surfaces
- Failure occurs easily if object is deformed or surface is not smooth enough
- Need adaptive local feedback policy for inaccurate estimates or damaged/occluded boxes
- Difficult to simulate accurately due to complex deformable interactions

**Setup:**

- UR5e robot arm with Robotiq EPick vacuum gripper
- Two Intel RealSense D405 wrist-mounted cameras
- 7D action space: position delta (3D), rotation delta (3D), suction control (1D)
- 27D base observation space: pose, velocity, forces, torques, gripper info, previous action

## Methodology

### Framework: SERL (Sample-Efficient Robotic Reinforcement Learning)

Built on RLPD (Reinforcement Learning with Prior Data) algorithm, which extends SAC (Soft Actor-Critic) with:

- High update-to-data ratio during training
- Symmetric sampling of prior and on-policy data
- Layer-norm regularization
- Data augmentation via DRQ (Data-Regularized Q)

### Visual Representations Compared

**1. RGB Images (Baseline)**

- Pre-trained ResNet18 backbone with Spatial Softmax pooling
- 128x128x3 resolution
- Single wrist camera view
- Fixed pre-trained weights during training

**2. Depth Images**

- Simple Conv2D-based encoder
- 128x128x1 resolution
- Maximum distance capped at 20cm (local surroundings only)
- Two wrist cameras used simultaneously
- Only 1% of trainable weights compared to ResNet18

**3. Voxel Grid (Novel Approach)**

- 3D convolutional architecture inspired by VoxNet
- Resolution: 50x50x40 voxels (2mm voxel size)
- Coverage: 10x10x8 cm³ around suction cup
- Generated from combined point clouds of both cameras
- Always relative to end-effector (normalized representation)
- Two variants: trained from scratch or using pre-trained VoxNet weights

### Key Technical Innovations

**Observation Space Symmetries:**

- Exploited 4-fold rotation symmetry around z-axis
- Transformed all observations into first/fifth octant
- Simplified learning by eliminating redundant quadrant learning
- Actions and observations always in consistent reference frame

**Relative Observation/Action Frame:**

- All observations relative to starting position
- Transformations applied to forces, torques, and spatial data
- Enables better generalization across different initial conditions

**Reward Function:**

$$
R(s) = R_{goal} - R_{step} - R_{pose} - R_{action} + R_{suction}
$$

- R_goal = 100 when box picked and 1cm above start
- Penalties for: deviation from start, large actions, each step, unnecessary suction activation

## Experimental Setup

### Training Environment

- 10 different boxes with varying:
    - Colors, sizes, rigidity, surface textures
    - Obstructions (zip-ties, holes, deformations)
- Predetermined positions with slight randomization
- 20 expert demonstrations for bootstrapping
- Maximum 100 steps (10 seconds) per episode
- Training time: 46-85 minutes on single RTX 3080 GPU

### Evaluation Setup

**Seen Boxes (30 trials per policy):**

- Same 10 boxes used during training
- Seeded initial poses for fair comparison

**Unseen Boxes (30 trials per policy):**

- 5 novel boxes with different:
    - Colors and sizes
    - Orientations (including tilted)
    - Obstructions (zip-tie + hole)
    - Initial angles (simulating wrong estimates)

**Metrics:**

1. Success rate (binary: 0 or 1)
2. Cumulative reward
3. Time to completion

## Results

### Performance on Seen Boxes

|Policy|Success Rate|Reward|Time (s)|
|---|---|---|---|
|Behavior Tree|86.7%|47.6|4.8|
|Behavioral Cloning|80%|-3.5|5.03|
|SAC (no vision)|73.3%|36.9|5.59|
|DRQ RGB|96.7%|58.5|4.08|
|DRQ Depth|**100%**|83.8|3.01|
|DRQ Voxel (scratch)|**100%**|86.7|2.95|
|DRQ Voxel (pretrained)|**100%**|**90.8**|**2.74**|
|DRQ Voxel (pretrained + symmetry)|96.7%|83.9|2.84|

### Performance on Unseen Boxes

|Policy|Success Rate|Reward|Time (s)|
|---|---|---|---|
|Behavior Tree|86.7%|47.6|5.32|
|Behavioral Cloning|60%|-24.6|7.22|
|SAC (no vision)|50%|-21.2|7.48|
|DRQ RGB|80%|-22.3|5.28|
|DRQ Depth|86.7%|-25.8|5.32|
|DRQ Voxel (scratch)|66.7%|11.1|5.74|
|DRQ Voxel (pretrained)|93.3%|55.4|3.88|
|**DRQ Voxel (pretrained + symmetry)**|**96.7%**|**65.3**|4.34|

### Key Findings

**Spatial > Visual:**

- Voxel-based policies significantly outperformed RGB images on unseen boxes (96.7% vs 80%)
- Depth images performed well on seen boxes (100%) but struggled on unseen (86.7%)

**Pre-training Matters:**

- Pre-trained VoxNet weights crucial for generalization
- Improved success rate from 66.7% → 93.3% on unseen boxes
- Prevented undertrained backbone issues

**Symmetry Exploitation:**

- Further boosted performance when combined with pre-trained voxel encoder
- Success rate: 93.3% → 96.7% on unseen boxes
- Simplified learning by eliminating redundant spatial configurations

**Speed Benefits:**

- Voxel-based policies consistently faster (~3s) than visual (~4s) and baselines (~5s)
- Best performing policy achieved 96.7% success in 4.34s average

## Ablation Studies

### ResNet Architecture

- ResNet18 outperformed ResNet10 significantly (66.7% vs 96.7% on seen boxes)
- Likely due to more heterogeneous objects requiring richer feature representations

### Voxel Grid Only (Minimal Proprioception)

- Policy with only voxel grid + 2D gripper info (no full 27D state) achieved:
    - 93.3% on seen boxes
    - **100% on unseen boxes** (surpassing full-state version!)
- Demonstrates spatial encoder captures most necessary information
- Suggests proprioceptive states may be partially redundant

### Temporal Ensembling

- Applied action smoothing post-training: `a_t = [a^s_t, a^s_{t-1}, a^s_{t-2}, a^s_{t-3}] × [0.5, 0.3, 0.2, 0.1]^T`
- Reduced action jitter by ~50% (smoothness metric: 2.46 → 1.19)
- Minimal performance impact (96.7% success maintained)
- Important for motor longevity in real systems

## Technical Implementation Details

### Controller

- Two-layer control hierarchy:
    1. RL policy at 10 Hz providing target poses
    2. Impedance controller at 100 Hz: `F = k_p · e + k_d · ė`
    3. Safety bounds: |e| ≤ Δ to prevent collisions

### Modified Rodrigues Parameters

- Used instead of Euler angles for orientation representation
- Singularity-free rotation representation
- Rotation-axis independent

### Voxel Grid Generation

- Combined point clouds from both cameras using Open3D multiway registration
- Voxel occupied if ≥1 point inside
- Always relative to end-effector (auto-normalized)
- 3D shift augmentation: pad by 3 voxels, random crop back to original size

## Limitations and Future Work

**Current Limitations:**

- Single GPU training (RTX 3080) limited to one RGB image
- Two simultaneous images did not converge
- Focuses on last-inch manipulation (assumes rough object pose estimate)
- Single vacuum gripper (not multiple suction cups)

**Future Directions:**

- Test PointNet or Point Transformer architectures instead of VoxNet
- Apply to dual-arm manipulation and cooperative handovers
- Explore residual RL approaches building on behavioral cloning
- Additional 3D augmentation methods (occlusion, noise injection)
- Integration with perception systems for unstructured environments

## Significance

**First Work To:**

- Train real-world RL policy for vacuum gripper using 3D point cloud inputs
- Demonstrate superior generalization of spatial over visual representations in real robotics
- Show voxel grids can capture sufficient information with minimal proprioception

**Practical Impact:**

- Sample-efficient approach suitable for real-world deployment (< 90 min training)
- Addresses critical industrial automation need (vacuum grippers widely used)
- Demonstrates sim-to-real gap necessitates real-world training for deformable contact tasks

**Key Insight:** Spatial 3D representations (voxel grids) enable significantly better generalization than 2D visual representations for manipulation tasks requiring precise contact reasoning, especially when combined with pre-trained encoders and symmetry exploitation.