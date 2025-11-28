This research tackles the challenging problem of shape control for Deformable Linear Objects (DLOs) using offline goal-conditioned reinforcement learning. The work compares the learned approach with classical shape-servoing methods on two materials with different properties: a soft rope and an elastic cord.

## Problem Statement

**Task:** Control the planar shape of a DLO by manipulating both endpoints with a dual-arm robot

**Key Challenges:**

1. Middle part of DLO not directly controlled by end-effectors
2. Target shapes may require moving away from goal to invert curvature (e.g., concave to convex)
3. Multiple DLO shapes possible with same gripper poses (history-dependent behavior)
4. Surface friction affects dynamics
5. Material properties significantly impact deformation behavior

**Previous Approaches:**

- **Shape-servoing (Jacobian-based):** Use local models and instantaneous error; struggle with complex, long-term dynamics
- **Deep Learning methods:** Often require extensive online data collection or suffer from sim-to-real gap

## Methodology

### System Architecture (Modular Approach)

Unlike end-to-end learning, the system separates concerns:

**1. DLO Tracking (ARAP-based)**

- RGB-D camera (Intel RealSense D435) provides top-view
- HSV filtering for color-based segmentation
- As-Rigid-As-Possible (ARAP) deformation model with 3×3×M lattice
- End-effector poses used to constrain lattice ends
- Output: N=18 evenly spaced points along DLO (state representation)

**2. Low-Level Controller (HQP)**

- Task-space control (3 DOF per arm: xy translation + z rotation)
- Hierarchical Quadratic Programming for inverse kinematics
- Constraint hierarchy:
    1. Joint velocity limits
    2. Joint position limits
    3. Elbow proximity (collision avoidance)
    4. Desired motion
    5. Joint position potential (safe configuration)
- Cubic interpolation generates end-effector trajectories
- Controller runs at 50 Hz; policy updates at 2 Hz

**3. RL Policy (Goal-Conditioned)**

- Input: Concatenated current and goal DLO shapes + end-effector poses (78D)
- Output: Desired end-effector poses (6D actions)
- Reward: Negative RMSE between current and goal shapes

### Offline Goal-Conditioned RL

**Algorithm: TD3+BC (Twin Delayed DDPG + Behavior Cloning)**

Modified policy update with BC regularization:

$$
π = argmax [λQ(s,π(s,g),g) - (π(s,g) - a)²]
$$

Where:

- λ is adaptive weighting factor based on Q-values
- α hyperparameter controls BC strength (larger α = less BC impact)
- Prevents value overestimation of unseen state-action pairs

**Network Architecture:**

- D2RL architecture (state concatenated to each hidden layer input)
- 4 hidden layers × 256 neurons
- ReLU activations
- Batch normalization + 0.5 dropout
- Training: 1M environment steps, batch size 256
- Discount factor γ = 0.95

### Data Collection and Augmentation

**Collection Procedure (3 hours per DLO):**

- 1010 episodes total (1000 training, 10 test)
- Random sampling from safe workspace (separate regions per arm)
- 0.3 probability: move left, right, or both end-effectors
- 0.1 probability: execute curvature inversion sequence
- Recorded at 20 Hz, downsampled to 10 Hz
- Average episode: 10.3 ± 3.4 seconds

**Safe Workspace Definition:**

- x: [0.3, 0.6] m
- y_left: [0.1, 0.3] m
- y_right: [-0.3, -0.1] m
- θ: [-π/4, π/4] rad
- Prevents collisions, entanglements, and keeps DLO in camera FOV

**Data Augmentation (Inspired by HER):**

Three goal-sampling strategies tested:

1. **Intra:** Goals from intermediate shapes in same episode (shorter episodes)
2. **Inter:** Goals from future episodes (longer episodes)
3. **Mixed:** Goals from intermediate shapes in future episodes

Augmentation ratios: 1x (baseline), 2x, 4x, 6x, 8x

**Best performing:** Intra strategy - creates goals close to starting state (near-future goals more valuable)

## Experimental Setup

**Hardware:**

- ABB YuMi dual-arm robot
- Intel RealSense D435 camera (fixed top-view)
- Custom 3D-printed fingers for DLO attachment

**Test DLOs:**

- (a) Soft rope (red): 1 cm diameter, 55 cm length
- (b) Elastic cord (yellow): 1 cm diameter, 55 cm length, wrapped in plastic sleeve

**Test Protocol:**

- 8 goal shapes per test sequence
- Balanced: 2 straight, 3 convex, 3 concave
- Initial state: DLO stretched along x=0.5m line
- Duration: 40 seconds without reset
- Same test used for both baseline and RL policies

**Baseline:** Berenson's shape-servoing method (2013) - Jacobian-based approach with diminishing rigidity value k=1

## Results

### Augmentation Experiment (Soft DLO)

Goal sampling strategy comparison:

- **Intra:** 0.0381 ± 0.0253 m (best)
- **Mixed:** 0.0466 ± 0.0318 m
- **Inter:** 0.0494 ± 0.0314 m

Augmentation ratio effects (with α=2.5):

- 1x (baseline): ~0.09 m
- 2x: ~0.06 m
- **4x: 0.0227 ± 0.0086 m**
- 6x: 0.0236 ± 0.0082 m
- 8x: 0.0251 ± 0.0102 m

**Finding:** Benefits plateau after 4x augmentation

### BC Regularization Experiment (8x Augmented Soft DLO)

Effect of α parameter:

- α = 1.0: ~0.09 m
- α = 1.5: ~0.07 m
- α = 2.0: ~0.04 m
- α = 2.5: ~0.025 m
- **α = 3.0: 0.0225 ± 0.0093 m (best)**

**Finding:** Larger α (less BC impact) improves results

- Baseline comparison: 0.0439 ± 0.0394 m

### Material Properties Comparison

|Method|Soft Rope (a)|Elastic Cord (b)|
|---|---|---|
|**Baseline**|0.044 ± 0.039 m|0.044 ± 0.045 m|
||[0.012, 0.097]|[0.007, 0.103]|
|**RL Policy**|0.023 ± 0.009 m|**0.015 ± 0.004 m**|
||[0.008, 0.035]|[0.010, 0.023]|

**Key Findings:**

- RL approach improved over baseline for both materials
- **Better performance on elastic cord** - simpler dynamics due to higher stiffness
- Baseline showed no material-dependent difference
- **Successful curvature inversion** - main advantage over shape-servoing

### Curvature Inversion Success

The RL policy successfully inverted DLO curvature (concave → convex), while the shape-servoing baseline got stuck at local minima. This demonstrates the advantage of learning long-term manipulation strategies that may initially increase error before decreasing it.

**Visual Results (8 Test Shapes):**

- RL policy closely matched goal shapes across all tests
- Baseline struggled particularly with shapes requiring curvature inversion
- Material properties visible: elastic DLO follows end-effector angles more closely in straight shapes

## Technical Contributions

1. **First real-world offline GCRL implementation** for DLO shape control (no simulation)
2. **Modular architecture** separating perception, control, and learning
3. **Data-efficient augmentation** reducing experimental data requirements
4. **Material-agnostic approach** validated on soft and elastic objects
5. **Curvature inversion capability** beyond classical methods

## Limitations and Future Work

**Current Limitations:**

- Struggles with consecutive very different shapes
- Test sequence order affects performance
- Trained/tested once per configuration (no variance analysis)
- Limited to predetermined test sequence

**Future Directions:**

- Better state/action representations
- Impact of low-level control frequency analysis
- Different RL algorithms exploration
- Hyperparameter optimization
- More extensive testing with multiple trials
- Adaptive test sequences
- Extension to 3D manipulation
- Different end-effector designs

## Key Insights

**Why Offline RL?**

- Avoids unsafe online exploration with real robot
- Sim-to-real gap problematic for deformable object dynamics
- Sample-efficient compared to online methods

**Why Goal-Conditioning?**

- Shape control naturally multi-goal task
- Enables generalization to unseen shapes
- Data augmentation creates additional training signal

**Why Modular Architecture?**

- DLO tracking is challenging independent problem
- Task-space control more intuitive than joint-space
- Separates concerns for easier debugging and improvement

**Success Factors:**

1. ARAP-based tracking provides reliable state representation
2. Intra-episode goal augmentation increases goal coverage
3. Larger α (weaker BC) allows better Q-value optimization
4. Material stiffness affects learning difficulty (elastic easier)

## Practical Impact

**Demonstrates:**

- Feasibility of real-world RL for deformable object manipulation
- Importance of data augmentation in offline GCRL
- Potential to surpass classical methods on complex manipulation tasks
- Applicability across different material properties

**Applications:**

- Cable/wire manipulation in manufacturing
- Medical suturing and catheter manipulation
- Food processing (dough, noodles)
- Agricultural tasks (plant manipulation)