
This paper addresses a critical challenge in agricultural automation: detecting grape stems under occlusion conditions for robotic harvesting. With global grape production exceeding 75 million tons annually, automating harvest operations is essential for addressing labor shortages. Unlike fruits such as apples or strawberries, grapes require precise stem cutting for successful harvesting, making accurate stem detection under complex occlusion conditions a fundamental technical requirement.

## Problem Statement

The authors identify several key challenges in grape harvesting robotics:

1. **Complex occlusion patterns**: Grape bunches exhibit irregular positioning with multi-fruit clustering, variable shapes, and disordered branch growth
2. **Dynamic environments**: Natural orchards present unpredictable occlusion states from leaves and obstacles
3. **Limitations of existing approaches**: Current detection methods struggle with severe occlusion scenarios
4. **Inadequacy of reconstruction methods**: These depend heavily on initial viewpoint data quality
5. **Manual parameter dependency**: Existing viewpoint planning methods rely on manually designed information gain metrics

## Methodology

### System Architecture

The proposed framework consists of three integrated components:

#### 1. Instance Segmentation (Mask R-CNN)

- Extracts 2D masks of grape bunches from RGB images
- Trained on a specialized dataset with three occlusion categories:
    - General occlusion (1-35%)
    - Severe occlusion (35-80%)
    - Complete occlusion (>80%)
- Dataset: 3,012 images collected from Sunshine Rose grapes in V-shaped trellis systems
- Performance: 96.1% precision at IOU threshold 0.5; 83.2% at IOU threshold 0.75

#### 2. Volumetric Mapping

- **Voxel downsampling**: Reduces computational load while preserving spatial information
- **Depth-based targeting**: Uses depth information to specifically focus on target grapes
- **Movement constraints**: Limits robot arm activity space based on world coordinates:

```
  Xgw - 0.1 ≤ xr ≤ Xgw + 0.1
  Ygw - 0.3 ≤ yr ≤ Ygw + 0.3
  Zgw - 0.2 ≤ zr ≤ Zgw + 0.2
```

- **Octree structure**: Transforms grape point cloud data into hierarchical octree format using OctoMap
- **Occupancy categorization**: Distinguishes between known occupancy (pink), newly discovered occupancy (red), and free space (white)

#### 3. DDPG-Based Viewpoint Planning

**Algorithm Architecture:**

- **Actor network**: Outputs deterministic continuous actions μ(s|θμ)
- **Critic network**: Evaluates actions based on expected reward Q(s,a|θQ)
- **Target networks**: Use soft updates with rate τ = 0.05
- **Experience replay buffer**: Size M = 5,000 for storing past experiences

**State Space (8-dimensional):**

```
st = [xt, yt, zt, φt, θt, ψt, Oroi, Onew]
```

- Camera pose: position (x, y, z) and orientation (φ, θ, ψ)
- Spatial occupancy: observed ROI voxels (Oroi) and newly discovered ROI voxels (Onew)

**Action Space (4-dimensional):**

```
at = [xt, yt, zt, ψt]
```

- Continuous control of camera position and yaw angle
- Movement constrained to [-0.03, 0.03] meters
- Joint angles limited to [-60°, 60°]

**Novel Reward Function:**

```
R = α*Oroi + β*Onew - φ*t
```

The reward function incorporates three weighted components:

- **α (information gain weight = 0.1)**: Encourages selecting high-visibility regions
- **β (movement cost weight = 10)**: Promotes exploration of areas with new perspectives (weighted heavily to encourage discovery)
- **φ (step penalty weight = 0.1)**: Discourages inefficient paths and excessive wandering

This formulation drives the network to balance information maximization with movement efficiency.

**Network Updates:**

Critic network minimization:

```
minLθQ = (1/Nmini) Σ(yi - Q(si, ai|θQ))²
where yi = ri + γQ'(si+1, μ'(si+1|θμ')|θQ')
```

Actor network gradient ascent:

```
∇θμJ ≈ (1/Nmini) Σ ∇aQ(s,a|θQ)|s=si,a=μ(si) ∇θμμ(s|θμ)|si
```

### Training Configuration

**Hyperparameters:**

- Episodes: 800 (300 additional beyond initial 500 for sufficient training)
- Maximum steps per episode: 100
- Mini-batch size: 32
- Learning rates: 0.001 (both Actor and Critic)
- Discount factor γ: 0.9
- Experience replay buffer: 5,000

**Training Environment:**

- ROS-based Gazebo simulation
- 3D grape models created in SolidWorks, textured in Blender
- Image size: 640×480 pixels
- Training epochs: 300 for Mask R-CNN

**Training Progression:**

- Initial 200 steps: Exploration phase with significant reward fluctuations (negative region)
- Continued training: Decreased fluctuations, rewards shifted to positive range
- By 500 steps: Effective strategy learning with stable positive rewards
- Final 300 episodes: Consolidation and refinement

## Experimental Validation

### Hardware Platform

**Robotic System:**

- **Robotic arm**: Aubo-i5 (6 DOF)
- **Camera**: Intel RealSense L515 (RGB-D)
- **Gripper**: Robotiq 2F-85 adaptive gripper
- **Mobile platform**: AgileX-Bunker
- **Computing**: Industrial PC with Intel i7-1165G7 and RTX 2060 GPU
- **Training computer**: RTX 3080 GPU (12 GB VRAM)

**Software Stack:**

- Ubuntu 18.04 with ROS (Robot Operating System)
- Moveit framework for motion planning
- Eye-in-Hand configuration with calibrated transformation matrices

**System Calibration:**

- Hand-Eye calibration using fixed calibration plate
- D-H parameter-based forward kinematics for Aubo-i5
- Coordinate transformation: BP = BTE × ETC × CP

### Testing Environments

#### Laboratory Environment

- Simulated natural grape growth with wooden fence
- Randomly placed branches, leaves, and plastic grapes
- Controlled testing conditions
- Sample size: 20 samples per occlusion category (60 total)

#### Vineyard Environment (Real-world)

- Location: Shangguo Orchard, Panyu District, Guangzhou, China
- Grape variety: Sunshine Rose
- Average bunch height: 130 cm
- Aisle width: 110 cm
- V-shaped trellis system
- Sample size: 22 general, 18 severe, 16 complete occlusion (56 total)

### Performance Metrics

**Success Rate (α):**

```
α = (NS/NA) × 100%
```

where NS = successfully detected samples, NA = total samples

**Success Criteria (path cost thresholds):**

- General occlusion: ≤ 0.312 m
- Severe occlusion: ≤ 0.519 m
- Complete occlusion: ≤ 0.779 m

**Average Iterations (NT):**

```
NT = Σ(Ni)/NA
```

where Ni = viewpoint adjustments for i-th target

**Path Cost Calculation:** Euclidean distance between viewpoints:

```
d(P,Q) = √[(q1-p1)² + (q2-p2)² + (q3-p3)²]
```

## Results

### Laboratory Environment Performance

**Detection Success Rates:**

|Method|General|Severe|Complete|
|---|---|---|---|
|Zaenker et al.|80%|50%|15%|
|Zeng et al.|85%|60%|40%|
|**Proposed**|**90%**|**80%**|**55%**|

**Average Number of Iterations:**

|Method|General|Severe|Complete|
|---|---|---|---|
|Zaenker et al.|8.5|20.6|34.7|
|Zeng et al.|8.3|18.6|30.6|
|**Proposed**|**4.5**|**8.2**|**15.4**|

### Vineyard Environment Performance

**Detection Success Rates:**

|Method|General|Severe|Complete|
|---|---|---|---|
|Zaenker et al.|72.73%|61.11%|18.75%|
|Zeng et al.|77.27%|66.67%|37.5%|
|**Proposed**|**86.36%**|**77.78%**|**56.25%**|

**Average Number of Iterations:**

|Method|General|Severe|Complete|
|---|---|---|---|
|Zaenker et al.|9.5|20.3|33.9|
|Zeng et al.|8.9|17.6|29.7|
|**Proposed**|**4.6**|**8.3**|**15.2**|

### Key Observations

1. **Consistent superiority**: The proposed method outperforms comparison methods across all occlusion levels in both environments
2. **Path cost efficiency**: The proposed method consistently shows the lowest path costs, with stacked area charts demonstrating clear advantages as occlusion severity increases
3. **Vineyard vs. laboratory**: Counterintuitively, vineyard environments required slightly less effort due to larger gaps between grape leaves and fruits
4. **Continuous action advantage**: The continuous action space design significantly reduces computational burden compared to discrete action methods
5. **Adaptive exploration**: The method effectively adjusts viewpoints toward information-rich areas, progressively minimizing occlusion impacts

## Technical Innovations

### 1. Fully End-to-End Framework

- **Direct learning**: System learns optimal viewpoints through environmental interaction
- **No manual design**: Eliminates need for manually designed information gain metrics
- **Unified process**: Integrates perception and action into single learning framework

### 2. Novel Reward Function Design

- **Differential weighting**: Assigns different occupancy weights to explored vs. newly discovered regions
- **Information gain integration**: Incorporates spatial occupancy dynamics
- **Step penalty**: Prevents excessive wandering and promotes efficient exploration

### 3. Continuous Action Space

- **Smooth control**: Enables fine-grained camera adjustments
- **Reduced complexity**: Avoids discrete action space limitations
- **Enhanced efficiency**: Fewer steps required compared to discrete methods

### 4. Octree-Based Environmental Modeling

- **Hierarchical representation**: Efficient storage and retrieval of spatial information
- **Dynamic updates**: Real-time incorporation of newly discovered regions
- **Categorized occupancy**: Distinguishes known, new, and free space

## Comparative Analysis

### Advantages Over Zaenker et al. (2021)

- **Computational complexity**: Zaenker's method requires frequent ROI sampling mode switching
- **Occlusion handling**: Struggles with multi-layered occlusions
- **Coverage gaps**: Some heavily occluded areas remain undetected
- **Path efficiency**: Higher path costs across all occlusion levels

### Advantages Over Zeng et al. (2022)

- **Prior knowledge dependency**: Zeng's method requires prior environmental models
- **Hidden ROI identification**: Limited ability to identify all occluded regions
- **Discrete actions**: Less smooth and efficient than continuous control
- **Adaptability**: Lower performance in completely occluded scenarios

### Unique Contributions

1. **No prior knowledge required**: Operates effectively without environmental pre-mapping
2. **Dynamic adaptability**: Handles unpredictable occlusion patterns
3. **Robustness**: Maintains >50% success rate even under complete occlusion
4. **Efficiency**: Requires 2-3× fewer iterations than comparison methods

## Limitations and Future Directions

### Current Limitations

1. **Extreme occlusion challenges**
    - Increased uncertainty in movement decisions
    - Higher task complexity requiring extensive trial-and-error
    - Multiple occlusion layers create ambiguous information landscapes
2. **Perception limitations**
    - Visual data alone insufficient under severe occlusion
    - Difficulty identifying stem posture when completely obscured
    - Some grape bunches (e.g., short stems with multi-level occlusion) remain undetectable
3. **Robotic arm singularities**
    - DDPG algorithm occasionally leads to motion failures
    - Workspace constraints at extreme poses
    - Limited redundancy in movement options
4. **Environmental sensitivity**
    - Performance degradation under low lighting conditions
    - Challenges with dense occlusions
    - Variation in detection quality with background complexity

### Proposed Future Research Directions

1. **Multi-modal sensing integration**
    - Combine visual and tactile information
    - Incorporate force/torque feedback for occluded stem detection
    - Develop sensor fusion frameworks for enhanced perception accuracy
2. **Spatial information optimization**
    - Integrate 3D spatial reasoning into movement decisions
    - Develop predictive models for occlusion patterns
    - Optimize exploration strategies using geometric constraints
3. **Multi-agent collaboration**
    - Coordinate mobile platform and robotic arm movements
    - Distribute exploration tasks across multiple agents
    - Improve workspace coverage and reduce singularity issues
4. **Environmental adaptation**
    - Implement lighting compensation techniques
    - Develop robust feature extraction under varying conditions
    - Optimize parameters for different vineyard configurations
5. **Advanced learning strategies**
    - Explore curriculum learning approaches
    - Investigate meta-learning for rapid adaptation
    - Develop hierarchical reinforcement learning for complex decision-making

## Implications for Agricultural Robotics

### Theoretical Contributions

1. **Active perception paradigm**: Demonstrates effectiveness of DRL-based viewpoint planning for agricultural applications
2. **End-to-end learning**: Shows feasibility of eliminating manual parameter tuning in complex agricultural environments
3. **Information-theoretic rewards**: Validates information gain-based reward formulation for exploration tasks

### Practical Applications

1. **Grape harvesting automation**: Directly applicable to commercial vineyard operations
2. **Generalizability**: Framework potentially adaptable to other fruit crops requiring stem cutting (kiwifruit, tomatoes, cucumbers)
3. **Robotic autonomy**: Advances toward fully autonomous agricultural robots operating in unstructured environments

### Industry Impact

1. **Labor shortage mitigation**: Provides technological solution to agricultural workforce challenges
2. **Harvest efficiency**: Reduces time and cost associated with manual harvesting
3. **Quality consistency**: Enables standardized harvesting procedures
4. **Scalability**: Framework designed for deployment across multiple robotic platforms

## Conclusion

This research presents a novel, fully end-to-end active visual perception framework for detecting occluded grape stems using deep reinforcement learning. The key innovation lies in eliminating manual parameter design through direct environmental interaction and learning, while maintaining high detection success rates (>55% even under complete occlusion) and exceptional efficiency (requiring 2-3× fewer iterations than state-of-the-art methods).

The DDPG-based viewpoint planning algorithm, combined with intelligent volumetric mapping and a novel information gain-based reward function, enables robotic systems to autonomously navigate complex occlusion scenarios. Comprehensive validation in both controlled laboratory settings and real-world vineyards demonstrates the method's robustness and practical applicability.

This work provides foundational support for the next generation of highly autonomous agricultural robots capable of adaptive operation in complex, unstructured environments, representing a significant advancement toward fully automated grape harvesting systems.