
---

## Executive Summary

This comprehensive survey provides the first systematic review of diffusion models (DMs) applied to robotic manipulation, covering grasp learning, trajectory planning, and data augmentation. Diffusion models have emerged as highly promising approaches in robotics due to their exceptional ability to model multi-modal distributions and their robustness to high-dimensional input and output spaces, surpassing traditional methods like Gaussian Mixture Models (GMMs) and Energy-based Models (EBMs).

---

## Table of Contents

- [Introduction](#introduction)
- [Mathematical Framework](#mathematical-framework)
- [Network Architectures](#network-architectures)
- [Applications](#applications)
  - [Trajectory Generation](#trajectory-generation)
  - [Robotic Grasp Generation](#robotic-grasp-generation)
  - [Visual Data Augmentation](#visual-data-augmentation)
- [Experiments and Benchmarks](#experiments-and-benchmarks)
- [Challenges and Limitations](#challenges-and-limitations)
- [Future Directions](#future-directions)
- [Conclusion](#conclusion)

---

## Introduction

### Why Diffusion Models for Robotics?

Diffusion models have demonstrated remarkable success in visual domains and have recently emerged as a transformative approach in robotics, particularly for manipulation tasks. Their key advantages include:

1. **Multi-modal Distribution Modeling**: Ability to capture multiple valid solutions (e.g., multiple feasible trajectories or grasp poses)
2. **High-Dimensional Robustness**: Effective handling of complex visual and action spaces
3. **Training Stability**: More stable than GANs, avoiding mode collapse
4. **Smooth Trajectories**: Naturally generate smooth, physically plausible motion

### Key Challenges in Traditional Approaches

- **GMMs**: Limited capacity for complex distributions, sensitive to hyperparameters
- **GANs**: Training instability, mode collapse, hyperparameter sensitivity
- **EBMs/IBC**: May bias toward specific modes despite theoretical multi-modal capability

---

## Mathematical Framework

### Two Main Approaches

#### 1. Score-Based Diffusion Models (SMLD)

**Forward Process:**
- Add noise progressively according to variance schedule $\{σ_k\}^K_{k=1}$
- Perturb data from $p_{data}(x)$ toward N(0,I)

**Reverse Process:**
- Learn score function $\nabla_x log p_{σ_k}(x_k|x)$ using Noise Conditional Score Network (NCSN)
- Apply Langevin dynamics iteratively to denoise samples

**Training Objective:**
$$
L = \frac{1}{2K} \sum_{k=1}^K σ_k^2 E_{p_{data}(x)} E_{x_k \sim p} (x_k|x)[||∇_{x_k}  p_{σ_k}(x_k|x) - s_\theta(x_k,σ_k)||_2^2]
$$

#### 2. Denoising Diffusion Probabilistic Models (DDPM)

**Forward Process:**
- Markovian noise addition: $p(x_{k+1}|x_k) = N(x_k; /sqrt{(1-\beta_k)}x_k, \beta_k I)$
- Closed-form noise addition: $p(x_{k+1}|x_0) = N(x_k; \sqrt{\tilde{\alpha}_k} x_0, (1-\tilde{\alpha}_k)I)$

**Reverse Process:**
- Model $p_θ(x_{k-1}|x_k)$ as Gaussian distribution
- Predict mean using learned noise: $x_{k-1} = \frac{1}{\sqrt{\alpha_k}}(x_k - \frac{1-α_k}{\sqrt{1-\tilde{\alpha}_k}} \epsilon_θ(x_k,k)) + \sigma_k z$

**Training Objective:**
$$
L = E_{k,x_0,\epsilon}[||\varepsilon - \varepsilon_\theta(x_k,k)||_2^2]
$$

### Architectural Improvements

**Key Innovations for Speed and Quality:**

The forward diffusion process can be formulated as a stochastic differential equation (SDE)
SDE can be replaced by an ordinary differential equation (ODE).

1. **DDIM (Denoising Diffusion Implicit Models)**
   - Deterministic sampling process
   - Decouples training and inference steps
   - Enables 10-100 sampling steps vs. 1000+ in DDPM

2. **DPM-Solver**
   - Second-order ODE solver
   - Non-uniform step sizes
   - Reduced truncation error

3. **Flow Matching**
   - Learns velocity field directly
   - Simpler training objective
   - More numerically stable
   - No noise schedule required

4. **Self-Consistency in Flow Matching**
   - Single-step sampling capability
   - Maximizes consistency across step sizes
   - Minimal performance drop with fewer steps

### Adaptations for Robotic Manipulation

**Key Requirements:**

1. **Conditioning on Observations**
   - Visual observations (RGB, RGB-D, point clouds)
   - Robot proprioception
   - Natural language instructions
   - Implemented via FiLM (Feature-wise Linear Modulation) or cross-attention

2. **Temporal Correlation**
   - Trajectories have temporal dependencies (vs. spatial for images)
   - **Receding Horizon Control**: Generate sub-trajectories iteratively
   - Planning horizon H, control horizon H_c ≤ H
   - Execute H_c steps, replan with updated observations

---

## Network Architectures

### 1. Convolutional Neural Networks (U-Net)

**Temporal U-Net** (adapted from image generation):
- Replace 2D spatial convolutions with 1D temporal convolutions
- Most frequently used architecture
- Sample efficient, generalizes well with small datasets

**Diffusion Policy (DP) Architecture:**
- Conditions on observation history via FiLM
- Generates action trajectories (not joint state-action like Diffuser)
- Extension to multiple modalities via feature concatenation

**Advantages:**
- Sample efficiency
- Lower hyperparameter sensitivity than transformers
- Good for high-frequency trajectories

**Disadvantages:**
- May cause over-smoothing
- Less effective for velocity control

### 2. Transformers

**Architecture:**
- Observations, denoising timestep, and actions as input tokens
- Multi-head cross-attention for conditioning
- Integration via self-attention and cross-attention

**Common Variants:**
- Multi-head cross-attention transformer
- Diffusion Transformers (DiT)
- Custom architectures (e.g., transformer output → MLP → noise prediction)

**Advantages:**
- Captures long-range dependencies
- Robust for high-dimensional data
- Superior for long-horizon tasks

**Disadvantages:**
- High computational cost
- Longer inference time
- More complex to train

### 3. Multi-Layer Perceptrons (MLPs)

**Architecture:**
- Concatenated inputs (observations, actions, timestep)
- 2-4 hidden layers typical
- Mish activation common

**Advantages:**
- Computational efficiency
- Training stability
- Fast inference

**Disadvantages:**
- Limited capacity for complex problems
- Poor performance with high-dimensional visual input
- Requires pre-processing for images (CNN encoder)

**Primary Use:** Reinforcement learning applications with low-dimensional state

### Comparison Summary

| Architecture | Power | Efficiency | Best For |
|-------------|-------|------------|----------|
| **U-Net** | Medium | High | Vision-based tasks, high-frequency control |
| **Transformer** | High | Low | Long-horizon, high-dimensional, complex reasoning |
| **MLP** | Low | Very High | RL with state info, real-time constraints |

### Sampling Steps

**Common Configurations:**
- **Training:** 50-100 noise levels
- **Inference (DDIM):** 5-10 steps (sometimes 3-4 or up to 30)
- **Trade-off:** Sample quality vs. inference time

---

## Applications

### Trajectory Generation

Trajectory planning is the most dominant application of DMs in robotic manipulation, primarily through imitation learning, with growing integration into reinforcement learning.

#### Imitation Learning Approaches

##### 1. Actions and Pose Representations

**Task Space (Most Common):**
- End-effector positions (translation + rotation)
- Representations: Euler angles, quaternions
- Examples: DP, 3D-DP, DP-VLA

**Joint Space:**
- Direct joint angle control
- Reduces singularities
- Less common
- Examples: Pearce et al., Saha et al.

**SE(3) Representations:**
- Lie group structure for continuous interpolation
- Physically grounded transformations
- More complex training
- Examples: Xian et al., Liu et al., Ryu et al.

**Receding Horizon Control:**
- Generate sub-trajectories of length H
- Execute H_c ≤ H steps
- Replan with updated observations
- Balances reactivity and coherence

##### 2. Visual Data Modalities

**2D Visual Observations:**
- RGB images most common
- Feature encodings (CLIP, DINOv2)
- Examples: DP, SI et al., Li et al.

**3D Visual Observations:**
- Point clouds (direct or via depth estimation)
- Better geometric understanding
- Robust to lighting variations
- Superior performance on complex tasks
- Examples: 3D-DP, 3D Diffuser Actor, Li et al.

**Single-view vs. Multi-view:**
- Multi-view: More complete scene information, requires more hardware
- Single-view: Simpler setup, may need reconstruction

##### 3. Trajectory Planning as Image Generation

**Approach:**
- Predict sequence of images showing robot/object motion
- Leverage internet-scale video training data
- Examples: Ko et al., Zhou et al., Du et al.

**Advantages:**
- Access to massive video datasets
- No need for action-space mapping
- Embodiment-agnostic potential

**Challenges:**
- High-dimensional prediction
- Temporal consistency
- Physical plausibility
- Requires extensive compute (e.g., 100 V100 GPUs, 70k demos)

**Variants:**
- Point tracking (Bharadhwaj et al.)
- Image-space action effects (Vosylius et al.)

##### 4. Long-Horizon and Multi-Task Learning

**Hierarchical Approaches:**

**Skill-Based:**
- Single skill-conditioned DM or multiple DMs per skill
- Higher-level planner (VAE, regression model)
- Examples: Mishra et al., Kim et al., Xu et al.

**Coarse-to-Fine:**
- High-level policy predicts goal states
- Low-level policy executes to goals
- No predefined skill enumeration
- Examples: Zhang et al., Ma et al., Xian et al.

**Multi-Modal Inputs:**
- Videos, images, language instructions
- Tactile, point clouds
- Enables versatile skill chaining
- Examples: Liang et al., Wang et al.

**Lifelong Learning:**
- Largely unexplored with DMs
- Examples: Huang et al. (continuous update), Di Palo et al. (lifelong buffer)
- Challenges: Catastrophic forgetting, computational efficiency

##### 5. Vision-Language-Action (VLA) Models

**Motivation:**
- VLAs: Strong generalization from internet-scale pretraining
- Limitations: Slow inference, imprecise actions from discretization

**Integration with Diffusion:**
- **Refinement Approach**: VLA predicts coarse action → DM refines
- **Unified Approach**: DM as action decoder for VLA backbone
- Examples: TinyVLA, Pan et al., Shentu et al., Team et al.

**Flow Matching Alternative:**
- Direct velocity field learning
- Faster inference, more stable
- Examples: π0, Black et al., Zhang & Gienger

**Advantages:**
- Leverages VLA generalization
- Diffusion provides action precision
- Handles multi-modal action distributions

**Challenges:**
- Combined slow inference (VLA + DM)
- Requires efficient sampling strategies

##### 6. Constrained Planning

**Types of Constraints:**
- Obstacle avoidance
- Goal-oriented planning
- Object-centric manipulation
- Safety constraints (e.g., surgical robotics)

**Methods:**

**Inpainting (Janner et al.):**
- Replace specific states after each denoising step
- Simple but limited to point-wise constraints
- May cause instability

**Classifier Guidance:**
- Train separate score model
- Add gradient to denoising process
- Examples: Mishra et al., Liang et al., Carvalho et al.
- Disadvantages: Additional training, computational cost

**Classifier-Free Guidance:**
- Train conditional and unconditional DMs in parallel
- Weighted mixture during sampling
- No new model for new constraints
- Examples: Ho et al., Saha et al., Li et al., Power et al.

**Constraint Tightening:**
- Guarantee satisfaction via tightening in reverse process
- Römer et al.
- Limited evaluation to date

**Affordance-Based:**
- Optimize trajectories based on affordances
- Primarily in grasp learning (Section 4.2)

---

#### Reinforcement Learning Approaches

##### Integration with Offline RL

**Diffuser (Janner et al.):**
- Classifier-based guidance with return prediction model R_φ(τ_k)
- Sampling: $p(τ_{k-1}|τ_k,O) ≈ N(τ_{k-1}; μ + Σ∇R_φ(μ), Σ)$
- Reward-independent DM training (like imitation learning)
- Examples: Suh et al., Liang et al.

**Decision Diffuser (Ajay et al.):**
- Direct conditioning on return via classifier-free guidance
- Improves over Diffuser but limited generalization
- More like guided imitation learning

**Diffusion Q-Learning (Wang et al.):**
- Integrate Q-function critic into DM training
- Policy improvement: L_RL = L + αL_c where L_c = -E[Q_φ(s,a^0)]
- Better generalization than return conditioning
- Examples: Ada et al., Kim et al., Venkatraman et al.

##### Advantages of Offline RL with DMs
- Sample efficiency
- No real-time data collection
- Lower computational cost
- Better for suboptimal diverse data

##### Limitations
- Requires reward-labeled data
- More prone to overfitting than imitation
- Unable to react to distribution shifts
- Most methods use ground-truth state (not visual)

##### Skill Composition
- Combine skill learning with RL
- Examples: Ajay et al., Kim et al., Venkatraman et al.

##### Online and Offline-to-Online RL
- Very limited research
- Examples: Ding & Jin, Ren et al., Huang et al.

---

### Robotic Grasp Generation

Grasp learning is a crucial skill for manipulation, and diffusion models provide unique advantages for generating diverse, feasible grasp poses.

#### Methodological Approaches

##### 1. Diffusion on SE(3) Grasp Poses

**Challenge:**
- Standard diffusion formulated in Euclidean space
- SE(3) is non-Euclidean manifold
- Gaussian noise fails to preserve rotation/translation constraints

**Solutions:**

**SE(3)-Diff (Urain et al.):**
- Energy-based model (EBM) with smooth cost function
- Score matching on Lie group
- Direct grasp quality evaluation
- Disadvantage: Extensive sampling, generalization challenges

**Flow Matching Approaches:**

**EquiGraspFlow (Lim et al.):**
- Continuous normalizing flows (CNFs) as ODE solvers
- Learn angular (SO(3)) and linear (R(3)) velocities
- Preserves SE(3)-equivariance
- No auxiliary supervision (e.g., SDF) needed

**Grasp Diffusion Network (Carvalho et al.):**
- Similar CNF-based approach
- Competitive performance without SDF

**SE(3) Bi-Equivariance:**
- Critical property: transformations in input space → consistent output space
- Ensures spatial/geometric relationship invariance

**Equivariant Descriptor Field (EDF) - Ryu et al.:**
- Bi-equivariance in Lie group representation
- Improved sample efficiency for pick-and-place
- Extension: Bi-equivariant score matching for diffusion

**Multi-Embodiment (Freiberg et al.):**
- Adapts Ryu et al. for multi-gripper generalization
- Equivariant encoder for gripper embeddings

**Parallel Jaw vs. Dexterous Grasping:**
- Most methods focus on parallel jaw grippers
- Growing work on dexterous manipulation
- Examples: Wu et al., Weng et al., Wang et al.

##### 2. Diffusion in Latent Space

**GraspLDM (Barad et al.):**
- VAE-based latent diffusion
- Model grasp distribution in latent space
- Conditioned on point cloud and task latent
- No explicit SE(3) constraint
- May limit physical plausibility

##### 3. Functional Approaches

**Language-Guided Grasp Diffusion:**
- Natural language shapes generation process
- Task-oriented grasping
- Examples: Nguyen et al., Vuong et al., Chang & Sun

**Affordance-Driven Diffusion:**
- Object pose diffusion for rearrangement (Liu et al., Zhao et al.)
- Affordance-guided reorientation (Mishra & Chen)
- Imitation learning for pre-grasp (Wu et al., Ma et al.)

**Hand-Object Interaction (HOI) Synthesis:**
- Model hand's adaptive responses to object shapes
- Realistic, functional interactions with dexterity
- Examples: Ye et al., Wang et al., Zhang et al., Cao et al.

**Sim-to-Real and Feature Extraction:**
- DM as generator for domain transfer (Li et al.)
- Stable Diffusion for semantic feature extraction (Tsagkas et al.)

---

### Visual Data Augmentation

Diffusion models' strong image generation capabilities enable scaling datasets and enhancing scene understanding for vision-based manipulation.

#### 1. Scaling Data and Scene Augmentation

**Motivation:**
- Data-driven methods require extensive demonstrations
- Offline RL needs comprehensive state-action coverage
- Real-world data collection is time-consuming and expensive

**Approach:**
- Use pretrained DMs (e.g., Stable Diffusion) for semantic augmentation
- Inpainting for object/texture changes
- Object replacement with language descriptions
- Examples: Chen et al., Yu et al., Mandi et al.

**Augmentation Types:**

**Object Manipulation:**
- Color/texture changes
- Object replacement
- Increases task generalization

**Background Augmentation:**
- Robustness to irrelevant scene information

**Pose and Embodiment:**
- Camera viewpoint augmentation (Zhang et al.)
- Robot embodiment changes
- Simulation scene generation (Katara et al.)

**Offline RL Enhancement:**
- Hindsight experience replay with visual adaptation (Di Palo et al.)
- Align observations to new task instructions
- Increase successful executions in replay buffer

**Advantages over Domain Randomization:**
- Grounded in real-world data
- No complex per-task tuning
- Physical plausibility

**Limitations:**
- Most methods don't augment actions (only observations)
- Limited to augmentations that don't change actions
- Additional computational cost

#### 2. Sensor Data Reconstruction

**Motivation:**
- Incomplete sensor data (especially single-view)
- Occlusions
- Noisy/inaccurate sensors

**Viewpoint Reconstruction (Kasahara et al.):**
1. Project existing RGBD to new viewpoint
2. Segment objects with SAM
3. Inpaint missing regions with Dall·E
4. Consistency filtering across viewpoints
5. Predict missing depth information

**View Planning (Pan et al.):**
- DM generates geometric priors from 2D image
- View planner samples minimal viewpoint set
- Train NeRF for 3D reconstruction
- Minimize movement cost

**Limitations:**
- High computational cost
- Cannot handle completely occluded objects
- Limited adoption in manipulation (more common in broader robotics/CV)

**Alternative Approaches:**
- Make policies robust to incomplete/noisy data
- Examples: 3D-DP, 3D Diffuser Actor
- Challenge: Strong occlusions remain difficult

#### 3. Object Rearrangement

**Task:**
- Generate target object arrangements from language prompts
- Examples: "set dinner table", "clear kitchen counter"

**Evolution:**

**Early Methods (Zero-Shot):**
- Pretrained VLM Dall·E for generation
- Examples: Kapelyukh et al., Liu et al.
- Issues: Scene inconsistencies, lack of geometric understanding

**Advanced Methods:**
- Combine LLMs + VLMs (CLIP) + visual processing (NeRF, SAM)
- Custom DMs
- Examples: Xu et al., Kapelyukh et al.

**Relation to Object Pose Diffusion:**
- Similar to methods in Section 4.2
- Difference: Multiple objects, sparse language input
- Less focus on grasp/motion planning integration
- All methods validated on real robots

---

## Experiments and Benchmarks

### Common Benchmarks

**Simulation:**
- **CALVIN**: Long-horizon language-conditioned tasks
- **RLBench**: RGB-D tabletop manipulation
- **RelayKitchen**: Kitchen environment
- **Meta-World**: Multi-task, meta-RL
- **D4RL Kitchen**: Offline RL (primarily)
- **Adroit**: Dexterous manipulation
- **LIBERO**: Lifelong learning
- **LapGym**: Medical/surgical tasks

**Real-World:**
- **FurnitureBench**: Furniture assembly
- Custom real-world setups

### Common Diffusion Baselines

**Grasp Generation:**
- **SE(3)-Diffusion Policy** (Urain et al.)

**Reinforcement Learning:**
- **Diffuser** (Janner et al.)
- **Diffusion-QL** (Wang et al.)
- **Decision Diffuser** (Ajay et al.)

**Imitation Learning:**
- **Diffusion Policy (DP)** (Chi et al.)
- **3D Diffusion Policy** (Ze et al.)
- **3D Diffuser Actor** (Ke et al.)

### Key Performance Comparisons

**3D-DP vs. DP:**
- Average success: 74.4% vs. 50.2% (simulation)
- Real-world: 85.0% vs. 35.0%
- 24.2% improvement in simulation, 50% in real-world

**3D Diffuser Actor:**
- Outperforms 3D-DP on CALVIN (especially zero-shot long-horizon)
- No real-world comparison provided

**Decision Diffuser vs. Diffuser:**
- Outperforms on most tasks
- Especially on manipulation: block stacking, rearrangement
- Neither evaluated on real-world tasks

### Evaluation Patterns

**Training Data Sources:**
- Most: Real-world data
- Some: Simulation only (primarily RL methods)
- Few: Sim-to-real transfer (domain randomization or reconstruction)

**Real-World vs. Simulation:**
- Majority: Both simulation and real-world
- RL methods: Often simulation only
- Examples of sim-only: Wang et al., Janner et al., Pearce et al.

### Dataset Sizes

**Imitation Learning:**
- Wide range: 10-100 demos (common)
- Up to 70k+ for large-scale methods
- Real-world: 4.5h - 100+ demos typical

**Reinforcement Learning:**
- D4RL Kitchen: ~10k transitions
- Varies by benchmark

---

## Challenges and Limitations

### 1. Generalizability

**Imitation Learning Limitations:**
- Dependence on data quality and diversity
- Covariate shift problem
- Difficulty with out-of-distribution situations
- Requires high-quality expert demonstrations

**Offline RL Limitations:**
- Requires reward-labeled data
- Prone to overfitting
- Cannot react to distribution shifts
- More complex tuning than imitation learning

**Data Augmentation Limitations:**
- Most methods don't augment actions
- Only increase robustness to similar settings (colors, textures, backgrounds)
- Doesn't fundamentally solve generalization problem

**VLA Integration:**
- Strong multi-task generalization
- But: Action imprecision requires refinement
- Risk of restricting VLA generalizability

**Current State:**
- Good generalization: Object types, lighting, task complexity
- Limited generalization: Completely novel scenarios, strong domain shift

### 2. Sampling Speed

**Core Problem:**
- Iterative sampling process
- Time-intensive compared to single forward pass (GANs, VAEs)
- Impediment to real-time control

**Current Solutions:**

**DDIM (Most Common):**
- 10 steps typical
- 10× faster than DDPM
- ~5.6% performance drop (Ko et al.)
- Real-time possible: 0.1s latency on Nvidia 3080 (DP with 10 steps)

**Other Samplers:**
- DPM-solver: Superior to DDIM on image benchmarks
- Need validation on robotic tasks
- Many advanced samplers underutilized in robotics

**Problem Severity:**
- Task-dependent impact
- Already real-time capable with receding horizon control
- More critical when combined with VLAs

**Recent Advances:**
- BRIDGeR (Chen et al.): Sample from informed distribution
- Distillation techniques (Prasad et al.): Teacher-student approach
- Flow matching with self-consistency (Frans et al.): Single-step sampling

### 3. Visual Observations

**State Information Dependency:**
- Many RL methods use ground-truth state
- Only available in simulation
- Limits real-world applicability

**Exceptions:**
- Ren et al., Huang et al.: Process visual observations in RL

### 4. Research Gaps

**Online RL:**
- Very limited work with DMs
- Ding & Jin, Ajay et al.

**Continual Learning:**
- Widely unexplored
- Limited examples: Di Palo et al., Mendez-Mendez et al.
- Current limitations: Requires predefined skills, replays all data, doesn't prevent catastrophic forgetting

**View Planning and Occlusion:**
- Limited methods consider scene reconstruction
- High computational cost
- Cannot handle complete occlusions
- Strong occlusions remain major challenge

---

## Future Directions

### 1. Generalizability Enhancement

**Continual Learning:**
- Improve adaptability in dynamic environments
- Prevent catastrophic forgetting
- Computationally efficient updates
- **Promising approaches**: Combine with VLAs for semantic reasoning

**Unified Policies:**
- Cross-embodiment generalization
- Hierarchical architectures for diverse morphologies

**Data Scaling:**
- Action augmentation (not just observation)
- Leverage foundation models
- Internet-scale pretraining

### 2. Sampling Efficiency

**Unexplored Samplers:**
- Test DPM-solver and other CV methods on robotics tasks
- Validate performance beyond image benchmarks

**Robotics-Specific Methods:**
- BRIDGeR (Chen et al.)
- Distillation approaches
- Flow matching with self-consistency

**Critical for VLA Integration:**
- Combined VLA + DM inference bottleneck
- Need sub-0.1s latencies for reactive control

### 3. Scene Understanding

**View Planning:**
- Iterative planning with 3D representations
- Handle complete occlusions
- Integrate with existing 3D DMs (3D-DP, 3D Diffuser Actor)

**Semantic Reasoning:**
- Leverage VLMs for complex cluttered scenes
- Affordance-based planning
- Functional understanding

### 4. Robust Constraint Satisfaction

**Current Limitations:**
- Classifier guidance: No guarantees, additional training
- Classifier-free: Doesn't generalize to new constraints
- Constraint tightening: Limited evaluation

**Promising Directions:**
- Integrate movement primitives (Scheikl et al.)
- Multi-constraint settings
- Guaranteed satisfaction for safety-critical applications (e.g., surgery)

### 5. Dexterous Manipulation

**Current State:**
- Most work uses parallel grippers
- Limited dexterous grasp generation
- Examples: Si et al., Ma et al., Ze et al., Chen et al.

**Needs:**
- High-dimensional action spaces
- Complex contact dynamics
- Force control integration

### 6. Continual and Lifelong Learning

**Research Gap:**
- Almost completely unexplored with DMs
- Two limited examples

**Requirements:**
- Continuous skill acquisition
- Memory structures that grow over time
- Self-organized experience abstraction
- Exploration and feedback loops

---

## Conclusion

### Key Achievements

Diffusion models have emerged as state-of-the-art for robotic manipulation, offering:

1. **Multi-Modal Distribution Modeling**: Capture multiple valid solutions naturally
2. **High-Dimensional Robustness**: Handle complex visual and action spaces
3. **Training Stability**: Avoid mode collapse, more stable than GANs
4. **Smooth Generation**: Naturally produce physically plausible trajectories

### Architecture Insights

**Three Main Network Types:**
- **U-Net**: Most common, sample efficient, good for vision-based tasks
- **Transformer**: Best for long-horizon, high-dimensional, complex reasoning
- **MLP**: Most efficient, good for RL with state information

**Choice depends on:**
- Task complexity
- Input dimensionality
- Real-time requirements
- Available computational resources

### Application Landscape

**Trajectory Generation:**
- Dominant application
- Primarily imitation learning
- Growing RL integration
- Multi-modal inputs (vision, language, tactile)
- Long-horizon via hierarchical approaches
- VLA integration for generalization

**Grasp Generation:**
- SE(3) representations via score matching or flow matching
- Bi-equivariance for sample efficiency
- Language-guided and affordance-driven approaches
- Growing dexterous manipulation work

**Data Augmentation:**
- Scene scaling and augmentation
- Viewpoint reconstruction
- Object rearrangement
- Sim-to-real transfer

### Open Challenges

1. **Sampling Speed**: Still slower than single-pass methods, critical for real-time
2. **Generalizability**: Out-of-distribution scenarios remain challenging
3. **Continual Learning**: Almost completely unexplored
4. **Scene Understanding**: Occlusions and view planning underexplored
5. **Constraint Satisfaction**: No guaranteed satisfaction yet

### Promising Research Directions

1. **Efficient Sampling**: Test CV methods on robotics, develop robotics-specific samplers
2. **VLA Integration**: Balance generalization with action precision
3. **Continual Learning**: Develop non-catastrophic learning mechanisms
4. **3D Understanding**: Better occlusion handling, view planning
5. **Multi-Modal Fusion**: Tactile, force, audio integration
6. **Dexterous Manipulation**: High-DOF grasping and manipulation

### Final Perspective

Diffusion models represent a transformative paradigm for robotic manipulation, bridging the gap between high-dimensional perception and precise control. While challenges remain—particularly in sampling efficiency and generalization—the field is rapidly evolving. The integration with foundation models (VLMs, VLAs) and the development of more efficient sampling strategies promise to make diffusion-based manipulation more practical and widely deployable.

The comprehensive taxonomy and analysis provided in this survey establishes a foundation for future research, highlighting both the remarkable progress achieved and the exciting opportunities that lie ahead in this dynamic field.

---

## Key Statistics

- **First comprehensive survey** of diffusion models specifically for robotic manipulation
- **140+ papers** reviewed and categorized
- **Three main applications**: Trajectory generation, grasp synthesis, data augmentation
- **Three network architectures**: U-Net (most common), Transformer (most powerful), MLP (most efficient)
- **Two learning paradigms**: Imitation learning (dominant) and reinforcement learning (growing)
- **Common inference**: 5-10 DDIM steps (from 50-100 training steps)
- **Real-time capable**: 0.1s latency demonstrated with proper architecture choices

---

## References

**DOI:** 10.3389/frobt.2025.1606247  
**Open Access:** Creative Commons Attribution License (CC BY)  
**Supplementary Materials:** Available at frontiersin.org

---

*Survey compiled by the AI and Robotics (AIR) team at Karlsruhe Institute of Technology (KIT), September 2025*