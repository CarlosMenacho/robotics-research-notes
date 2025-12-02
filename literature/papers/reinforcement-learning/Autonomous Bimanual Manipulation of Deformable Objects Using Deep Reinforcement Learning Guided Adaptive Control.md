This paper proposes a deep reinforcement learning guided adaptive control (RLAC) framework that combines learning-based and Jacobian-based methods for autonomous deformable object manipulation in robot-assisted surgery. The system uses DRL policy sampling from simulations to estimate the initial deformation Jacobian, enabling more efficient and near-optimal positioning paths with robust sim-to-real performance.

## Problem Statement

Deformable object manipulation represents a critical challenge in robot-assisted surgery due to:

- Complex nonlinear deformation dynamics with high degrees of freedom
- Difficulty in accurate modeling using traditional rigid-body methods
- Trade-off between deformation model accuracy and computational cost
- Safety requirements for clinical applications

**Existing Approach Limitations:**

_Model-Based Methods:_

- Require precise model parameters and physical properties not always accessible
- Model-solving time cannot be neglected in real-time control
- FEM, XPBD, and ARAP models face accuracy-efficiency trade-offs

_Jacobian-Based Adaptive Control:_

- High versatility without requiring precise object modeling
- Potential non-convergence due to poor initial Jacobian estimation
- May require many iterations for preliminary adjustments

_Learning-Based Methods:_

- Require extensive simulation exploration time
- Sensitive to sim-to-real transfer differences and noise
- Challenging to achieve precise target coincidence

## Proposed RLAC Framework

**Core Concept:** RLAC harnesses DRL policy exploration in simulation to solve a reasonable estimation of the initial deformation Jacobian. In early control iterations, DRL agent actions are adopted until the estimated real-time Jacobian approximates the actual deformation model. Subsequently, independent Jacobian-based adaptive control executes with sufficient initial deformation awareness to achieve precise manipulation.

**Three-Phase Architecture:**

### Phase A: Offline Training (Proximal Policy Optimization)

**Training Environment:**

- Gazebo-based simulation powered by SOFA simulator
- 100×100 mm² two-dimensional tissue modeled as 50×50 mass-spring system
- Stiffness: 90 N/m, damping coefficient: 0.3 N·s/m
- Two grippers manipulating tissue edges to control internal feature points

**MDP Formulation:**

- State space: $st = [x_t^T, p_t^T, p_{targ}^T - p_t^T]^T \in R^{(2(M+2K))}$
    - xt: grasping point positions (M points)
    - pt: feature point positions (K points)
    - ptarg: target point positions
- Action space: $a_t = \delta x = x_{t+1} - x_t \in R^{2M}$ (discrete control of grasping points)

**Reward Structure:**

$$
r_t = ρ_1(d_{t-1} - dt) + ρ_2·sgn(d{t-1} - d_t) - ρ_3
$$

Where dt = ||ptarg - pt||2 (Euclidean distance to targets)

**Network Architecture:**

- Three-layer MLP: 256-512-256 dimensions
- PPO loss function with clipped gradients for stability
- Minimal training: 1,500 rounds (vs. 4,000 for full convergence)
- Training time: ~2 hours on Intel Core i9-14900K CPU

**Key Insight:** The actor network only needs to learn general approach behaviors, not precise convergence, significantly simplifying the training process.

### Phase B: Preliminary Adjustments Guided by Actor Network

**Initial Jacobian Estimation:**

The differential motion relationship: $\delta P_t = J_t· \delta x_t$

To estimate J0, the system collects multiple relationship pairs through Gaussian sampling:

$$
\hat{P}_0 = \hat{J}_0·X_0
$$

Where:

- $\hat{P}_0 = [δp0, δp0^1, ..., δp0^l] \in R^{(2K×(l+1))}$: collection of deformation vectors
- $X_0 = [δx_0, δx_0^1, ..., δx_0^l] ∈ R^{(2M×(l+1))}$: corresponding actor network actions

Sampling strategy: Gaussian distributions in polar coordinates around target points

- $δp_0^i = \tau(p^\varphi) - p_0,  p^φ \sim N(p^φ_{targ}, \sum)$

**Least Squares Solution:**

$$
\hat J_0 = \hat P_0 ·X_0^T·(X_0·X_0^T)^{-1}
$$

**Dynamic Scaling Adjustment:**

For subsequent iterations (t ≥ 1):

$$
\hat J_t = \hat J_{t-1}·exp(\lambda·log \frac{||p_t - p_{t-1}||_2}{||Ĵ_{t-1}·δx_{t-1}||_2}) - \gamma·e·δx_{t-1}^T
$$

- λ: scaling factor for magnitude adjustment
- γ·e·δxt-1^T: compensation for estimation error from AC algorithm

**Transition Criterion:**

$$
||Ĵ_{t-1}·δx_{t-1} - δp_{t-1}||_2 ≤ εJ
$$

When satisfied, execution transfers to pure AC phase.

### Phase C: Adaptive Control for Precise Positioning

**Estimator Update Rule:**

Prediction error: $e = Ĵ_{t-1}·δx_{t-1} - δp_{t-1} ∈ R^{2K}$

Gradient descent minimization of quadratic function $Q = e^T·e/2$:

$$
Ĵ_t = Ĵ_{t-1} - \gamma·e·δx_{t-1}^T
$$

**Controller Design:**

Minimizes distance function: $R = (p_{targ} - p_t)^T·(p_{targ} - p_t)/2$

Optimal motion direction (avoiding Jacobian inversion):

$$
\frac{d}{dt}x = -κ \frac{\delta R}{\delta x} = κ·(p_{targ} - p_t)·\hat J_t
$$

**Saturated Velocity Controller:**

$$
x_{t+1} = x_t + min(u_t, s·u_t/||u_t||_2)
$$

Where s determines maximum end-effector speed for safety.

**Convergence Criterion:** Task complete when ||pt - ptarg||2 ≤ ε for all feature points.

## Experimental Validation

### Simulation Experiments

**Setup:**

- 1,000 experiments per method (RL, AC, RLAC)
- Distance thresholds: 5-15 mm
- Equal distance requirements on both sides (d0l = d0r)
- AC initialized with Ĵ0 = I4 + E (E with |eij| < 0.5)

**Hyperparameters:**

|λ|γ|εJ|κ|s|
|---|---|---|---|---|
|0.05|0.05|0.10|0.50|1.00|

**Success Rate Results:**

- AC: Worst performance due to fixed initial Jacobian
- RL: Consistently high within training domain
- RLAC: Highest overall, significantly exceeds AC

**Path Efficiency Metric (PER):**

$$
PER = log \frac{l_l + l_r}{d_{0l} + d_{0r}}
$$

Where ll, lr are actual path lengths. Lower PER indicates more optimal (straighter) paths.

RLAC demonstrated significantly lower PER than both RL and AC, confirming near-optimal positioning paths.

### Real-World Experiments

**Physical Setup:**

- Two UR5 robots with grippers
- 400×400×0.4 mm³ latex tissue (4× simulation scale)
- Tissue secured at corners with tension
- RGB camera for feature point tracking
- Distance threshold: 10-40 mm (random)

**Success Rates (100 trials each):**

- RL: 75/100 (75%)
- AC: 84/100 (84%)
- RLAC: 91/100 (91%)

**Key Observations:**

- RLAC and AC showed minimal deviation from simulation results, demonstrating robust sim-to-real transfer
- RL performed worst in real world despite best simulation performance (sensitivity to environmental changes)
- RL sometimes lingered near targets due to detection fluctuations
- No cases where other methods succeeded but RLAC failed
- RLAC achieved immediate tissue damage/separation failures due to inappropriate paths

**Path Efficiency Analysis:**

- T-test confirmed RLAC and RL executed shorter paths than AC
- Levene's test showed RLAC had smaller PER variance (greater stability)
- RLAC followed near-straight paths from early iterations
- AC started from imprecise directions, requiring more adjustment steps
- RLAC approached targets with consistent pace and excellent convergence

## Technical Contributions

1. **Hybrid Architecture**: Novel integration of DRL policy networks with Jacobian-based adaptive control, leveraging advantages of both approaches
2. **DRL-Guided Initialization**: Uses actor network sampling to solve initial Jacobian estimation, avoiding random or manual initialization
3. **Smooth Transition Mechanism**: Clear criterion for switching from DRL guidance to pure AC execution based on Jacobian accuracy
4. **Simplified Training Requirements**: Actor network only needs to learn approach behaviors, not precise convergence, enabling minimal training time
5. **Path Optimality**: Adaptive Jacobian refinement enables near-optimal (nearly straight) positioning paths, minimizing tissue deformation and damage
6. **Robust Sim-to-Real Transfer**: Demonstrated consistent performance across simulation and physical experiments with different material properties

## Advantages Over Existing Methods

**Compared to Pure AC:**

- Significantly improved success rate through better initialization
- More efficient preliminary adjustments
- Near-optimal positioning paths from early iterations

**Compared to Pure DRL:**

- Simpler, faster training process
- Better sim-to-real transfer robustness
- Precise target convergence without extended training
- Stable performance with detection fluctuations

**Compared to Model-Based Methods:**

- No requirement for precise physical parameters
- Real-time execution without model-solving overhead
- Adaptability to varying material properties

## Limitations and Future Work

**Current Limitations:**

- Tested only within single task configuration
- Fixed contact positions and configurations
- Two-dimensional task representation
- Limited to internal point manipulation

**Proposed Extensions:**

- Refining learning-guided-control pipeline for diverse, complex tasks
- Non-fixed contact configurations and positions for flexible operations
- Extension to three-dimensional manipulation
- Application to broader robotic manipulation domains beyond surgery
- Integration with more advanced contact mode options

## Significance

This work addresses a fundamental challenge in surgical robotics by providing a practical framework that:

1. Combines strengths of learning-based exploration with model-free adaptive control
2. Achieves high success rates with minimal training requirements
3. Demonstrates robust sim-to-real transfer despite material property differences
4. Generates near-optimal manipulation paths that minimize tissue damage
5. Provides a generalizable approach applicable to various robotic manipulation tasks

The RLAC framework represents a significant step toward safe, efficient autonomous deformable object manipulation in robot-assisted surgery, with potential applications extending to broader robotics domains requiring precise, safe manipulation of compliant objects.