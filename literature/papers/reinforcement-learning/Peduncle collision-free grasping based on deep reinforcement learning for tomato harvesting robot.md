---

---
- DRL for tomato harvesting
- Achieves 85.5% success rate
- Proposes an improved HER-SAC algorithm with heuristic action fusion strategy that dynamically generates optimal grasping postures and collision-free paths.

---

## 1. Problem Statement

### Challenges in Tomato Harvesting

- Cherry tomato bunches have **thin, delicate peduncles** (6mm diameter) connecting clusters to main stem
- Traditional robots use **fixed postures** → frequent collisions → low success rates
- Plants grow in **varying spatial configurations** → need adaptive approach
- Requires **simultaneous optimization** of grasping posture and collision-free path

### Research Gap

- Previous methods treat posture and path planning as separate algorithms
- Limited generalization to diverse plant morphologies
- Numerical methods face exponential complexity

---

## 2. Proposed Solution

### 2.1 Robotic System

- **Manipulator:** 7-DOF Rokea xMateER3Pro (760mm reach, 3kg payload)
- **Vision:** Intel RealSense D435i RGB-D camera (1280×720)
- **End-Effector:** Pneumatic integrated gripper + cutter (154mm)
- **Computing:** i7-10700K CPU, 32GB RAM, GTX 1080Ti GPU

### 2.2 Multi-Keypoint Tomato Model (7 Feature Points)

**Components:**

- **Main Stem (M):** M1 (upper), M2 (lower) - modeled as cylinder (Ø14mm)
- **Peduncle (P):** P0 (junction), P3 (midpoint), P4 (endpoint) - two cylinders (Ø6mm)
- **Fruit Cluster (T):** TA, TB (bounding box corners) - cuboid (Ø25mm)

**Detection Performance:**

- MTA-YOLACT network: 95.4% F1-score (clusters), 80.2% mAP50 (peduncle/stem)
- Real-time: 13.3 FPS

### 2.3 Harvesting Optimal Posture Plane (HOPP)

**Novel Contribution:** Defines optimal grasping postures geometrically

**Three Components:**

1. **Grasping Point (Pg):** Located 10mm from main stem on peduncle
2. **Target Harvesting Zone (T):** Hemispherical region (r=150mm) guiding approach direction
3. **Optimal Posture Plane:** Perpendicular to cutting plane, parallel to main stem
    - Normal vector: $nt = v_{P0-M1} \times (v_{P0-P3} \times v_{P0-M1})$

### 2.4 Improved HER-SAC Algorithm

**Base Components:**

- **SAC (Soft Actor-Critic):** Maximum entropy DRL, off-policy
- **HER (Hindsight Experience Replay):** Addresses sparse rewards

**Novel Enhancements:**

**1. Heuristic Action Fusion Strategy:**

$$
q_{t+1} =  SLERP(\dot{q}, q_{Z-B}, \alpha(t)) ⊗ q_t
$$

- Combines DRL policy with geometric heuristics
- Accelerates learning while maintaining exploration

**2. Dynamic Gain Module:**

$$
α(t) = e^(-ωt), ω = 2.0
$$

- Time-decaying weight: early episodes use more heuristic, later more DRL
- Prevents local optima

**State Space (18D):**

- End-effector pose (7D: position + quaternion)
- Joint angles (7D)
- Distance to goal (1D)
- Normal to optimal plane (3D)
- Collision status (1D)

**Action Space (7D):**

- Position increment (3D: ±50mm steps)
- Orientation increment (4D: quaternion)

**Reward Function:**

$$
r_t(s_t,a_t) = \lambda_1 r_{goal} + \lambda_2×r_{pos} + \lambda_3×r_{ctrl} + r_{obs}
$$

- $r_{goal}$: Distance-based reaching
$$

r_{goal} = 
\begin{cases}
-\frac{1}{2} d_g^2 d_g \leq \varphi \\
-\varphi (|d_g| - \frac{1}{2}\varphi)d_g >\varphi
\end{cases}

$$


- $r_{pos}$: Angle to optimal plane
$$
r_{goal} =  
\begin{cases}
- (\cos^{-1} (\frac{V_x^e \bullet n_t}{|| v_x^e|| ||n_t||}))^2 \text{if} d_g  \leq \varphi \text{and} p_e \in T \\
- 10 \text{if} d_g \leq \varphi \text{and} p_e \notin T
\end{cases}
$$
- $r_{ctrl}$: Smooth motion
$$
r_{ctrl} = - \sqrt{\sum_{i=1}^{n_{links}} (\theta_t^{i} - \theta_{t-1}^i)^2}
$$
- $r_{obs}$: Collision penalty (0 or  -100)

$$
r_{obs} = 
\begin{cases}
100 \text{if collition} \\
0 \text{otherwise}
\end{cases}
$$

---

## 3. Experimental Results

### 3.1 Simulation Results (MuJoCo)

**Dataset:** 500 digital tomato models (400 train, 100 test)

**Ablation Study:**

|Gain Module|Convergence Episodes|Success Rate|
|---|---|---|
|**α(t)=e^(-2t)** (proposed)|**2,000**|**~100%**|
|α(t)=0.5 (fixed)|7,500|~98%|
|α(t)=0 (no heuristic)|8,000|~95%|

→ **75% faster convergence** with heuristic fusion

**Algorithm Comparison:**

|Algorithm|Success Rate|Distance Error (mm)|Orientation Error (rad)|Steps|
|---|---|---|---|---|
|**HER-SAC (ours)**|**97%**|**5.79 ± 2.32**|**0.0054 ± 0.0026**|**9.19 ± 1.25**|
|SAC|83%|11.96 ± 5.74|0.0316 ± 0.0351|10.88 ± 2.92|
|HER-DDPG|81%|13.92 ± 5.20|0.0331 ± 0.0168|15.90 ± 2.84|
|DDPG|73%|15.88 ± 6.08|0.0379 ± 0.0249|12.23 ± 2.54|
|HER-TD3|35%|23.84 ± 9.22|0.0771 ± 0.0380|15.32 ± 4.17|
|TD3|19%|27.79 ± 12.38|0.0972 ± 0.0449|15.98 ± 4.81|

**Key Findings:**

- +14-78% success rate improvement over other algorithms
- Best positioning and orientation accuracy
- Minimum 15.5% fewer steps required

### 3.2 Field Experiment Results (Real Greenhouse)

**50 tomato bunches tested across 4 experimental conditions:**

|Experiment|Posture Strategy|Path Planner|Attempts|Success|Success Rate|Time (s)|
|---|---|---|---|---|---|---|
|#1|Horizontal (fixed)|BIT-RRT|78|22|**28.2%**|10.08|
|#2|Parallel to stem|BIT-RRT|73|31|**42.5%**|10.22|
|#3|Optimal plane (single)|BIT-RRT|64|42|**65.6%**|11.02|
|**#4**|**Optimal plane (DRL)**|**HER-SAC**|**55**|**47**|**85.5%**|**11.42**|

**Performance Improvements:**

- **+57.3%** vs. fixed horizontal posture
- **+43.0%** vs. parallel to stem
- **+19.9%** vs. single optimal vector
- **21.8% faster** than comparable systems (Rong et al. 14.6s → 11.42s)

**Failure Analysis:**

|Failure Type|Exp #1|Exp #2|Exp #3|**Exp #4**|
|---|---|---|---|---|
|Collided with peduncle/stem|27|11|6|**2**|
|Collided with neighbors|1|3|8|**0**|
|Path planning failed|22|22|2|**0**|
|Inaccurate keypoints|6|6|6|**6**|

**Key Insights:**

- **100% path planning success** (vs. 65% for BIT-RRT)
- **0 neighbor collisions** (learned avoidance)
- Depth perception remains limitation (11-12% failures)

---

## 4. Key Contributions

### Methodological Innovations

1. **Harvesting Optimal Posture Plane (HOPP):** Geometric framework defining optimal grasping postures
2. **Improved HER-SAC:** Enhanced with heuristic fusion and dynamic gain module
3. **Integrated Approach:** Simultaneous optimization of posture and path
4. **Successful Sim-to-Real Transfer:** Validated in real greenhouse conditions

### Performance Achievements

- **85.5% field success rate** (state-of-the-art for full-bunch harvesting)
- **11.42s operation time** (faster than existing systems)
- **97% simulation success** with best accuracy metrics
- **75% faster learning** convergence with heuristic fusion

---

## 5. Comparison with State-of-the-Art

|System|Success Rate|Time (s)|Target|Key Feature|
|---|---|---|---|---|
|**Proposed (2024)**|**85.5%**|**11.42**|Full bunch|Adaptive DRL posture|
|Shi et al. (2023)|93.0%|N/A|Single tomato|Force-sensing|
|Wang et al. (2023)|88.0%|20.0|Single tomato|RRT + interpolation|
|Rong et al. (2022)|~70%|14.6|Full bunch|Pose recognition|
|Ye et al. (2021)|100%|4.24*|Litchi|Bi-RRT|

*Planning time only, not full harvest cycle

**Differentiators:**

- Focus on **full bunch** harvesting (higher throughput)
- **Adaptive posture** optimization (not fixed)
- **Learning-based** approach (continuous improvement potential)

---

## 6. Limitations & Future Work

### Current Limitations

1. **Path Complexity:** 33% longer paths due to posture optimization
2. **Depth Perception:** 11-12% failures from inaccurate keypoint positioning
3. **Limited Range:** Uses only 400×400×200mm of 760mm workspace
4. **Leaf Handling:** Requires pre-pruned environment

### Proposed Solutions

**Short-Term (1-2 years):**

- Active vision with optimal viewpoint selection
- Real-time pose refinement using end-effector camera
- Path optimization with multi-stage planning
- Workspace expansion

**Medium-Term (2-5 years):**

- Multi-robot coordination systems
- Foundation models for vision-language-action
- Online learning during deployment
- Integrated leaf manipulation

**Long-Term (5+ years):**

- Fully autonomous greenhouse operations
- Generalist agricultural robots (multiple crops)
- Human-robot collaboration

---

## 7. Technical Insights

### Why This Approach Works

**1. Entropy Regularization (SAC):**

- Encourages exploration during training
- Prevents premature convergence
- Automatic temperature tuning

**2. Hindsight Experience Replay (HER):**

- Addresses sparse reward problem
- Generates additional training signal from failures
- Improves sample efficiency

**3. Heuristic Fusion:**

- Provides good initialization
- Accelerates early learning
- Maintains exploration capability

**4. Dynamic Gain:**

- Balances heuristic vs. learned policy over time
- Prevents local optima
- Enables fine-tuning

### Critical Hyperparameters

- Reward scaling: λ1=10, λ2=2, λ3=0.2 (tuned via grid search)
- Dynamic gain: ω=2.0 (empirical testing)
- Threshold: φ=150mm (end-effector size + safety margin)
- Episodes: 100,000 for convergence

---

## 8. Implementation Guide

### Hardware Requirements

**Minimum:**

- 6-7 DOF manipulator (500mm+ reach)
- RGB-D camera (640×480+)
- i7 CPU, 16GB RAM, GTX 1060 GPU

**Recommended:**

- 7 DOF with force/torque sensor
- High-res RGB-D (1280×720+)
- i9 CPU, 32GB RAM, RTX 3080 GPU

### Development Timeline

**Total: 6-9 months**

1. Data Collection (2-4 weeks)
2. Perception System (4-6 weeks)
3. DRL Training (6-8 weeks)
4. Sim-to-Real Transfer (4-6 weeks)
5. Field Trials (8-12 weeks)



---

## 9. Key Takeaways

### For Researchers

- Sim-to-real transfer works with accurate physics + domain randomization  
- Heuristic integration accelerates DRL learning  
- Geometric constraints reduce search space effectively  
- Multi-objective rewards require careful tuning

---

## 10. Conclusion

This research successfully demonstrates that **Deep Reinforcement Learning** can solve complex agricultural manipulation tasks. The system achieves:

- **85.5% field success rate** (57.3% improvement)  
- **11.42s operation time** (21.8% faster than comparable systems)  
- **97% simulation success** with best accuracy  
- **100% path planning success** vs. 65% for traditional methods  
- **Successful sim-to-real transfer** with minimal fine-tuning

**Impact:**

- Advances autonomous harvesting technology
- Validates learning-based manipulation in real-world scenarios
- Reduces labor costs while increasing yield
- Provides methodology applicable to other bunch crops

**This work represents a significant step toward fully autonomous agricultural operations, addressing critical labor shortages while maintaining high harvest quality.**
