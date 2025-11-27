This systematic review examines **dual-arm robotic systems for agricultural applications**, analyzing their advantages over single-arm systems in handling complex tasks involving **occlusion, manipulation difficulty, and coordination requirements**. The review covers 33 papers and addresses four research perspectives:

1. **RP1: Tasks** - Agricultural operations addressed by dual-arm robots
2. **RP2: Platforms** - Hardware design (base, manipulators, grippers, sensors)
3. **RP3: Controls** - Motion planning, coordination, and feedback strategies
4. **RP4: Future Directions** - Remaining challenges and research opportunities

**Key Findings:**
- **48.4%** of studies validated only in laboratory/simulation (16 of 33 papers)
- Only **33%** (11/33) implemented explicit bimanual coordination
- Only **30%** (10/33) used heterogeneous grippers for role differentiation
- Main applications: harvesting (55%), pruning (12%), transportation (15%)

**Critical Gap:** Most systems mirror single-arm operations in parallel (goal-coordinated) rather than implementing true bimanual cooperation where arms perform complementary roles.

---

## 1. Introduction & Motivation

### 1.1 Why Dual-Arm Robots in Agriculture?

**Limitations of Single-Arm Systems:**
- Target occlusion by leaves, branches, or other fruits
- Limited workspace accessibility in dense canopies
- Inability to stabilize plants while manipulating
- Low efficiency for tasks requiring coordination
- Motion planning difficulties in cluttered environments

**Advantages of Dual-Arm Systems:**
- **Bimanual manipulation:** One arm holds, the other cuts/harvests
- **Goal-coordinated operation:** Both arms work simultaneously in different zones
- **Load distribution:** Improved payload capacity and stability
- **Human-like dexterity:** Can replicate human workers' two-handed strategies
- **Occlusion handling:** One arm clears obstacles while other works

### 1.2 Hierarchy of Dual-Arm Manipulation

**Classification (following Smith et al., 2012):**

| Category | Definition | Example in Agriculture |
|----------|------------|------------------------|
| **Un-coordinated** | Arms perform independent, unrelated tasks | Left arm welding, right arm palletizing |
| **Goal-coordinated** | Arms perform identical, independent sequences | Both arms harvesting fruits simultaneously |
| **Bimanual** | Arms perform different, complementary tasks toward common goal | One arm stabilizes branch, other harvests |

**Agricultural Focus:** This review concentrates on **coordinated** and **bimanual** categories.

### 1.3 Research Gap Analysis

**Keyword Co-occurrence Networks:**
- **"Agricultural robot"** network: High density, mature field
  - Keywords: obstacle detection, uncertainty, machine vision, navigation
  
- **"Dual-arm agricultural robot"** network: Low density, emerging field
  - Limited terms, fewer connections
  - Indicates early-stage research with limited in-depth exploration

**Implication:** Dual-arm agricultural robotics requires systematic investigation of task requirements, hardware design, control systems, and practical deployment strategies.

---

## 2. Agricultural Tasks

### 2.1 Pruning

**Definition:** Trimming overgrown branches to maintain plant structure, enhance yield, reduce disease risk.

**Two Main Approaches:**

**A. Morphology Adjustment**
- **Goal:** Maintain overall plant shape
- **Decision factors:** Simple—cut protruding branches
- **Complexity:** Lower
- **Example:** Fixed-pattern trimming

**B. Density Adjustment**
- **Goal:** Optimize branch density for airflow and light penetration
- **Decision factors:** Complex—must perceive most branches, determine optimal cutting points
- **Complexity:** Higher (preferred for quality management)
- **Example:** Selective branch removal in fruit trees

**Key Challenges:**
1. **Perception:** Overlapping branches hinder identification of pruning points
2. **Control:** Irregular growth patterns cause kinematic singularities, infeasible motion planning

**Dual-Arm Solution:** **Bimanual operation** - one arm grasps/stabilizes branch, other arm cuts
- Reduces occlusion
- Stabilizes target
- Enables precise positioning

**Representative Studies:**

| Reference | Platform | Application | DoF | Validation |
|-----------|----------|-------------|-----|------------|
| Korayem et al. [13] | UGV + dual arms | Tree pruning | High DoF | Laboratory |
| Wu et al. [35] | Wheeled UGV | Tea picking | 7-DoF each | Field |
| Nekoo et al. [36] | Ornithopter UAV | Leaf sampling | 94.1g dual-arm | Laboratory + Field |

**Key Innovation (Nekoo et al.):**
- Lightweight dual-arm (94.1g) for UAV platform (500g payload)
- Heterogeneous grippers: scissors-type + collection gripper
- Mimics human finger operations
- Perches on stems for manipulation

---

### 2.2 Fruit Thinning & Harvesting

**Challenges for Single-Arm Systems:**
- **Occlusion:** Fruits hidden by leaves, branches, other fruits
- **Motion planning:** Difficult to reach occluded targets
- **Low efficiency:** Sequential operation, long cycle times

**Dual-Arm Strategies:**

**Strategy 1: Bimanual (Role Differentiation)**
- **One arm:** Holds/clears obstacle (leaves, branches)
- **Other arm:** Harvests target fruit
- **Advantage:** Improves perception, enables precise cutting
- **Examples:** Grape harvesting [23], tomato picking [41], [72]

**Strategy 2: Goal-Coordinated (Parallel Harvesting)**
- **Both arms:** Harvest different fruits simultaneously
- **Same end-effectors:** Homogeneous grippers
- **Advantage:** Doubles throughput
- **Examples:** Apple/pear harvesting [21], multi-arm systems [20]

**Cluster Fruit Complexity:**
- Obstacles include **other non-target fruits**
- Requires **high-level decision-making:** harvest ordering
- **Sequential planning:** Which fruit to pick first to minimize interference

**Representative Studies:**

| Reference             | Crop       | Strategy    | End-Effectors    | Success Rate | Notes                     |
| --------------------- | ---------- | ----------- | ---------------- | ------------ | ------------------------- |
| Xiao et al. [41]      | Tomato     | Bimanual    | Cutting + vacuum | 87.5%        | Dual task-specific        |
| Yoshida et al. [21]   | Pear/Apple | Goal-coord. | Rotation-based   | N/A          | Outdoor conditions        |
| Stavridis et al. [23] | Grape      | Bimanual    | Grasp + cut      | Lab + field  | Synchronized manipulation |
| Li et al. [20]        | Apple      | Goal-coord. | Multiple arms    | N/A          | Shared workspace          |

**Key Technical Achievement (Stavridis et al.):**
- One arm manipulates grape cluster to expose stem
- Other arm detects and cuts stem
- Synchronized bimanual operation solves occlusion
- Validated in cluttered vineyard environments

---

### 2.3 Transportation

**Challenge:** Handling irregularly shaped, fragile, or densely packed objects while maintaining stability.

**Problem:** **Shifts in center of mass** when moving boxes with multiple objects
- Compromises stability
- Reduces handling precision
- Can damage delicate produce

**Single-Arm Limitations:**
- Restricted payload capacity
- Reduced handling versatility
- Requires advanced control (sliding mode, ZMP control)

**Dual-Arm Advantages:**
1. **Load distribution:** Two arms share weight
2. **Enhanced stability:** Better handling of mass moment shifts
3. **Increased payload:** Combined capacity exceeds single arm
4. **Precise force control:** Delicate handling of fruits/vegetables

**Control Strategies:**

**A. With Rigid Grippers:**
- Object-level force coordination
- Shared load management
- Internal force regulation

**B. With Soft Grippers:**
- Adaptive compliance
- Gentle contact with fragile surfaces
- Slip detection and prevention

**Representative Studies:**

| Reference | Application | Key Innovation | Payload |
|-----------|-------------|----------------|---------|
| Wang et al. [54] | Logistics | Navigation in uneven terrain | N/A |
| Kim et al. [55] | Load repositioning | Whiffletree mechanism for load distribution | 7 kg (2× UR5) |
| Kim et al. [49] | Delicate objects | Deep imitation learning (global + local networks) | Banana peeling demo |

**Technical Highlight (Kim et al. [49]):**
- **Dual-network architecture:**
  - **Global network:** High-level motion planning
  - **Local network:** Fine control for fragile object interaction
- **Coordinated transitions:** Smooth switching between networks
- **Application:** Banana peeling, demonstrating delicate manipulation

---

### 2.4 Other Applications

**Tasks Not Requiring Intensive Coordination:**
- Spraying
- Seeding
- Pollination
- Monitoring
- Sampling

**Why Dual-Arm?** **Efficiency improvement** through simultaneous parallel execution rather than occlusion handling or complex coordination.

**Representative Studies:**

| Reference | Application | System | Key Feature |
|-----------|-------------|--------|-------------|
| [56] | Pesticide spraying | SprayRo | Dual nozzles, mobile platform, dedicated app |
| [57] | Seeding | Dual PR arms | Digging + planting + watering + soil covering |
| [58] | Pollination | Stickbug (6 arms) | Parallel execution, kiwi drive, contact-based with felt-tip |

**Technical Note (Stickbug):**
- **Six-armed system** for maximum parallel execution
- Detection model + classifier
- Felt-tip end-effector for contact-based pollination
- Kiwi drive mechanism for maneuvering

---

## 3. Agricultural Platforms

### 3.1 Base Platforms

**Two Main Types:** UGVs (ground) and UAVs (aerial)

#### 3.1.1 Unpiloted Ground Vehicles (UGVs)

**Characteristics:**
- Extended operational time (hours)
- High payload capacity (10-100+ kg)
- High stability
- Limited to planar motion
- Cannot reach high canopies

**Two Subtypes:**

**A. Wheeled UGVs:**
- **Advantages:** 
  - Better mobility on firm ground
  - Lower soil compaction
  - Easier steering
  - More common in literature
- **Disadvantages:**
  - Slippage on loose soil
  - Limited traction
- **Applications:** Most harvesting robots [35], [60]

**B. Tracked UGVs:**
- **Advantages:**
  - Superior traction on loose soil
  - Higher payload capacity
  - Better stability on uneven terrain
- **Disadvantages:**
  - Higher soil compaction
  - Reduced maneuverability
  - Less common in research
- **Applications:** Heavy-load harvesters with bins [59]

**Trade-off:** **Mobility vs. Payload**
- Wheeled → Better for navigation, lighter loads
- Tracked → Better for heavy loads, challenging terrain

#### 3.1.2 Unpiloted Aerial Vehicles (UAVs)

**Characteristics:**
- Enhanced mobility (3D motion)
- Broader operational range
- Can reach high canopies
- Short operational time (< 15 min)
- Very low payload capacity (< 6 kg)
- Limited stability

**Example Systems:**

| Reference | Application | Manipulator Weight | Platform Payload | Flight Time |
|-----------|-------------|-------------------|------------------|-------------|
| Nekoo et al. [36] | Leaf sampling | 94.1 g | 500 g | Short |
| Liu et al. [61] | Avocado harvesting | 5.36 kg | ~6 kg | Limited |

**Critical Bottleneck:** **Energy density**
- Arm-to-airframe mass ratio << UGVs
- Orders of magnitude difference
- Limits applications to canopy-level access only

**Platform Choice Decision:**
- **UGVs:** Long duty cycles, heavy tooling, endurance critical
- **UAVs:** Vertical reach, canopy access, mobility critical

---

### 3.2 Dual-Arm Robots (Manipulators)

**Key Design Factors:**
- Degrees of Freedom (DoF)
- Configuration (anthropomorphic, Cartesian, SCARA)
- Link/joint types
- Workspace requirements

**Challenge:** Higher DoF = more complexity
- More motion planning complexity
- Higher risk of self-collision between arms
- Kinematic inefficiencies possible

#### 3.2.1 Configuration Types

**A. Anthropomorphic Manipulators (Most Common)**

**Characteristics:**
- **DoF:** 6-7 per arm
- **Structure:** Resembles human arm
- **Advantages:**
  - High dexterity
  - Can weave around foliage
  - Navigate irregular trellises
  - Suitable for cluttered orchards
- **Disadvantages:**
  - Complex inverse kinematics
  - Higher self-collision risk
  - Longer planning time

**Applications:** 
- Orchards with dense canopies [22], [27]
- Irregular environments
- Cluttered scenarios

**B. Cartesian Systems**

**Characteristics:**
- **Structure:** Orthogonal linear slides (XYZ)
- **Advantages:**
  - Simple kinematics
  - Fast planning
  - Virtually no self-collision
  - High repeatability
- **Disadvantages:**
  - Larger footprint
  - Box-shaped workspace only
  - Limited dexterity

**Applications:**
- Straight crop rows [20], [31], [67], [71]
- Uniform canopy height
- Structured environments

**When to Use:** Crop rows straight, canopy uniform → simplicity outweighs dexterity

**C. SCARA (Selective Compliance Assembly Robot Arm)**

**Characteristics:**
- **Structure:** Planar rotary joints + vertical linear motion
- **Advantages:**
  - High-speed operation
  - High accuracy in plane
  - Good stiffness
  - Repetitive picking efficiency
- **Disadvantages:**
  - Requires flat, leveled environment
  - Cannot reach around obstacles
  - Limited vertical workspace

**Applications:**
- Indoor benches [41], [54], [72]
- Height-controlled environments
- Flat surfaces

**Configuration Selection Guide:**

| Environment Type | Best Configuration | Reason |
|-----------------|-------------------|---------|
| Unstructured orchard | Anthropomorphic | Maximum reach, dexterity |
| Straight-row crops | Cartesian | Simplicity, no self-collision |
| Flat indoor lines | SCARA | Speed, accuracy, repeatability |

**Key Principle:** Configuration should follow **task geometry** rather than one-size-fits-all.

---

### 3.3 Grippers (End-Effectors)

**Critical Design Question:** Heterogeneous or Homogeneous?

**Answer depends on control strategy:**
- **Bimanual control** → Heterogeneous grippers (different roles)
- **Goal-coordinated control** → Homogeneous grippers (same task)

**Survey Finding:** 
- Out of 33 papers, only **10 (30%)** used heterogeneous designs
- **23 (70%)** used homogeneous grippers
- **Implication:** Role-divided tooling is underexplored

#### 3.3.1 Heterogeneous Grippers (Bimanual Control)

**Purpose:** Each arm performs **different subtask** toward common goal

**Examples:**

| Reference | Application | Gripper 1 | Gripper 2 | Strategy |
|-----------|-------------|-----------|-----------|----------|
| [57] | Seeding | Soil digger | Soil coverer | Sequential subtasks |
| [23] | Grape harvesting | Grasper | Cutter | Hold + cut |
| [41], [72] | Tomato harvesting | Cutting device | Vacuum cup | Separate + grip |

**Advantages:**
- Role specialization
- Reduced cycle time (no tool changes)
- Optimized for specific subtasks

**Disadvantages:**
- Added mechanical complexity
- Higher cost
- More maintenance

**When to Use:** Workflow requires **role differentiation** (e.g., hold + cut, dig + cover)

#### 3.3.2 Homogeneous Grippers (Goal-Coordinated Control)

**Purpose:** Both arms perform **identical task** in separate zones concurrently

**Examples:**

| Reference | Application | Gripper Type | Detachment Method |
|-----------|-------------|--------------|-------------------|
| [56] | Spraying | Nozzles | N/A |
| [69] | Eggplant harvesting | Three-finger grippers | Grip + rotate |
| [21] | Apple/pear | Rotation-based | Twist-off |
| [59] | Grape | Curved cut-clip finger | Clip stem |
| [67] | Apple | Soft silicone vacuum | Gentle suction |

**Advantages:**
- Simplified maintenance
- Easier control (symmetric)
- Lower cost
- Parallel throughput

**Disadvantages:**
- Limited flexibility
- Cannot handle asymmetric forces
- Both arms must be identical

**When to Use:** High-throughput, **symmetric operations** where parallel work is main goal

**Design Decision Framework:**
- **Complementary roles needed** (hold + cut, dig + cover) → Heterogeneous
- **Parallel throughput primary** (simultaneous harvesting) → Homogeneous

---

### 3.4 Sensors

**Purpose in Dual-Arm Systems:** Synchronizing two independent manipulators

**Key Sensor Types:**

#### 3.4.1 LiDAR (Light Detection and Ranging)

**Applications:**
- Terrain analysis [59]
- 3D mapping [68]
- Detailed spatial data [64], [70]
- Spray pattern analysis [73], [74]

**Advantages:**
- High-resolution spatial data
- Works in various lighting
- Precise distance measurements

**Use in Dual-Arm:**
- Provides **shared geometric map** for both arms
- Obstacle detection and avoidance
- Workspace boundary definition

#### 3.4.2 Cameras (RGB, RGB-D, Stereo, Multispectral, Infrared)

**Mounting Strategies:**
- **Eye-in-hand:** Camera on wrist/gripper
- **Multi-view:** Multiple cameras around chassis
- **Hybrid:** Both approaches combined

**Example Deployments:**

| Reference | Camera Setup | Application | Key Metric |
|-----------|-------------|-------------|------------|
| [41] | Binocular stereo | Tomato harvesting | 96% detection, 87.5% harvest success |
| [21] | 4× Intel RealSense D435 | Apple/pear | Multi-height coverage, reduced blind spots |

**Fusion Approaches:**
- **LiDAR + Camera:** Structure + semantics
- **Advantage:** Improved object delineation in cluttered scenes [75]
- **Benefit for Dual-Arm:** Both arms can reach into occluded canopy with confidence

**Use in Dual-Arm:**
- **Semantic cues:** Fruit/branch recognition
- **Target localization:** 3D pose estimation
- **Occlusion reasoning:** What to move, what to harvest

#### 3.4.3 Infrared (IR) Proximity Sensors

**Purpose:** **Millimeter-scale range** data during final approach

**Applications:**
- Final centimeters before contact
- Deceleration control
- Avoiding fruit bruising

**Use in Dual-Arm:**
- Two wrists decelerate **together**
- Synchronized approach
- Gentle contact

#### 3.4.4 Inertial Measurement Units (IMUs)

**Measurements:**
- Acceleration
- Angular velocity
- Orientation

**Applications:**
- Filter out base vibration
- Dynamic alignment
- Motion stabilization [71]

**Use in Dual-Arm:**
- Keep manipulators **dynamically aligned** during parallel work
- Compensate for mobile base movement

#### 3.4.5 Tactile Sensors (Force/Torque, Slip Detection)

**Purpose:** **Force-level feedback** that vision cannot provide

**Types:**
- **Wrist F/T sensors:** 6-axis force-torque at wrist
- **Tactile arrays:** Distributed pressure on gripper surfaces
- **Slip sensors:** Detect object slipping

**Applications:**
- Measure normal force
- Detect slip onset
- Redistribute load between arms

**Use in Dual-Arm:**
- If one side loosens, **immediately redistribute load** to partner
- Maintain gentle grasp force below bruise threshold
- Coordinated force control

**Multimodal Integration:**
- LiDAR → Shared geometric map
- Cameras → Semantic recognition (fruit, branch, leaf)
- IR/Tactile → Force loop closure

**Result:** Enable tightly coordinated tasks (grape-cluster harvesting, selective pruning, two-arm bin loading)

---

## 4. Control Mechanisms

### 4.1 Motion Planning

**Challenge:** Generate **collision-free trajectories** for both arms while avoiding:
- External obstacles (plants, structures)
- **Self-collision** between the two arms

**Difficulty:** Self-collision risk is **much higher** in dual-arm systems

**Three Complementary Approaches:**

#### 4.1.1 Workspace-Partition Methods

**Concept:** Divide workspace into **sub-zones**, assign zones to each arm

**Advantages:**
- Shrinks search space
- Reduces inverse-kinematic branches
- Simpler planning

**Examples:**

| Reference | Method                               | Key Feature                                |
| --------- | ------------------------------------ | ------------------------------------------ |
| [69]      | Zone division + approach constraints | Narrower zones, fewer IK branches          |
| [63]      | Brainstorm optimizer + multi-TSP     | Distance thresholds guarantee no collision |

**Limitation:** Requires clear zone boundaries (works well in structured environments)

#### 4.1.2 Constraint-Hierarchy Planners

**Concept:** Keep all zones active, resolve priorities at **each control step**

**Method:** Stack constraints in priority order:
1. Joint limits (highest priority)
2. Self-collision margins
3. Moving-obstacle constraints
4. Task objectives (lowest priority)

**Example:**
- [40] Quadratic-program cascade
- Always finds feasible motion
- Higher computation cost than partition

**Advantage:**  Guaranteed feasibility in clutter
**Disadvantage:**  Higher computation time

#### 4.1.3 Adaptive-Sampling Planners

**Concept:** Improve exploration efficiency through **intelligent sampling**

**Example: EDDS bi-RRT [65]**
- **Rotate tree-growth direction** when node encounters obstacle
- **Halve step size** if search stagnates
- Integrated into 17-DoF humanoid
- **Result:** ~1/3 fewer nodes than classic bi-RRT

**Division-Merge IK [66]:**
- Solve each arm's sub-chain **analytically**
- Merge poses
- **Eliminates iterative solvers** from inner loop
- Faster computation

**Advantages:**
- Robust in unstructured orchards
- Efficient exploration

**Disadvantage:**
- Still depends on fast collision checking

**Method Comparison:**

| Method | Best For | Trade-off |
|--------|----------|-----------|
| **Partition** | Clear zone division (crop rows) | Simplicity vs. flexibility |
| **Constraint hierarchy** | Cluttered environments | Feasibility vs. computation time |
| **Adaptive sampling** | Unstructured orchards | Robustness vs. collision-check speed |

**Promising Direction:** **Combine approaches**
- Example: Seed EDDS with partition-based waypoints
- Goal: Millisecond-level dual-arm replanning in field

---

### 4.2 Harvest Ordering

**Problem:** Sequence in which two arms harvest multiple fruits **dominates cycle time**

**Two Problem Framings:**

#### 4.2.1 Time-Logic Coordination

**Example: Li et al. [20]**

**Constraints Modeled:**
1. **Laser scanner interference:** Alternating scan cycles
2. **Overlapping suction lines:** Central vacuum prevents simultaneous zone work

**Solution:** Temporal rules encoded as constraints

**Result:** Meaningful idle-time reduction vs. naïve simultaneous planning

**Advantages:**
- Simple to implement
- Intuitive

**Disadvantages:**
- Hand-tuned constraints
- Difficult to adapt to new scenarios

#### 4.2.2 Task-Sequence Assignment

**Example: Lammers et al. [67]**

**Approach:**
- Partition workspace into **"fire-extinguisher" style regions**
- Use **Markov decision-process RL**
- Assign 5 motion phases to whichever arm can execute without conflict:
  1. Approach
  2. Extension
  3. Grasp
  4. Retraction
  5. Placement

**Grouping Strategy:** Phases that can run in **parallel**

**Result:** Shorter completion time than nearest-fruit heuristics

**Advantages:**
- Automatic adaptation
- Learns from data

**Disadvantages:**
- Requires training data
- Added computation

**Comparison:**

| Approach | Implementation | Adaptability | Computation |
|----------|---------------|--------------|-------------|
| **Time-logic** | Simple | Low (hand-tuned) | Low |
| **Sequence learning** | Complex | High (learns) | Higher |

**Hybrid Opportunity:** Learning-based ordering **seeded** with time-logic rules
- Combines robustness with adaptability
- Promising for field trials

---

### 4.3 Tactile Feedback Control

**Purpose:** Provide **contact, friction, pressure data** that vision alone cannot

**Benefit:** Two arms can **adapt forces** when manipulating fragile/occluded produce

**Note:** Most examples use single manipulator, but techniques **scale naturally to dual-arm**

**Three Technical Strands:**

#### 4.3.1 Contact-Triggered Motion Reconfiguration

**Concept:** Switch trajectory as soon as **contact detected**

**Example: [77]**
- Place taxels along arm
- Upon contact → switch to alternate joint trajectory
- **Result:** Arm slides along obstacle instead of stopping

**Extension: [78]**
- Redundant mobile base + multiple arms
- Add **self-collision distances** to task-priority stack
- **Dual-arm benefit:** Millisecond-level reactions with minimal code changes

**Advantage:** Reflex-like reactions

#### 4.3.2 Predictive Force Control

**Traditional Approach:** Linear contact model, replan at fixed intervals
- **Problem:** Underestimates force peaks on irregular fruit clusters

**Neural Network Approach: [79]**
- Replace linear model with **neural network**
- **Predicts next contact wrench** from recent tactile images
- **Dual-arm transfer:** Network runs per gripper
- **Benefit:** Anticipate partner-induced disturbances

**Advantage:** Forecast contact dynamics

#### 4.3.3 Human-in-the-Loop Interfaces

**Concept:** Operator injects expertise through **gestures** when perception uncertain

**Example: [80]**
- **Data glove** + OptiTrack tracker
- Operator steers tomato harvester in **6 DoF**
- Embedded tactile sensors stream **real-time contact forces**

**Benefits:**
- Warns operator of incipient slip
- Modulates grasp force against crop-specific bruise thresholds
- Transferable safety layer

**Dual-Arm Extension:**
- Interface addresses each wrist **independently**
- Can teleoperate either arm alone or **both cooperatively**
- Straightforward fallback for coordinated harvesters

**Three Ingredients Summary:**

| Approach | Function | Dual-Arm Benefit |
|----------|----------|------------------|
| **Contact-triggered** | Reflex-like reactions | Fast adaptation per arm |
| **Predictive** | Forecast dynamics | Anticipate partner disturbances |
| **Gesture override** | Human insight | Cooperative teleoperation |

**Field Deployment Hurdle:** **Outdoor-rated tactile skins**
- Must survive dust, moisture, temperature swings
- Currently limiting factor

**Immediate Steps:**
1. Engineer outdoor-rated skins
2. Fuse data-driven force prediction with human gestures
3. Enable reliable bimanual operation

---

### 4.4 Dual-Arm Coordination

**Goal:** Coordinate two manipulators to handle **one constrained object** or perform **complementary tasks**

#### 4.4.1 Historical Foundation

**Early Work: Hybrid Position-Force Framework**

**Yoshikawa & Zheng (1993) [81]:**
- Extended single-arm formulation [82] to **two manipulators**
- Separate object motion from **internal forces**
- Internal forces: do not affect object pose but maintain grasp

**Key Concepts:**
- **Task Jacobian:** Maps joint velocities to task-space velocities
- **Hand-constraint matrix:** Specifies grasp constraints

**Application:** Standard for agricultural robots that must **hold branch while cutting**

#### 4.4.2 Stability and Dynamic Coupling

**Nakamura et al. (1989) [83]:**
- Introduced **generalized Jacobian**
- Captures interaction between two arms

**Schneider & Cannon (1989) [84]:**
- **Object-level impedance control**
- Regulates both motion AND internal force

**Application:** Recent impedance/admittance controllers
- One arm absorbs fruit motion
- Other arm completes cut

#### 4.4.3 Modern Extensions

**Survey by Smith et al. (2012) [2]:**
- Theories migrated from fixed-base → mobile + humanoid systems

**Recent Example: Stavridis et al. (2025) [23]:**
- **Grape harvesting robot**
- One arm stabilizes cluster
- Second arm cuts stem
- Demonstrated in field

#### 4.4.4 Current Limitations

**Most agricultural prototypes:**
- Rely on quasi-static models
- Low-frequency force sensing
- Limits cycle time
- Difficult real-time internal-force regulation in oscillating plants

**Requirements for Next Generation:**
- High-bandwidth six-axis force-torque sensors
- Fast inverse-dynamics solvers

**Beyond "Hold and Cut":**
- Cooperative carrying of fruit trays
- Dynamic manipulation
- Real-time force adaptation

#### 4.4.5 Future Integration

**Needed Capabilities:**
1. **Object-level impedance** with vision-based state estimation
   - Internal force targets adapt to branch stiffness
   
2. **LLM planners** for high-level role assignment
   - Example: "Left arm stabilize, right arm cut"
   - Realized by object-space controller (Yoshikawa framework)

**Goal:** Push dual-arm robots from **laboratory demos** → **field-ready, fully coordinated harvesters**

---

## 5. Challenges and Future Perspectives

### 5.1 RP1: Tasks - Field Deployment Gap

**Current State:**
- **48.4%** validation only in lab/simulation (16 of 33 papers)
- **51.6%** when considering only papers with identifiable validation (16 of 31)

**Problem:** Limited field deployment, sparse development

**Required Improvements:**

**1. Task Characteristic Identification:**
- Fully identify task-performing methods (Fig. 4, Fig. 7)
- Understand constraints (e.g., occluded area restoration)
- Cope with site uncertainty

**2. High-Level Perception & Decision Framework:**
- Object cluster recognition [85]
- 3D morphology recognition [86]
- Object characteristic classification [87]
- Environmental constraint alleviation [37]

**Example Framework:** Task-relevant plant part searching [88]
- Automate harvesting and de-leafing
- Distinguish between peduncle, petiole, tomato
- Multi-stem identification

**3. Robust Performance Guarantees:**
- Tune frameworks for field characteristics
- Integrate with robot system
- Validate across multiple seasons/sites

**Future Direction:** More field experiments, comprehensive validation, practical deployment focus

---

### 5.2 RP2: Platforms - Gripper Configuration Gap

**Survey Finding:**
- Out of 33 papers, only **10 (30%)** adopted heterogeneous grippers
- Among 11 **bimanual** studies, **4 (36%)** still used homogeneous grippers

**Problem:** Not fully exploiting structural advantages of dual-arm systems

**Gap:** Current practices don't align with functional requirements of bimanual operation

**Required Improvements:**

**1. Functionally Distinct Gripper Design:**
- Tailored to specific subtasks (grasping [89], cutting [90], pushing [79])
- Dedicated end-effectors enable clear role allocation
- Examples: scissors vs. collection gripper, digger vs. coverer

**2. Task-Role Hierarchical Planners:**
- Automatically assign subtasks to appropriate manipulator
- Based on gripper capabilities
- Example: [88] plant part searching framework

**Benefits:**
- Role specialization
- Reduced cycle time
- Optimized performance

**Future Direction:** Design more heterogeneous gripper systems, develop automatic role-assignment planners

---

### 5.3 RP3: Controls - Coordination Gap

**Survey Finding:**
- Only **11 of 33** papers embedded explicit bimanual coordination
- Most still **mirror single-arm routines** (parallel, not cooperative)

**Problem:** Two manipulators move in **parallel** rather than in **cooperation**

**Required Progress on Four Fronts:**

#### 5.3.1 Task-Semantic Role Allocation

**Current Gap:** Geometry-centric planners decide **where** but not **why**

**Needed:** High-level decision-making
- Which arm stabilizes branch?
- Which arm cuts?
- When to swap roles? (as occlusions/weight distribution change)

**Task Methods:** [93], [94]
**Crop Characteristics:** [95]

**Solution:** **LLM-powered planners** [91]
- Generate high-level assignments
- Example: "Left arm stabilize, right arm cut"
- Adapt to changing conditions

#### 5.3.2 Object-Level Force Sharing

**Current Gap:** Simply mirroring joint torques ignores asymmetric fruit mass and branch stiffness

**Problem:** One gripper slips while other carries load

**Needed:** Extend Yoshikawa's hybrid position-force framework to crops
- Estimate internal forces in **real time**
- Adapt compliance to **living-plant variability**

**Challenge:** Living plants have variable stiffness, not constant like industrial objects

#### 5.3.3 Multimodal Feedback Fusion

**Current Gap:** Few systems fuse vision + tactile at control rate

**Vision:** Localizes fruit
**Tactile/F-T:** Feels slip or bruising

**Needed:** Pipeline blending RGB-D + kilohertz tactile
- Each arm predicts incipient slip
- Modulates grasp forces **before damage occurs**

**Benefit:** Prevent fruit bruising through coordinated force control

#### 5.3.4 Shared Evaluation Metrics

**Current Gap:** Most studies report isolated success rates → difficult comparison

**Needed Benchmarks:**
- **Cycle time** per fruit cluster in dense canopy
- **Root-mean-square internal-force error** during cooperative cuts
- **Cumulative bruise ratio**

**Benefit:** Allow community to:
- Quantify gains
- Compare methods
- Focus efforts where most needed

**Summary - Four Intertwined Challenges:**

| Challenge | Current State | Needed Solution |
|-----------|---------------|-----------------|
| **Role allocation** | Geometry-only | Semantic LLM planners |
| **Force sharing** | Mirror torques | Object-level impedance |
| **Feedback fusion** | Separate streams | Real-time multimodal |
| **Metrics** | Isolated rates | Common benchmarks |

**Goal:** Move beyond duplicated single-arm loops → truly cooperative dual-arm field robots

---

### 5.4 RP4: Future Directions - Scene Understanding

**Current Limitation:** Systems falter in cluttered, changing orchards requiring **real-time replanning**

**Problem:** Bounding-box detection insufficient

**Needed:** **Scene understanding** - how fruits, leaves, stems **depend on one another**

**Concept:** **Relational awareness** before deciding arm cooperation

#### 5.4.1 Scene Graphs for Agriculture

**Recent Work (Other Fields):**
- Convert images into **scene graphs** [97]
- Language-like structures listing objects
- Label spatial/functional ties between them

**Agricultural Requirements:**
- **Detect objects:** Fruits, leaves, stems
- **Infer positions:** Up, down, left, right, near
- **Recognize attributes:** Ripeness, rigidity, color
- **Understand dependencies:** Stem-leaf, stem-fruit, leaf-fruit

**Current Limitation:** Limited agricultural datasets

**Solution:**
- Build scene-label datasets
- Pre-train models
- Fine-tune for agricultural applications [98]

#### 5.4.2 Benefits of Scene Understanding

**Example Use Cases:**

**1. Occlusion Reasoning:**
- Know that one leaf occludes fruit
- Assign left arm to clear foliage
- Right arm cuts
- Minimize branch oscillation and tool interference

**2. Language Grounding:**
- Operator: "Stabilize the branch, then cut here"
- Robot translates to complementary roles for two arms

**3. Harvest Ordering:**
- Understand which fruit blocks access to others
- Plan optimal sequence
- Reduce total manipulation time

#### 5.4.3 Multimodal Safety

**Vision Alone Insufficient:** Cannot guarantee safe grasping outdoors

**Required Sensors:**
- **Tactile skins:** Detect incipient slip
- **Wrist F/T sensors:** Report load sharing
- **Fusion:** Combine with RGB-D at control-loop speed

**Benefits:**
- React **before fruit damage** occurs
- Clamp force just below **crop-specific bruise limits**
- Remain stable despite lighting, wind, branch motion

#### 5.4.4 Path to Full Autonomy

**Three Essential Components:**

1. **Agricultural scene-graph datasets**
   - Large-scale labeled data
   - Multiple crops, environments, seasons

2. **Relation-aware planners**
   - Balance forces AND geometry
   - Understand object dependencies

3. **Multimodal feedback loops**
   - Close gap between global perception and local interaction
   - Real-time force control

**Vision:** Fully autonomous dual-arm field robots that:
- Understand complex scenes
- Reason about object relationships
- Adapt forces in real time
- Operate reliably outdoors

---

## 6. Summary Tables

### 6.1 Task Summary

| Task | Key Challenge | Dual-Arm Strategy | Success Rate Range | Validation |
|------|---------------|-------------------|-------------------|------------|
| **Pruning** | Occlusion, irregular growth | Bimanual (hold + cut) | N/A | Mostly lab |
| **Harvesting** | Occlusion, efficiency | Bimanual / Goal-coordinated | 65-96% | Lab + field |
| **Transportation** | Mass shifts, fragility | Bimanual (load distribution) | N/A | Lab |
| **Spraying** | Coverage | Goal-coordinated (parallel) | N/A | Field |
| **Seeding** | Precision | Sequential dual-arm | N/A | Lab |
| **Pollination** | Throughput | Goal-coordinated (6 arms) | N/A | Lab |

### 6.2 Platform Comparison

| Component | Type | Advantages | Disadvantages | Best Use |
|-----------|------|-----------|---------------|----------|
| **Base: UGV** | Wheeled | Better mobility, low compaction | Slippage on loose soil | Most applications |
| **Base: UGV** | Tracked | High payload, stability | Soil compaction, less common | Heavy loads |
| **Base: UAV** | Aerial | 3D motion, canopy access | Low payload (<6kg), short time | High canopy |
| **Arm: Anthropomorphic** | 6-7 DoF | High dexterity, cluttered envs | Complex IK, self-collision | Orchards |
| **Arm: Cartesian** | XYZ slides | Simple, fast, no collision | Large footprint, box workspace | Straight rows |
| **Arm: SCARA** | Planar + Z | High speed, accuracy | Requires flat surface | Indoor benches |
| **Gripper: Heterogeneous** | Different | Role specialization | Higher complexity | Bimanual tasks |
| **Gripper: Homogeneous** | Identical | Simple maintenance | Limited flexibility | Parallel tasks |

### 6.3 Control Method Comparison

| Method | Purpose | Advantage | Disadvantage | When to Use |
|--------|---------|-----------|--------------|-------------|
| **Workspace partition** | Motion planning | Simple, fast | Less flexible | Clear zones |
| **Constraint hierarchy** | Motion planning | Always feasible | Higher computation | Cluttered |
| **Adaptive sampling** | Motion planning | Robust exploration | Needs fast collision check | Unstructured |
| **Time-logic** | Harvest ordering | Simple implementation | Hand-tuned | Simple constraints |
| **Sequence learning** | Harvest ordering | Automatic adaptation | Needs training | Complex tasks |
| **Contact-triggered** | Tactile control | Reflex-like | Binary switching | Obstacle contact |
| **Predictive force** | Tactile control | Forecast dynamics | Needs training | Fragile objects |
| **Gesture override** | Tactile control | Human expertise | Requires operator | Uncertain cases |

### 6.4 Research Gap Summary

| Area | Current State | Gap | Required Solution |
|------|---------------|-----|-------------------|
| **Validation** | 48.4% lab/simulation only | Lack of field deployment | Comprehensive field trials |
| **Coordination** | Only 33% bimanual (11/33) | Parallel vs. cooperative | True coordination control |
| **Grippers** | Only 30% heterogeneous (10/33) | Role specialization unused | Task-role planners |
| **Perception** | Bounding-box detection | No scene understanding | Scene graphs, relations |
| **Feedback** | Separate modalities | No fusion at control rate | Multimodal integration |
| **Metrics** | Isolated success rates | Hard to compare | Common benchmarks |

---

## 7. Key Findings & Insights

### 7.1 Major Contributions of This Review

**1. Comprehensive Taxonomy:**
- Systematic classification of agricultural tasks
- Platform design trade-offs clearly identified
- Control strategies categorized by application

**2. Gap Analysis:**
- Quantified validation gap (48.4% lab-only)
- Identified coordination gap (67% not truly bimanual)
- Revealed gripper design gap (70% homogeneous despite different roles)

**3. Future Roadmap:**
- Four critical research fronts identified (role allocation, force sharing, feedback fusion, metrics)
- Scene understanding highlighted as key enabler
- Specific technical solutions proposed (LLM planners, object-level impedance, scene graphs)

### 7.2 Critical Insights

**Insight 1: Configuration Follows Geometry**
- NOT one-size-fits-all
- Anthropomorphic for cluttered orchards
- Cartesian for straight rows
- SCARA for flat benches

**Insight 2: Strategy Determines Grippers**
- Bimanual control → Heterogeneous grippers (complementary roles)
- Goal-coordinated → Homogeneous grippers (parallel work)
- Most current systems underutilize heterogeneous designs

**Insight 3: Coordination Beyond Parallel**
- Most systems: Two arms work independently in separate zones
- True bimanual: Arms perform complementary tasks toward common goal
- Gap: Only 33% of studies implement true coordination

**Insight 4: Multimodal Sensing Essential**
- LiDAR: Shared geometric map
- Cameras: Semantic recognition
- Tactile/F-T: Force loop closure
- Fusion at control rate: Still rare but critical

**Insight 5: Scene Understanding is Key**
- Current: Bounding-box detection insufficient
- Needed: Relational reasoning (how objects depend on each other)
- Solution: Agricultural scene graphs + relation-aware planners

### 7.3 Practical Recommendations

**For Researchers:**

1. **Focus on field validation**
   - Move beyond lab/simulation (currently 48.4% lab-only)
   - Test in real agricultural environments
   - Multiple seasons, multiple crops

2. **Design for true bimanual coordination**
   - Not just parallel operation
   - Implement complementary roles
   - Use heterogeneous grippers when appropriate

3. **Develop scene understanding**
   - Build agricultural scene-graph datasets
   - Train relation-aware planners
   - Enable high-level reasoning

4. **Integrate multimodal feedback**
   - Fuse vision + tactile at control rate
   - Real-time force adaptation
   - Prevent damage before it occurs

**For Practitioners:**

1. **Choose platform by task geometry**
   - Unstructured orchard → Anthropomorphic arms
   - Straight rows → Cartesian gantries
   - Indoor benches → SCARA systems

2. **Match control strategy to task**
   - Need role differentiation → Heterogeneous grippers + bimanual control
   - Need parallel throughput → Homogeneous grippers + goal-coordinated

3. **Prioritize robustness**
   - Outdoor-rated sensors
   - Reliable force feedback
   - Failsafe teleoperation

**For Industry:**

1. **Invest in heterogeneous gripper development**
   - 70% current systems use homogeneous → underexploited potential
   - Task-role planners needed

2. **Develop common evaluation metrics**
   - Cycle time per cluster
   - Internal-force error
   - Cumulative bruise ratio
   - Enable direct comparison, drive progress

3. **Support agricultural scene-graph datasets**
   - Essential for scene understanding
   - Pre-training foundation models
   - Transfer learning to agriculture

---

## 8. Conclusion

This systematic review of **dual-arm manipulation in agriculture** reveals both significant progress and critical gaps in current research:

### 8.1 Progress Made

**Task Characterization:** Clear taxonomy of agricultural tasks (pruning, harvesting, transportation, etc.) with dual-arm advantages identified  
**Platform Design:** Comprehensive analysis of UGVs, UAVs, manipulator configurations, grippers, and sensors  
**Control Methods:** Motion planning, harvest ordering, tactile feedback, and coordination strategies documented  
**Field Demonstrations:** Several successful systems deployed (grape, tomato, apple, pear harvesting)

### 8.2 Critical Gaps

❌ **Limited Field Validation:** 48.4% of studies validated only in lab/simulation  
❌ **Underutilized Coordination:** Only 33% implement true bimanual cooperation (rest are parallel)  
❌ **Homogeneous Gripper Dominance:** 70% use identical grippers despite different task roles  
❌ **No Scene Understanding:** Current systems rely on bounding-box detection, lack relational reasoning  
❌ **Fragmented Evaluation:** No common metrics for comparison across studies

### 8.3 Path Forward

The review identifies **four intertwined challenges** that must be addressed to achieve fully autonomous dual-arm field robots:

1. **Task-semantic role allocation** using LLM planners
2. **Object-level force sharing** with adaptive impedance control
3. **Multimodal feedback fusion** at control-loop rates
4. **Shared evaluation metrics** for systematic comparison

**Key Enabler:** **Agricultural scene understanding** through scene graphs, relation-aware planning, and multimodal feedback loops.

### 8.4 Impact

This review provides **scalable and extensible insights** for:
- **Researchers:** Identify research gaps, promising directions, technical solutions
- **Practitioners:** Platform selection, control strategy choice, deployment considerations
- **Industry:** Investment priorities, technology readiness, ROI expectations

**Vision:** Fully autonomous dual-arm agricultural robots that:
- Understand complex scenes relationally
- Coordinate forces at object level
- Fuse multimodal feedback in real time
- Operate reliably across seasons and crops

**Timeline to Deployment:**
- **Near-term (1-3 years):** Improved grippers, better planners, field validation
- **Mid-term (3-5 years):** Scene understanding, multimodal fusion, true coordination
- **Long-term (5-10 years):** Fully autonomous systems, human-robot collaboration, commercial deployment

---

## 9. References (Key Papers)

**Platform Design:**
- [13] Korayem et al., "Dual-arm mobile manipulators for fruit-picking and pruning," *Computers and Electronics in Agriculture*, 2014
- [20] Li et al., "Multi-arm robot system for efficient apple harvesting," *Computers and Electronics in Agriculture*, 2023
- [35] Wu et al., "Premium tea-picking robot with deep learning," *Applied Sciences*, 2024

**Harvesting:**
- [21] Yoshida et al., "Automated harvesting by dual-arm fruit harvesting robot," *ROBOMECH Journal*, 2022
- [23] Stavridis et al., "Bimanual grape manipulation for robotic harvesting," *IEEE/ASME Transactions on Mechatronics*, 2025
- [41] Xiao et al., "Dual-arm cooperation for robotic harvesting tomato," *Robotics and Autonomous Systems*, 2019

**Transportation:**
- [49] Kim et al., "Goal-conditioned dual-action imitation learning for dexterous manipulation," *IEEE Transactions on Robotics*, 2024
- [55] Kim et al., "Enhancing payload capacity with dual-arm manipulation," *Journal of Mechanisms and Robotics*, 2021

**Control:**
- [65] EDDS bi-RRT algorithm for motion planning
- [67] Lammers et al., "Dual-arm robotic apple harvesting system," *Computers and Electronics in Agriculture*, 2024
- [79] Nazari et al., "Deep functional predictive control for robot pushing," *IEEE/RSJ IROS*, 2023

**Coordination:**
- [2] Smith et al., "Dual arm manipulation—A survey," *Robotics and Autonomous Systems*, 2012
- [81] Yoshikawa & Zheng, "Coordinated dynamic hybrid position/force control," *International Journal of Robotics Research*, 1993

**Scene Understanding:**
- [92] Scene understanding for dual-arm harvesting (Xiao et al.)
- [97] Duan et al., "Multimodal graph inference network for scene graph generation," *Applied Intelligence*, 2021

---

## 10. Appendix: Research Methodology

### Review Protocol
- **Databases:** ScienceDirect, Scopus, Web of Science, Springer Link, Wiley, Google Scholar
- **Search terms:** "dual-arm agricultural robot," "bimanual manipulation agriculture," "goal-coordinated harvesting"
- **Papers reviewed:** 33 (after exclusion criteria)
- **Time period:** 2009-2025 (emphasis on 2020-2025)

### Exclusion Criteria
- Not agricultural application
- Single-arm only
- No experimental validation
- Not peer-reviewed

### Four Research Perspectives (RPs)
- **RP1:** Tasks addressed by dual-arm platforms
- **RP2:** Platform components (base, arms, grippers, sensors)
- **RP3:** Control mechanisms (planning, coordination, feedback)
- **RP4:** Challenges and future directions

---