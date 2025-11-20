## Overview

This survey provides the first systematic, taxonomy-oriented review of large Vision-Language Model (VLM)-based Vision-Language-Action (VLA) models for robotic manipulation. It addresses critical gaps in the field by consolidating recent advances, resolving inconsistencies in existing taxonomies, and providing a comprehensive framework for understanding this emerging paradigm.

---

## Introduction 
- Traditional manipulation methods have been predominantly anchored in meticulously engineered control policies and rigidly predefined task specifications.
- thereby exposing inherent limitations in their scalability and generalization capabilities.
- It enables robots to interpret high-level human instructions, generalize to unseen objects and scenarios, reason about complex spatial relationships, and perform sophisticated manipulation tasks in dynamic, unstructured environments.

## Key Contributions

### 1. Longitudinal Synthesis
- Systematic review of VLM evolution
- Technical advancements in manipulation learning
- Emergence of the large VLM-based VLA paradigm
- Development of monolithic and hierarchical models

### 2. Cross-Cutting Synthesis
- Fine-grained comparative taxonomy
- Structural and functional analysis
- Advanced research frontiers
- Dataset and benchmark categorization

---

## Definition of Large VLM-based VLA Models

A large VLM-based VLA model must satisfy two conditions:

1. **Leverages a large VLM** to understand visual observations and natural language instructions
2. **Performs reasoning processes** that directly or indirectly serve robotic action generation

---

## Architectural Taxonomy

### Monolithic Models

Integration of perception, language understanding, and action generation within unified architectures.

#### Single-System Models

**Unified architecture** integrating both environmental comprehension and action generation.

##### Classic Paradigm: Autoregressive Decoding
- **RT-1**: Introduces Transformer for VLA with discrete action tokens
- **RT-2**: Co-fine-tunes on internet-scale vision-language and robot trajectory data
- **RT-2-X**: Improves cross-robot skill transfer using Open X-Embodiment dataset
- **OpenVLA**: 7B open-source model with SigLIP + DINOv2 encoders

##### Model Performance Enhancement

**Enhancing Perception Modalities:**
- **3D Perception:**
  - **Leo Agent**: Object-centric 3D point clouds with PointNet++
  - **SpatialVLA**: Ego3D position encoding from depth estimation
  - **BridgeVLA**: 3D reconstruction with orthographic projections

- **4D Perception:**
  - **TraceVLA**: Motion-point trajectory overlays for spatiotemporal understanding
  - **4D-VLA**: 3D coordinates + memory bank sampling for temporal reasoning
  - **ST-VLA**: Spatial traces combining temporal and depth information

- **Tactile & Auditory:**
  - **VTLA**: Vision-Tactile-Language-Action with tactile encoding
  - **VLAS**: Speech integration via Whisper Encoder
  - **FuSe**: Multimodal alignment with limited annotations

**Enhancing Reasoning Capabilities:**
- **ECoT**: Chain-of-thought reasoning with task planning
- **CoT-VLA**: Visual chain-of-thought with subgoal observation prediction
- **LoHoVLA**: Hierarchical closed-loop control for long-horizon tasks
- **ReFineVLA**: Selective transfer fine-tuning for multimodal understanding

**Enhancing Generalization:**
- **UniAct**: Universal action codebook for cross-embodiment learning
- **ReVLA**: Reversible training to preserve visual generalization
- **HybridVLA**: Integrates diffusion and autoregressive decoding
- **VOTE**: Adaptive ensemble voting for robust actions
- **WorldVLA**: Integrates action and world models
- **UnifiedVLA**: World model as post-training task
- **UP-VLA**: Implicit physical dynamics learning

##### Inference Efficiency Optimization

**Architectural Optimization:**
- **MoLe-VLA**: Spatio-Temporal Aware Router for dynamic layer selection
- **DeeR-VLA**: Dynamic early-exit mechanism
- **RoboMamba**: Mamba architecture for 3× faster inference

**Parameter Optimization:**
- **BitVLA**: 1-bit weight VLA model with BitNet b1.58
- **NORA**: Compact high-quality VLM with FAST+ tokenizer

**Inference Acceleration:**
- **Parallel Decoding**: RoboFlamingo, OpenVLA-OFT, PD-VLA
- **Spec-VLA**: Speculative decoding with relaxed acceptance
- **FlashVLA**: Skip inference when environment is stable

#### Dual-System Models

**System 2 (VLM Backbone)**: Slower, reflective reasoning  
**System 1 (Action Expert)**: Fast, reactive processing for actions

##### Cascade-based Methods

**Serial processing** where VLM backbone cascades to action expert.

DiT (Diffution Transformer architecture)

- **Separate Action Expert:**
  - **CogACT**: DiT-based action model with ensemble algorithm
  - **GR00T N1**: Dual-system for humanoid robots
  - **DP-VLA**: Behavioral Cloning transformer
  - **HiRT**: VLM at lower frequency + lightweight action strategy
  - **TriVLA**: Three-module system (video generation + VLM + diffusion)
  - **GF-VLA**: Information-based scene graphs for dual-arm control
  - **RationalVLA**: Learnable latent interface with instruction rejection
  - **VQ-VLA**: Convolutional residual VQ-VAE

- **Unified Action Expert:**
  - **Fast-in-Slow**: Action expert uses final transformer blocks of VLM

##### Parallel-based Methods

**Parallel operation** with interaction between VLM backbone and action expert.

- **Shared-attention Architecture:**

  - **π0**: Flow-matching action expert with shared self-attention
  - **ForceVLA**: FVLMoE module for force modality
  - **OneTwoVLA**: Mode switching between reasoning and acting
  - **π0.5**: Subtask prediction layer before π0
  - **π0.5-KI**: Prevents gradient flow to VLM during training
  - **π0-FAST**: DCT-driven action tokenization
  - **villa-X**: Joint latent and robot action diffusion
  - **Tactile-VLA**: Tactile sensing integration

- **Cross-attention Architecture:**
  - **SmolVLA**: Frozen lightweight VLM + Flow Matching Transformer
  - **GR-3**: Unifies vision-language with trajectory learning

---

### Hierarchical Models

**Explicit decoupling** of planning from policy execution via interpretable intermediate representations.

Particularly in scenarios where **long-horizon reasoning**, spatial abstraction, or action
decomposition is required.
#### Key Characteristics:
1. **Structured intermediate outputs**: Keypoints, affordance maps, trajectory proposals, subtasks, programs
2. **Decoupled training**: Independent optimization through specialized loss functions or API-mediated interactions

#### Planner-Only Models

Models that generate intermediate representations for existing policies.

##### Program-based Methods

- **Robot-executable programs:**
  - **Chain-of-Modality**: Multimodal prompting for Python program generation
  - **Instruct2Act**: Python code invoking APIs for robot control

- **Auxiliary programs:**
  - **ROVI**: Auxiliary programs describing potential actions
  - **ReLEP**: Task decomposition into basic skills from skill library

##### Keypoint-based Methods

predict salient points in an observation, typically corresponding to interactive regions that the gripper should reach
- **MoManipVLA**: Key waypoint generation via VLA model
- **RoboPoint**: Visual keypoints from natural language instructions
- **ManipLVM-R1**: GRPO training for affordance area and trajectory prediction
- **RoboBrain**: Task planning + affordance perception + trajectory estimation
- **RoVI**: Sketch-based interface with YOLOv8 extraction

##### Subtask-based Methods

Subtask-based hierarchical models bridge the planner and policy with instructions.

- **PaLM-E**: Unifies VQA with robot command generation
- **Embodied-Reasoner**: Observation–Thought–Action trajectories
- **Reinforced Planning**: SFT + GRPO reinforcement fine-tuning
- **Embodied-R**: VLM perception + small LM reasoning
- **ViLA**: GPT-4V as external planner

#### Planner+Policy Models

Complete systems with both planning and execution modules.

##### Keypoint-based Methods

- **HAMSTER**: Predicts trajectory keypoints with gradient color path overlay
- **ReKep**: DINOv2 + SAM for keypoint proposals → GPT-4o cost functions
- **A0**: Affordance-aware hierarchy with contact points and motion

##### Subtask-based Methods

- **HiRobot**: Open-ended instruction decomposition into atomic commands
- **DexVLA**: VLM planner + diffusion-based action policy
- **PointVLA**: Point cloud encoder/injector for spatial perception
- **RoBridge**: Invariant operable representation for primitive actions
- **SkillDiffuser**: High-level skill prediction + low-level diffusion
- **RoboMatrix**: Three-layer hierarchy (scheduling, skill, hardware)
- **HiBerNAC**: Asynchronous hierarchical framework
- **MALMM**: Three-agent system (planner, supervisor, coder)

---

## Core Advantages of Large VLM-based VLA Models

### 1. Open-World Generalization
- Handle novel objects and unseen environments
- Zero-shot and few-shot capabilities
- Transfer from web-scale pretraining to robotic control

### 2. Hierarchical Task Planning
- Decompose complex instructions into executable subtasks
- Long-horizon task completion
- Interpretable reasoning chains

### 3. Knowledge-Augmented Reasoning
- Leverage internet-scale vision-language knowledge
- Semantic understanding beyond pattern matching
- Contextual awareness and commonsense reasoning

### 4. Rich Multimodal Fusion
- Unified embedding space for vision, language, proprioception
- Token-level integration across modalities
- Extensible to tactile, audio, depth sensors

---

## Advanced Research Directions

### Reinforcement Learning-based Methods

**Addressing reward sparsity:**
- **VLA-RL**: Robotic Process Reward Model (RPRM)
- **ReWiND**: Visual similarity-based progress rewards
- **Grape & TGRPO**: VLM-prompted feedback rewards

**Hybrid training:**
- **ReWiND**: Offline IQL + Online SAC
- **HIL-SERL**: Human-in-the-loop intervention
- **ConRFT**: Two-phase Cal-ConRFT + HIL-ConRFT

**Data generation:**
- **RLDG**: Expert policy distillation via HIL-SERL
- **iRe-VLA**: Iterative RL + SFT expansion

### Training-Free Methods

**Efficiency without retraining:**
- **FlashVLA**: Skip inference when stable
- **EfficientVLA**: Layer pruning + token filtering
- **VLA-Cache**: Cached key-value reuse
- **SP-VLA**: Token pruning + action-aware scheduling
- **PD-VLA**: Parallel fixed-point iteration
- **FAST**: DCT compression for action sequences
- **RTC**: Dynamic control frequency adjustment

### Learning from Human Videos

**Cross-domain knowledge transfer:**
- **Human-Robot Semantic Alignment**: Vision encoder alignment
- **UniVLA**: Task-centric latent actions from videos
- **LAPA**: VQ-VAE quantized latent actions
- **VPDD**: Discrete diffusion over video tokens
- **3D-VLA**: Human-object interaction videos
- **Humanoid-VLA**: Pose-recovered motion trajectories

### World Model-based VLA

World models, characterized by their ability to learn compact latent representations of environment dynamics, have emerged as powerful tools for enabling predictive reasoning and long-horizon planning

**Predictive reasoning:**
- **WorldVLA**: Autoregressive action world model
- **World4Omni**: Subgoal image generation
- **3D-VLA**: Future goal image and point cloud prediction
- **RIGVid**: Diffusion-based video generation + VLM filtering
- **V-JEPA 2-AC**: Latent action-conditioned world model

---

## Key Characteristics

### Multimodal Fusion

**Shared Embedding Space:**
- Semantically aligned latent space for vision-language
- Unified representation reduces semantic drift
- Tight grounding between perception and commands

**Token-Level Integration:**
- Discretize continuous modalities into token sequences
- Fine-grained cross-modal coordination
- Dynamic attention across perception-action cycle

**Comprehensive Modal Compatibility:**
- Seamless accommodation of diverse sensors
- Modality-agnostic semantic alignment
- Add new modalities without full retraining

### Instruction Following

**Semantic Instruction Grounding:**
- Fluid, context-sensitive comprehension
- Rich world knowledge for interpretation
- Generalization beyond fixed templates

**Task Decomposition:**
- Hierarchical breakdown for long-horizon tasks
- Subtask generation in natural language
- Continuous alignment throughout execution

**Explicit Reasoning:**
- Chain-of-thought integration
- Visual goal prediction
- Iterative plan refinement

### Multi-Dimensional Generalization

**Cross-Task:**
- Zero-shot and few-shot transfer
- Novel object handling
- Unseen scenario adaptation

**Cross-Domain:**
- Web text + simulation + real robot co-training
- Out-of-distribution deployment
- Multi-stage task completion (>90% success)

**Cross-Embodiment:**
- Transfer across robot platforms
- Sim-to-real generalization
- Hierarchical architectures for diverse morphologies

---

## Datasets and Benchmarks

### Real-world Robot Datasets

**Core Datasets:**
- **BC-Z**: 100 tasks with expert demonstrations
- **RT-1**: 700+ daily activities
- **RT-2**: Web-scale vision-language data integration
- **BridgeData V2**: Cross-domain language-annotated demonstrations
- **RH20T**: 147 tasks with one-shot learning capability
- **DROID**: 564 tasks "in the wild"

**Large-Scale:**
- **Open X-Embodiment (OXE)**: 1M+ demonstrations, 22+ platforms, 500+ skills

**Challenges:**
- Long-tail distribution of open-world objects
- Underrepresented scenes and skills
- Need for broader, more diverse data

### Simulation Datasets and Benchmarks

**Household & Multi-step:**
- **BEHAVIOR**: Multi-step semantic control in cluttered settings
- **ALFRED**: Long-horizon egocentric language instructions
- **CALVIN**: Multi-stage unconstrained language instructions

**Tabletop Manipulation:**
- **RLBench**: RGB-D policy learning
- **RLBench2**: Bimanual manipulation
- **Meta-World**: Multi-skill scenarios
- **Franka Kitchen**: Complex kitchen tasks
- **LIBERO**: Multi-skill with language annotations

**Advanced Benchmarks:**
- **MIKASA-Robo**: Memory-centric partial observability
- **SIMPLER**: Sim-to-real gap reduction
- **Habitat / SAPIEN**: High-fidelity 3D environments
- **THE COLOSSEUM**: Robustness under distribution shifts
- **VLABench**: Universal language-conditioned manipulation

### Human Behavior Datasets

**Egocentric Video Corpora:**
- **Ego4D, Ego-Exo4D**: Diverse daily activities
- **EgoPlanBench, EgoVid-5**: First-person perspectives
- **EPIC-Kitchens, COMKitchens**: Fine-grained cooking tasks

**Reasoning-Oriented:**
- **EgoVQA, EgoTaskQA**: Spatial, temporal, causal reasoning

**Manipulation-Focused:**
- **EgoDex**: 829 hours with 3D hand tracking
- **DexCap, EgoMimic, PH2D**: Fine-grained hand-object coordination

**Scalable Generation:**
- **R2R2R**: Smartphone scans + human videos pipeline

### Embodied Datasets and Benchmarks

**Question-Answering:**
- **EmbodiedQA, IQUAD**: Vision-language-navigation integration
- **MT-EQA**: Multi-target reasoning
- **MP3D-EQA**: 3D point cloud inputs
- **EQA-MX**: Non-verbal multimodal cues
- **OpenEQA**: Open-ended functional reasoning

**Plan Execution:**
- **LoTa-Bench**: Direct executability evaluation

---

## Future Directions

### 1. Datasets and Benchmarking

**Current Gaps:**
- Simulation reality gap
- Limited real-world data diversity
- Focus on short-horizon pick-and-place tasks
- Simple success rate metrics

**Future Needs:**
- Large-scale real-world data acquisition
- Long-horizon planning tasks
- Mobile manipulation scenarios
- Multi-agent collaboration benchmarks
- Richer metrics: subtask success, time efficiency, robustness

### 2. Memory Mechanisms and Long-Term Planning

**Current Limitations:**
- Frame-by-frame reasoning
- Short-sighted behavior
- Limited historical context

**Future Directions:**
- Forward-looking planning architectures
- Episodic memory integration
- Contextual memory grounding
- Goal-driven action sequences

### 3. 3D and 4D Perception

**Current State:**
- Primarily 2D static visual inputs
- Limited spatial reasoning

**4D Perception Requirements:**
- Depth and point cloud integration
- Multi-modal fusion into unified representations
- Temporal context embedding
- Real-time replanning capabilities

### 4. Mobile Manipulation

**Challenges:**
- Synergistic locomotion + manipulation
- Tightly coupled perception and control
- Simultaneous navigation and interaction

**Solutions:**
- Integrated policies (not separate stages)
- Adaptive priority balancing
- Unified learning frameworks

### 5. Multi-Agent Cooperation

**Requirements:**
- Intention negotiation
- Teammate action adaptation
- Multi-step joint reasoning
- Communication protocols

**Approaches:**
- Emergent dialogue systems
- Shared world models
- Flexible subtask allocation
- Cohesive team coordination

### 6. Lifelong Learning in Open-World

**Challenges:**
- Catastrophic forgetting
- Unfamiliar object handling
- New experience incorporation

**Solutions:**
- Continual skill acquisition mechanisms
- Growing memory structures
- Self-organized experience abstraction
- Exploration and feedback loops

### 7. Model Efficiency

**Deployment Challenges:**
- Computational costs
- Memory requirements
- Real-time inference latency
- Resource-constrained platforms

**Optimization Strategies:**
- Task-aware dynamic token pruning
- Asynchronous inference
- Hardware-friendly quantization
- Compression without accuracy loss

---

## Comparison: Monolithic vs. Hierarchical

| Aspect                      | Monolithic Models                       | Hierarchical Models                        |
| --------------------------- | --------------------------------------- | ------------------------------------------ |
| **Architecture**            | Unified/dual-system pipeline            | Explicit planning-execution separation     |
| **Integration**             | Tight coupling, end-to-end optimization | Modular, independent components            |
| **Intermediate Processing** | Implicit reasoning in latent space      | Explicit, interpretable representations    |
| **Interpretability**        | Opaque internal processes               | Human-understandable outputs               |
| **Flexibility**             | Streamlined, minimal decomposition      | Easy component replacement                 |
| **Training**                | Joint optimization                      | Decoupled specialized training             |
| **Transparency**            | Limited external inspection             | Detailed task monitoring                   |
| **Validation**              | End-to-end evaluation                   | Independent module validation              |
| **Use Cases**               | Efficient unified learning              | Complex multi-stage tasks, safety-critical |
| **Strengths**               | Holistic learning, generalization       | Modularity, explainability, flexibility    |

**Complementary Strategies:**
- **Monolithic**: Power of unified learning, minimal manual decomposition
- **Hierarchical**: Greater transparency, modular flexibility for complex tasks

---

## Conclusion

This survey establishes the first systematic framework for understanding large VLM-based Vision-Language-Action models for robotic manipulation. By:

1. **Consolidating Recent Advances**: Comprehensive review of 140+ papers
2. **Resolving Taxonomic Inconsistencies**: Clear definitions and categorizations
3. **Mitigating Research Fragmentation**: Unified perspective across disciplines
4. **Identifying Future Directions**: Critical open challenges and opportunities

**Key Takeaways:**
- VLA models represent a transformative paradigm shift in robotic manipulation
- Integration of web-scale vision-language knowledge enables unprecedented generalization
- Both monolithic and hierarchical approaches offer complementary advantages
- Future progress requires addressing efficiency, memory, perception, and real-world deployment challenges

**Impact:**
The field is rapidly evolving toward:
- Greater cross-embodiment generalization
- Efficient real-time deployment
- Tighter coupling of reasoning and execution
- Leveraging human demonstrations at internet scale

This survey serves as a foundation for advancing embodied AI that truly unifies perception, language, and action for general-purpose robotic systems.

