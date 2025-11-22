
# - Thesis proposal

| Proposal                                                                                  | Primary Focus         | Features or                                 |
| ----------------------------------------------------------------------------------------- | --------------------- | ------------------------------------------- |
| Foundation Models for Data-Efficient Multi-Robot Agricultural Manipulation                | Data efficiency       | Agricultural VLA with few-shot coordination |
| Iterative Learning with Seasonal Memory for Long-Horizon Agricultural Multi-Robot Systems | Long-horizon learning | Multi-timescale memory architecture         |
| Sim-to-Real for Deformable Objects                                                        | Transfer learning     | Differentiable plant simulation             |

## Foundation Models for Data-Efficient Multi-Robot Agricultural Manipulation


Agricultural robots require **task-specific training for each crop variety**, making deployment prohibitively expensive. Foundation models show promise but lack agricultural-specific pre-training and multi-robot coordination capabilities [[Large VLM-based Vision-Language-Action Models for Robotic Manipulation A Survey]].  This proposal creates the first **vision-language-action foundation model for agricultural multi-robot systems**.

### some ideas
#### 1. Agricultural Vision-Language-Action Model
- Unified transformer architecture tokenizing visual observations, language instructions, proprioceptive state, and multi-agent coordination signals.
- Pre-trained on large-scale internet data + agricultural imagery + human demonstrations.
- Fine-tuned for multi-robot coordination with minimal task-specific data
#### 2. Few-Shot Multi-Robot Coordination
- Meta-learning framework enabling rapid adaptation to new crops with 10-50 demonstrations
- Language-conditioned task allocation ("Robot 1: hold branch gently; Robot 2: pick ripe fruits")
- Zero-shot generalization to different robot morphologies through embodiment tokens

## Iterative Learning with Seasonal Memory for Long-Horizon Agricultural Multi-Robot Systems

## ideas

#### 1. Episodic Memory Architecture for Agricultural Tasks
- Store successful multi-robot coordination episodes indexed by context (crop type, density, weather, growth stage)
- Retrieve relevant past experiences for current situation
- Learn when to rely on memory vs. explore new strategies
#### 2. Iterative Learning Control + Multi-Agent RL Integration
- Exploit daily field traversals for rapid improvement
- Acausal filtering between iterations (use future information from previous runs) [theory example](https://juliacontrol.github.io/ControlSystems.jl/dev/examples/ilc/)
- Multi-robot ILC: coordinate learning across robot experiences

## Safe Sim-to-Real Transfer for Multi-Robot Manipulation of Deformable Agricultural Objects

**Deformable plant mobile manipulation is the critical bottleneck** preventing sim-to-real transfer for agricultural robots. Current simulators cannot accurately model plant deformation, fruit detachment mechanics, or multi-robot contact with crops.
### ideas

- Multi-Robot Sim-to-Real with Safety Constraints
- Reality Gap Characterization via Interaction
- Multi-robot mobile learning transfer 
# - Paper proposals

- Action Space Design for Mobile Manipulator DRL, mobile manipulators have coupled base+arm dynamics: 
	- [ ] Joint-level control (wheel velocities, joint velocities) (12D continuous)
	- [ ] Task-space control (base velocity, end-effector velocity) (6D continuous)
	- [ ] Hierarchical, discrete macro-actions: "move to", "grasp", "place"
- Curriculum Learning for Coupled Mobile Manipulation, learning both navigation and manipulation simultaneously
	- [ ] May define a curriculum progression (5 stages: nav only, manipulation only,  sequential, then all together) 
	- [ ] Compare training strategies
	- [ ] Progressive curriculum through all stages
- Offline Multi-Agent RL from Suboptimal Demonstrations, collecting online RL data is slow in multi-robot systems.
	- [ ] offline dataset in Gazebo, by human teleoperation
	- [ ] Fine-tune with limited online RL
	- [ ] Multi robot navigation


--- 

# Enhanced problem statement

-  [[Perception-based problem statement]], robots currently are being applied in harvesting process, modern DL architectures are proposed to obtain fruit targets