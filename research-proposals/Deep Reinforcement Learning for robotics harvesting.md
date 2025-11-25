
# Occlusion aware Deep Reinforcement Learning using mobile manipulators

- In the field of the robotics harvesting, occlusion is one the main challenges that robots are currently facing. The robot need to navigate, 
- Lower success rate for partially or fully occluded fruits

---
# Precious works
- [[Find the Fruit Designing a Zero-Shot Sim2Real DeepRL Planner for Occlusion Aware Plant Manipulation]] proposes a RL agent simulated in Isaac Lab. Its main task is to find the occluded fruit and then deploy this knowledge for a zero-shot physical robot. For this, they defined a well-structured policy during training. Achieving 86% on real-world tests
- [[Deformable Cluster Manipulation via Whole-Arm Policy Learning]] proposes a RL agent for three tasks, clearing power lines, autonomous inspection, agricultural exposure. Everything runs via 3D point cloud and touch indicators. Also proposes a novel occlusion Heuristics for measuring fruit occlusion. Their algorithm uses Proximal Policy Approximation from RL games
- [[SplatSim Zero-Shot Sim2Real Transfer of RGB Manipulation PoliciesUsing Gaussian Splatting]] proposes a zero-shot transfer learning using using RGB manipulation by Gaussian Splatting rendering. Transforms frames should be alighted in order to enable RGB with multi-viewpoint using external cameras. 
- [[Zero-Shot Sim-to-Real Reinforcement Learning for Fruit Harvesting]] trains an agent for strawberry harvesting. Their results achieve 50% of success rate. They aim to solve sim2real problem by domain randomization techniques and RL emploting Dormant Ratio Minimization. 
- [[Coverage path planning for kiwifruit picking robots based on deepreinforcement learning]]


---

# Some proposals 

1. End to end RL agent capable of harvesting tasks in cluttered environment. Many works try to solve Sim2Real problem. 

	Challenges
	- Well defined reward for the entire agent's trajectory
	- Simulation environment that emulate plant's branches behavior
	- Physical robot if possible
	- Domain Randomization (environment settings)