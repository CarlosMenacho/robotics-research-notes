
# Occlusion aware Deep Reinforcement Learning using mobile manipulators

- In the field of the robotics harvesting, occlusion is one the main challenges that robots are currently facing. The robot need to navigate, 
- Lower success rate for partially or fully occluded fruits

---
# Precious works
- [[Find the Fruit Designing a Zero-Shot Sim2Real DeepRL Planner for Occlusion Aware Plant Manipulation]] proposes a RL agent simulated in Isaac Lab. Its main task is to find the occluded fruit and then deploy this knowledge for a zero-shot physical robot. For this, they defined a well-structured policy during training. Achieving 86% on real-world tests


---

# Some proposals 

1. End to end RL agent capable of harvesting tasks in cluttered environment. Many works try to solve Sim2Real problem. 

	Challenges
	- Well defined reward for the entire agent's trajectory
	- Simulation environment that emulate plant's branches behavior
	- Physical robot if possible
	- Domain Randomization (environment settings)