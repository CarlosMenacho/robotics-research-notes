
---
- Proposes DRL for path planning  for kiwifruit picking robot coverage.
- Uses LIDAR to collect the environmental point cloud information and to construct 2D grid map
	- fruit coordinate is collected in real-time
	- area division algorithm  is proposed which converts traditional grid-based coverage path planning into Travelling Salesman TSP
- Propose an improved  re-DQN algorithm to solve the traversal order of each region. 
- 31.6% shorter than that of the boustrophedon algorithm 
- Contribs:
	- fruit coordinate projection method to obtain distribution of kiwifruit
	- kiwifruit region partitioning algorithm (grid-based method to  TSP)
	- re-DQN 

---


- Covered path planning (CPP) is one of the key technologies of kiwifruit picking robots. 
- Reinforcement learning is a self-optimizing algorithm that can learn experience through a training process. 

---
## Materials and methods

- YOLOv4 is used to identify the kiwifruit via a real-sense camera and then obtain coordinate pose
### 1. kiwifruit coordinate projection
- Uses a coordinate system between camera and system coordinate 
- The mobile platform obtains a fruit projection effect in two dimensional map
### 2. Kiwifruit area division
- 