

-  Optimal viewpoint planning for fruit picking uses several DL architectures in order to leverage learning-based viewpoint planning methods for picking fruits. Nevertheless, using this approach requires high computation for inference that enables real-time processing. 
- for sure, it depends of the camera's technology, generally require depth estimations,




--- 
# Some proposals
1. Combine learning-based and geometric-based viewpoints planning into a unified DL architecture, in order to predict a from a single-image to geometric transformation for harvesting fruit

	**Challenges:** 
	- Camera technology
	- Real-time, perhaps the DL architecture will require large VRAM. (only static images)
	- Specialized dataset, need to search over internet
	
2. RL agent to formulate a well-defined reward in order use a mobile manipulator using data fusion like depth, RGB, 
	1. The robot learns to harvest
	**Challenges**
	-  simulation environment (perhaps IsaacSim), need more research
	- High computation
	
3. Related to (1.) define a unsupervised DL model to predict a point for harvesting 