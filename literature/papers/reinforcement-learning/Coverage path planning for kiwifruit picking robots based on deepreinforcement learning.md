
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
- Mobile manipulator has a coverage area which fruits inside this area are marked. 
- Instead using grid-based coverage, this paper uses the fruit coordinate projection to understand the distribution of the fruit
## Coverage Path Planning

- Using the coordinates and the division of the fruit regions, traditional grid-based coverage path planning is transformed into a travelling salesman TSP (Solution problem between regions)
- DRL has advantages for solving TSP problems

### DQN
- Q-target network

$$
L(\theta) = E [(r + \gamma maxQ (s',a',\theta') - Q(s,a,\theta))^2]
$$
- And the gradient of the loss function

$$
\nabla L(\theta) = E[(r+\gamma maxQ(s',a',\theta') - Q(s,a,\theta))^2 \nabla_\theta Q(s,a,\theta)]
$$

- Q main network is updated by stochastic gradient descent

## re-DQN

- guide the model training, an empirical backtracking mechanism is set up
- See algorithm 
### Action Selection

- discrete actions with $a_i (i=1,2,3,4)$ 

### Path quality Score

- Scores the path trained in each round of training

$$
F(n) = ln\mu (l_n)^-1
$$
- where $l_n$ is the path length of the current round and $\mu$ is the adjustment factor
### Reward  function


$$

r_t =
\begin{cases}
F(n) R_{\text{reach}}, & \text{if } P_g = P_{\text{cover}} \\
0, & \text{if } P_g \ne P_{\text{cover}}
\end{cases}


$$

### Reward update

- Traditional Q-value function, works well with deterministic final stages, but CPP problems, the end point of the path can be any point
$$
Q(s_t, a_t) = Q(s_t, a_t) + \alpha [r_t + \gamma maxQ(s_{t+1}, a_{t+1} - Q(s_t, a_t))]
$$
- where $\alpha$ is the learning rate and $\gamma$ is the depreciation rate
- In response, a reward value update method is proposed
$$
Q(s_t, a_t) = \frac{L_{s_t}}{L_{path}} Q(s_t, a_t)
$$

### Training results and discussions

- Some training hyperparameters
	- training steps 20000
	- learning rate 0.1
	- loss rate 0.9
	- decay factor 0.0003
	- adjustment factor 300
- Compared with the traditional DQN al­ gorithm, the re-DQN algorithm has a shorter path length in the initial stage of training because the overall update of the reward for each round makes the model have better global correlation 