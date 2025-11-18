
## Introduction

- The key of the robotic navigation relies on the chosen navigation method. 
- Navigation's objective is move the robot from a initial position to predefined goal location with minimal external intervention, emphasizing the importance of safe and efficient movement. 
- DRL (Deep Reinforcement Learning) using Recurrent networks RDNN, enhances capabilities in motion planning, collision avoidance and training duration.
- Since 2016, there has been an increasing inclinations implementing DRL in the research of mobile robot navigation, leading notable achievements.
- SLAM algorithms can be split into two key modules:  visual SLAM, laser SLAM.
- **Visual SLAM** involves the extraction of artificial image features. generates an obstacle map that allows the robot to understand its environment and navigate carefully from one location to another. 
- **Laser SLAM** use dense laser data to construct an obstacle map from the environment. 


## Related Work
### Primary information

- RL allows to learn from real-world experiences, aiming to formulate optimal decision-making strategies.
- In RL an "Agent" is defined as a decision-maker interacting with its surroundings, known as the "environment".
- The key target of the robot is to optimize the cumulative reward and receive a feedback signal in the form of a reward value over time. 
- The Markov Decision Process, offers a mathematical framework for representing decision-making in the environment with uncertainty influenced by the  decision-maker (agent).
- To map each state to an action, the agent relies on a policy denoted as $\pi(s,a)$, which can be either deterministic or stochastic: 
$$
\begin{cases}
a = \pi(s), & \text{if the policy is deterministic}, \\[6pt]
\pi(a|s) = P[A_t = a| S_t =s], & \text{if the policy is stochastic}.
\end{cases}
$$
- The calculation of the return value $G_t$ is determined by introducing the discount factor $\gamma \in [0-1]$  which describes the importance of the current state and the future rewards.     
- $V_\pi$ and $Q_\pi (s,a)$  represent the value function and the action-value function of state $s_t$ and taking action $a_t$ respectively.  
- We can extend the relationship between states $s=s_t$ and $s' = s_t+1$ in a recursive form based on Eqs: 
$$
G_t = r_t + \gamma' r_{t+1} + \gamma^2 r_{t+2} + ... + \gamma ^ {T -1} r _T  = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$
 $$
V_\pi(s) = E_\pi [R_t|s_t=s] = E_\pi[r_t+\gamma'r_{t+1} + \gamma^2r_{t+2} + ... + \gamma^{T-1} r_T|s_t = s]
$$
$$
Q_\pi(s,a) = E_\pi [ R_t|s_t=s, a_t=a] = E_\pi[r_t + \gamma' r_{t+1} + \gamma^2r_{t+2}+...+\gamma^{T-1}r_T|s_t=s,a_t=a] = \frac{1}{N} \sum_{n=1}^{N}R_t^n
$$
### Deep Reinforcement Learning Methods
- Traditional RL methods encounters challenges, particularly the curse of dimensionality in high-dimensional problems. 
- This occurs when the growth in the number of inputs sharply amplifies the computational requirements.
- Applying RL algorithms to discover a robust policy in a vast state space becomes more challenging. DNN is used to approximate nonlinear functions. 
- The integration of RL with DNN marks a significant breakthrough in the field, as it leverages the powerful representation capabilities of DNNs, giving the paradigm of DRL.
- There's two main approaches: **policy-based methods** and **value-based methods** 
- Policy-based methods directly employ function approximation to construct a policy network. The optimization of the policy network parameter is accomplished by leveraging the gradient direction to attain an optimized policy, with the target of optimizing the total reward value.
- Value-based methods takes an indirect approach to formulate  the agent's policy. **Achieved  through iterative updates to the value function**

### Value-based DRL Method
##### Deep Q-learning
- A solution implemented DRL model that effectively acquired control policies directly from high-dimensional sensory input. 
- Utilizing a model-based CNN trained with a modified version of Q-Learning, the system generated a value function to estimate future rewards, using raw pixels data as input. 
- To achieve optimal performance, the network ca be trained by optimizing a sequence of loss funcion $L_i(\theta_i)$  between the target value and the predicted value.  
$$
L_i(\theta_i) = E [(y_i- \hat{Q}(S_t, a_t, \theta_i) )^2]
$$
where $\hat{Q}$ represents the network, and the target value ($y_i$) for iteration $i$ is calculated as follows: 
$$
y_i = R(S_t, a_t) + \gamma max_a [\hat{Q}(S_{t+1}, a_t, \theta_{i-1})]
$$
Loss function of DQN can be calculated as: 
$$
L_i(\theta_i) = E[(R(S_t, a_t) + \gamma max_a [\hat{Q}(S_{t+1}, a_{t+1}, \theta_{i-1})] - \hat{Q}(S_t, a_t, \theta_i))^2]
$$

- Model's predictions, showed as showed as $\hat{Q}(S_{t+1}, a_{t+1})$ , correspond to the actual target, showed as $Q(S_{t+1}, a_{t+1})$, in practical scenarios. 
- The network weights  are updated after each time step, Stochastic Gradient Decent SGD is commonly employed to optimize the loss function.
#### Double DQN (DDQN) 
- Demonstrate advantages in certain navigation problems, DDQN algorithms address the overestimation issue. 
- DDQN incorporated an architecture with two networks within the target function Q.

$$
L_i(\theta_i) = E[(R(S_t, a_t) + \gamma \hat{Q} [s', argmax \hat{Q}(S', a', \theta_{i}) ; \theta_{i-1}] - \hat{Q}(S_t, a_t, \theta_i))^2]
$$

#### Policy-based DRL Method
- DQN and its variants are effective for addressing problems characterized by a high-dimensional observation space. But, these methods are constrained by a discrete and low-dimensional action spaces.
- Physical control tasks, feature ongoing and high dimensional action spaces. To tackle this challenge, one approach is to discretize the action space 

##### Deep Deterministic Policy gradient DDPG
- Is an algorithm based on policy gradient designed to directly achieve the best policies, suitable for **continuous action spaces**.
- DDPG utilizes a deterministic policy function ($a_t = \mu(S_t| \theta^\mu)$) .
- The method employs a neural network notated as $\hat{Q}$, to model both, the policy and the Q-function. 
- K samples are randomly collected from the experience pool, and the model is gently changed through gradient ascent. Loss is calculated as below: 
$$
L = \frac{1}{K} \sum_{i} (y_i - \hat{Q}(S_i, a_i | \theta^{\hat{Q}}))^2
$$
- Its important of the design a robust reward system for the intelligent agent to achieve its goals. 
- Good results in robot simulation GAZEBO. 

##### Asynchronous Advantage Actor-Critic (A3C)

- This algorithm adopts the traditional policy gradient approach to directly optimize the robot's policy. 
- In AC structure, the policy gradient algorithm functions as an actor responsible for collecting actions, while the value function method serves as a critic to evaluate those actions.
- AC architecture offers several advantages , significantly converting the series updates in policy gradient into a one-step value. This removes the necessity for data selection delays and minimizes the variance encountered by the policy gradient approach. 
- A3C are more computational economical compared to approaches such as DQN.
- The autors demonstrated its superior performance compared to existing methods. 

##### Proximal Policy Optimization PPO

 - To avoid parameter updates resulting in significant changes to the policy in a single step and to impose constrains on the size of the policy updates at each iteration, PPO is suitable. 
 - This algorithm has the capability to execute multiple epochs of mini-batch updates, enhancing the overall efficiency of utilization.
 - We can calculate the probability ratio between the new and old policies as follows: 

$$
r(\theta) = \frac{\pi_{\theta_{new}}(a|s)}{\pi_{\theta_{old}}(a|s)}
$$
We can express the objective function

$$
L(\theta) = E [r(\theta)A_{\theta_{old}}(s|a)]
$$
- This formula is intended  to enhance the action generated through the new policy compared to those produces through the old policy
- Significant improvement by the new policy may introduce  instability in the training algorithm. 
- The PPO algorithm refines the goal to achieve the new clipped surrogate objective 

## DRL-based Navigation

### Framework
- Researchers mostly have predominantly concentrated on navigation challenges within 2D spatial context. 
- There is two primary task in the mobile robot navigation field: Obstacle avoidance and Point-to-Point (P2P) movement. Agent uses mainly ultrasonic sensors, cameras, lasers, etc.
- P2P tasks involves establishing the relationship between the target place and the starting point. 
- DRL aims to guide the robot to the target position by determining the optimal policy through interactions of the robot with the environment.  Using DDPG, PPO, DQN
- Those methods  have adopted MDP to depict the navigation process where sensors observations serve as the state, aiming to optimize the expected reward of action. 
- DRL offer the benefit of operating without maps, exhibiting robust learning capabilities and demonstrated reduced reliance on sensor precision.
---
### Key Elements 
#### Action Space

- There are three types of actions considered in DRL-based mobile robot navigation.
1. Discrete actions: basic control like forward, turn left, right and backward. 
2. Continuous velocity: defining the angular and linear velocities of the mobile robot.
3. Motor Speeds commands: setting each motor velocity.
- Continuous velocity commands often necessitate a PID or alternative low-level motion controllers to formulate motion control instructions. 

#### Reward Function
- The RL agent accomplishes tasks by leveraging the reward function during training. 
- Agents typically receive positive or negative rewards when reaching the destination or facing obstacles
- Sparse reward can hinder the rapid convergence of the agent. To address this, dense reward-shaping methods are employed. 
1. Agent attains positive reward upon reaching the goal, with reward increasing as its movement closely aligns with the target.
2. Negative rewards, acting as collision penalties, are assigned when the agent collides with obstacles. or approaches them too closely
3. A penalty in the form of negative reward is applied at each time step to incentive the robot to traverse its path to the target more quickly.

#### State Space

- The exploration of mobile robot navigation research encompasses three fundamental state spaces: the starting point, target point, and obstacles.
- Some researchers choose to express the goal point by  **converting Cartesian coordinates into local path polar coordinates**, incorporating direction and distance relative to the robot, incorporating direction and distance relative to the robot.
- Obstacle state is characterized by the speed, position, and size of moving blocks at the agent level. Alternatively, sensor data could be directly treated as a sensor-level state, encompassing information such as ultrasonic/lidar ranging data.
---
## Different Kinds of Navigation

### Autonomous-based  Navigation

- The agent based on the sensor does not require pre-built maps of the environment.
- Instead, it autonomously predicts a trajectory through the environment to reach a tar- get position and dynamically responds to any unexpected obstacles encountered along the way.
- Here many approaches have been used, CNN, CNN + LSTM and sensors
- See tables and comparison on table. 

### SLAM-based Navigation

- The navigation of mobile robots has traditionally been tackled through route planning algorithms that depend on representations of the environment, typically in form of map.
- However, creating and maintaining a comprehensive map of the surroundings, especially in tasks such as surveillance, cleaning, and research, can be challenging or entail significant costs
- To address these challenges, researchers are now exploring approaches that integrate DRL algorithms with SLAM techniques to enhance the outcomes of mobile robot navigation.
- See Table

### Planning-based Navigation

- Some researchers employed a combination of geometric paths and a set of kinematic or stochastic constraints to formulate a sequence of control inputs through trajectory planning.
- The authors utilized path-planning methods to derive the geometric path, and collectively, these components effectively address the intricate challenges associated with motion planning
- See **Table**

