
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
- The integration of RL with DNN marks a significant breakthrough in the field, as it leverages the powerful representation capabilities of DNNs, giving the paradigm of DRL