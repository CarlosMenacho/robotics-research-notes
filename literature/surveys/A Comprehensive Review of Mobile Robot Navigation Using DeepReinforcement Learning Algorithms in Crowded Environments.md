
## Introduction

- The key of the robotic navigation relies on the chosen navigation method. 
- Navigation's objective is move the robot from a initial position to predefined goal location with minimal external intervention, emphasizing the importance of safe and efficient movement. 
- DRL (Deep Reinforcement Learning) using Recurrent networks RDNN, enhances capabilities in motion planning, collition avoidance and training duration.
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
- The calculation of the return value $G_t$ is determined 