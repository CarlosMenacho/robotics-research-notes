
- Robot interact with deformable plant to Reveals hidden objects using multimodal observations
- Decouples kinematics planning problem
- Achieves 86.7% of success on real-world trials
- The policy is trained entirely in simulation using a manually constructed generic plant model, comprising a central stem and flexible branches in NVIDIA Isaac Lab
- Proximal Policy Optimization PPO learns a control policy through direct interaction with sim env
- Combines extensive domain randomization with an impredance-controlled compliance layer, decoupling high-level motion planning 

# Introduction

- Agricultural harvesting presents greater challenges than industrial or predicted environments 
- Agricultural envs have dynamic, cluttered and flexible structures that robot should manage. 
- Without real-time sensory guidance, such strategies often lead to inefficient fruit searching. 
- The limitations highlight the need for adaptive, vision-guided, occlusion-aware manipulation strategies capable of interacting with complex, deformable plant structures.
- Deformable objects introduce unpredictable  behavior which alters the scene's geometry, perception input, and contact forces.
- In agriculture, some works are focused on:
	- removing individual leaves obstructing a known target,  
	- use self-supervised learning to predict the space revealed by a given action
- In contrast, an end-to-end learned policy, if trained and transferred successfully, could handle occlusions reactively and adaptively without relying on brittle intermediate representations. 
- RL in simulation offers a promising alternative by enabling agent to acquire adaptive behaviors
- RL agents remains a challenge
	- Agents often struggle when interacting with highly deformable object or dynamic occlusions
	- Policies trained on sim,  could be sensitive to discrepancies in perceptual and mechanical properties between simulation and real-world (sim2real transfer)


# Preliminaries
- RL is a framework where an agent learns to make sequential decisions by interacting with its env
- Modeled as a Markov Decision Process $(S, A, P, R, \gamma)$ 

# Methodology

- Adopt hierarchical control strategy to handle these interactions that decouples high-level planning from low-level actuation

## Reinforcement Learning Problem formulation

### State
- $s_t$ includes RGBA-D image $I_{RGBA-D, t}$ (rgb image, alpha fruit mask, depth) 
- Joint angles for the waist $(J_{b,t})$ and the left arm $(J_{i,t})$ 
- End-effector position $(EE_{pos,t})$ 

$$
s_t = [I_{RGBA-D, t}, J_{b,t}, J_{i,t}, EE_{pos, t}]
$$
### Action

- Mainly, joint position
$$
a_t = [\nabla j_{1-5}, \nabla j_b]
$$
### Reward

- Self-collision penalty, $r_{sc} = -5$ ; 0 otherwise
- Occlusion $r_{occ} = (1 - \frac{occluded}{ fruit pixels}) \times k$    ; k is the scaling constant controlling the occlusion reward
- Full visible reward $r_{fv}=3$ 
- Sustained visibility $r_{sus} = r_{sv}$ if full visibility is maintained > 10 steps; 0 otherwise
- Post-visibility penalty $r_{pv}= mask \times action$ magnitude, penalizes post-visibility motion


**By the end of training, the agent learns behaviors that move the plant’s stem to reveal the fruit and then stop **

Domain randomization is applied to improve robustness, including 360° variations in plant orientation and perturbed lighting conditions. 


## Evaluation Metrics

1. Success rate
2. Steps to goal
3. Generalization

## Training
- Using binary mask, for the fruit, improve and accelerate the training process
	- Deployment and evaluation were made with no mask
- Each episode, begins with randomized robot configuration and fruit location
- They apply domain randomization over physical properties (torque), visual conditions (lighting, textures) and sensor noise. **this prevents overfitting** 
- After 50 million simulation steps, the agent converges to a policy for the task instead of memorizing specific geometry



