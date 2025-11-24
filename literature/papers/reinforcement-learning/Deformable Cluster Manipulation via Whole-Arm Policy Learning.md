
- Proposes model-free policies integrating two modalities: 3D point clouds, propioceptive touch indicators (emphasising manipulation)
- Proposes a novel context-agnostic acclusion heuristic to clear deformables from a target region for exposure tasks
- Also, perform zero-shot Sim2Real policy transfer. 
- Their approach is to accomodate stiff resistance from multiple branches

---

# Introduction

- Clusters of deformables (cables, branch tree) have imprecise physics knowledge
- The resulting chaotic dynamics render model-based learning infeasible, while model-free solutions have high sample complexity 
- Reproducing Kernel Hilbert Space (RKHS) distributional representation of the cluster state by point cloud samples
- three skills:
	- Clearing Power Lines: adding a constrain (power line)
	- Autonomous inspection
	- Agricultural Exposure


# Approach

- tree-branch geometry is challenging, well studied in botanical literature under the L-system paradigm
- Can be exported to 3D physics simulators 

## Policy Learning

- Occlusion task as a distrete-time MDP. Agent learn a stochastic policy $\pi_w(a|o)$  
- MDP is defined as $(S, A, P_a, R_a, \beta)$; $\beta$ is the discount factor
- The agent aims to maximize

$$
\mathbb{E}_{\pi} [ \sum_{t=0}^{T-1} \beta^t R_a(s_t,a_t)]
$$
- The algorithm is Proximal Policy Optimization

### Kernel Mean Embedding
- RKHSa kernel mean operator lifts a probability function $\mathbb{P}$ to a single mean function in $\mathcal{H}$  
- The lifted kernel is defined as:
$$
\varphi(\mathbb{P}) = \mu_{\mathbb{P}} := \mathbb{E}_{x \sim \mathbb{P}} [k(. , x)] = \int k(.,x)\delta \mathbb{P} 
$$
- Average kernel similarity of parameter x to all domains points according to distribution P in RKHS
- Random Fourier Features significantly reduces the computational overhead
### Distribution Embedding for Deformables

- All observations are captured of four independent point clouds $P^{rob}, P^{clr}, P^{wbr}, P^{zbr}$ representing the robot clearance region
- a distributional interpretation is invariant to point permutations and robust to the high noise content characteristic of streaming point clouds
- Speed advantage stem from formulation in [kernel mean embeding][#Kernel Mean Embedding] 

### Occlusion Heuristics

- occlusion problem is determining an effective yet simple measure to capture the temporal progression in occlusion level from manipulation 
- Common metrics include: visible surface ratio, pixel counts, or labelled occlusion levels. (depends of prior knowledge) 
- Given a segmented point clouds $P^{(1)} \in \mathbb{R}^{N_1 \times 3}$ , $P^{(2)} \in \mathbb{R}^{N_2 \times 3}$  
- First define a pairwise distance matrix $D \in \mathbb{R}^{N_1 \times N_2}$ 
- it is the fraction of point pairs (from different groups) that breach a proximity threshold to the total number of pairs in the neighbourhood, where both the threshold dth and the neighbourhood size k are tunable

### Observation & Reward space

- Propioceptive -> Joints pose, vel, EE quaternion
- KME -> whole branches, zoomed-in branches, clearance region, robot
- Touch -> contact indicator (y/n)
- 