# Introduction
-  Foundation models are pre-trained on extensive internet-scale data and can be ﬁne-tuned for adaptation to a wide range of downstream tasks.
- Foundation models have the potential to unlock new possibilities in robotics domains such as autonomous driving, household robotics, industrial robotics, assistive robotics, medical robotics, ﬁeld robotics, and multi-robot systems.
- The integration of foundation models into robotics is a rapidly evolving area
- foundation models are pre-trained on extensive and diverse data, which has been proven in other domains.
- foundation models are pre-trained on extensive and diverse data, which has been proven in other domains
- Particularly relevant to robotics, multimodal foundation models can fuse and align multimodal heterogeneous data gathered from various sensors into compact homogeneous representations needed for robot understanding and reasoning
- language grounding in the 3D world, may enhance a robot’s spatial awareness by associating words with speciﬁc objects, locations, or actions within the 3D environment
- foundation models seem to offer the possibility to improve data efﬁciency

Foundation models in real-world robotics applications is still challenging
1. Data Scarcity, how to obtain internet-scale data for robot manipulation, locomotion, navigation and other tasks
2. High Variability, How to deal with the large diversity in physical environments
3. Uncertainty Quantification, how to deal with (i) instance-level uncertainty such as language ambiguity, or LLM hallucination. (ii) distribution-level, (iii)distribution-shift especially resulting from closed-loop robot deployment 
4. Safety Evaluation, test for the safety of a foundation model-based robotic system (i) prior to deployment (ii) as the model is updated throughout its lifecycle (iii) as the robot operates in its target environment  
5. Real-time performance, deal with the inference time
6. Reproducibility, how to reproduce research and benchmark robotic-specific foundation models developed on specific hardware setups
# Foundation models background 

- Acquiring, processing, and managing data can be costly.
- Additionally, training a foundation model is time-intensive, which can translate to even higher costs.

## Terminology and mathematical preliminaries

### Tokenization 

- Given a sequence of characters, tokenization is the process of dividing the sequence into smaller units, called tokens. Can be: characters, segments of words, complete word or portion of sencences. 
- Tokens are represented as 1-hot vectors of dimension equal to the size of the total vocabulary and are mapped to lower-dimensional vectors of real numbers through a learned embedding matrix
- Common tokenization, Byte-pair encoding starts with a token for each individual symbol (e.g., letter, punctuation), then recursively builds tokens by grouping pairs of symbols that commonly appear together, building up to assign tokens to larger and larger groups
### Generative Models

- A generative model is a model that learns to sample from a probability distribution to create examples of data that seem to be from the same distribution as the training data.
### Discriminative Models

- Discriminative models are used for regression or classiﬁcation tasks. 
- In contrast to generative models, discriminative models are trained to distinguish between different classes or categories.
- Their emphasis lies in learning the boundaries between classes within the input space.
- discriminative models learn to evaluate the probability distribution of the output labels given the input features
### Transformer architecture

- Most foundation models are built on the transformer architecture, which has been instrumental in the rise of foundation models and large language models.
- The key enabling innovation of the Transformer architecture is the multi-head self-attention mechanism originally proposed in the seminal work. 
- each attention head computes a vector of importance weights that corresponds to how strongly a token in the context window $x_i$ correlates with other tokens in the same window $x_j$.
- Mathematically, an attention head maps each token $x_i$ in the context  window to a query $q_i = W_{k}^{T} x_j$. The similarity between query and key is then measured though a scaled dot product $q_{i}^{T} K_j/ \sqrt{d}$, where d is the dimension of the query  and key vectors. 
- Transformers attention model can be efficiently computed with GPUs and TPUs

$$
attn(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
Where Q, K, V are matrices with rows $q_i^T$, $K_i^T$, and $v_i^T$ respectively. 
- Each head in the model produces this computation independently, with different Wq, Wk, Wv matrices to encode different kinds of attention.

The size of the transformer model is typically quantified by:
1. the size of the context window
2. number of the attention heads per layer 
3. the size of the attention vector in each head
4. the number of stacked attention layers
### Autoregressive Models

- The concept of autoregression has been applied in many ﬁelds as a representation of random processes whose outputs depend causally on the previous outputs.
- The model again predicts the next data point in the sequence, repeating this process indeﬁnitely
- These modeling concepts were adapted to deep learning models ﬁrst with RNNs, and later LSTMs, which are both types of learnable nonlinear autoregressive models.

### Masked auto-encoding

- To address the unidirectional limitation of the GPT family and allow the model to make bidirectional predictions, works such as BERT
- This is achieved through an architectural change, namely the addition of a bidirectional encoder, as well as a novel pre-training objective known as masked language modeling (MLM).
### Contrastive learning

- Visual-language foundation models such as CLIP, typically rely on different training methods from the ones used with large language models, which encourage explicitly predictive behavior.
- Visual-language models use contrastive representation learning, where the goal is to learn a joint embedding space between input modalities where similar sample pairs are closer than dissimilar ones
- used for multi-modal learning

### Diffusion Models

- The diffusion probabilistic model is a deep generative model that is trained in an iterative forward and reverse process.
- This means the forward process produces a trajectory of noise $q(x_{1:T}| x_0)$ as

$$
q(x_{1:T}| x_0) := \prod_{t=1}^{T} q(x_t| x_{t-1}) 
$$
- The reverse process requires the model to learn to the transitions that will de-noise then zero-mean Gaussian  and produce the input image. 

$$
p_\theta (x_{0:T}) := p(x_T) \prod_{t=1}^{T} p_\theta (x_{t-1}| x_t)
$$
- Diffusion models are trained using a reduced form of the evidence lower bound loss function that is typical of variational generative models like variational autoencoders (VAEs).

## Large Language Model LLM examples and historical context

- LLMs can also be ﬁne-tuned, a process by which the model parameters are adjusted with domain-speciﬁc data to align the performance of the LLM to a speciﬁc use case.

## Vision Transformers ViT

- is a transformer architecture for computer vision tasks including image classiﬁcation segmentation, and object detection.
- A ViT treats an image as a sequence of image patches referred to as tokens.
- DINO is a self-supervised learning method, for training ViT. DINO is a form of knowledge distillation with no labels. 

## Multimodal Vision-Language models VLMs

- Multimodal refers to the ability of a model to accept different “modalities” of inputs, for example, images, texts, or audio signals.
- CLIP model architecture contains a text encoder and an image encoder (a modiﬁed version of vision transformer ViT) that are trained jointly to maximize the cosine similarity of the image and text embeddings. Uses a **contrastive learning** to incorporate models for zero-shot image classification
- FLIP randomly masks out and removes a signiﬁcant portion of image patches during training. This approach aims to improve the training efﬁciency of CLIP while maintaining its performance.

## Embodied multimodal language models

- An embodied agent is an AI system that interacts with a virtual or physical world
- Typical vision-language models are trained on general vision- language tasks such as image captioning or visual question answering.
- PaLM-E’s architecture injects (continuous) inputs such as images, low- level states, or 3D neural scene representations into the language embedding space of a decoder-only language model to enable the model to reason about text and other modalities jointly.

---
# Robotics

- For instance, LLMs may facilitate the process oftask speciﬁcation, allowing robots to receive and interpret high-level instructions from humans.
- VLMs may also promise contributions to the field
- Language models also play a role in offering feedback for policy learning techniques

## Robot policy learning for decision making and control 

### Language-conditioned imitation learning for manipulation

- a goal conditioned policy $\pi_\theta (a_t|s_t, l)$ is learned that outputs actions $a_t \in A$ conditioned on the current state $s_t \in S$ and language instruction $l \in \Lambda$            
- Demonstrations can be represented as trajectories, or sequences of images, RGB-D voxel observations, etc. Language instructions are paired with demonstrations to be used as the training dataset.
- Each language-annotated demonstration $\tau_i$ consist of $\tau_i = \{(s_1, l_1, a_1), (s_2, l_2, a_2), ... \}$ 
- At test time, the robot is given a series of instructions, and the language-conditioned visuomotor policy πθ provides actions at in a closed loop given the instruction at each time step.

Challenges in this domain: 
1. obtaining a sufficient volume of demonstrations and conditioning labels to train a policy
2. distribution shift under the closed loop policy -- the feedback of the policy can lead the robot into regions of the state space that are not well-covered in the training data, negatively impacting performance

- The multi-Context imitation framework is based on relabeled imitation learning and labeled instruction following. MCIL assumes access to multiple contextual imitation datasets, for example, goal image demonstrations, language goal demonstrations, or one-hot task demonstrations.

### Language-assisted reinforcement Learning

- Reinforcement learning (RL) is a family of methods that enable a robot to optimize a policy through interaction with its environment by optimizing a reward function.
- Unlike imitation learning, RL does not require human demonstrations, and (in theory) has the potential to attain super-human performance. 
- In Adaptive Agent (AdA), the authors present an RL foundation model that is an agent pretrained on diverse tasks and is designed to quickly adapt to open-ended embodied 3D problems by using fast incontext learning from feedback.
- In AdA a transformer architecture is trained using model-based RL to train agents with large-scale attention-based memory, which is required for adaptation.
- to enhance reinforcement learning by integrating Large Language Models (LLMs) and Visual-Language Models (VLMs) to create a more uniﬁed RL framework. This work considers robot manipulation tasks.
- The authors use an LLM to decompose complex tasks into simpler sub-tasks, which are then utilized as inputs for a transformer-based agent to interact with the environment.

## Robot transformers

- Foundation models can be used for end-to-end control of robots by providing an integrated framework that combines perception, decision-making, and action generation.
- effectiveness is demonstrated of self-supervised visual pretraining using real-world images for learning motor control tasks directly from pixel inputs.
- They show that without any task-speciﬁc ﬁne-tuning of the pretrained encoder, the visual representations can be utilized for various motor control tasks.
- demonstrates a vision-language-action (VLA) model that takes a step further by learning from both web and robotics data.
- Both RT-1 and RT-2 consider robot manipulation and navigation tasks using a real-world mobile manipulator robot from Everyday Robots.
- RT-X explores training large cross-embodied robotic models, demonstrating positive transfer across robotic domains
- PACT (Perception-Action Causal Transformer) is a generative transformer architecture that builds representations from robot data with self-supervision
- SMART (Self-supervised Multi-task pretrAining with contRol Transformer) introduces self-supervised multi-task pretraining for control transformers
- Experimentation underscores SMART’s ability to enhance learning efﬁciency across tasks and domains.
- LATTE allows users to reshape robot trajectories using language instructions, leveraging pretrained language and visual-language models.
- LATTE transformer takes as input geometrical features of an initial trajectory guess along with the obstacle map conﬁguration.
- **One key obstacle to incorporating foundation models into robotics research is the reliance on real-world hardware experiments.**

## Language-image goal-conditioned value learning

- the aim is to **construct a value function** that aligns goals in different modalities and preserves temporal coherence due to the recursive nature of the value function.
- R3M (Reusable Representation for Robotic Manipulations) provides pretrained visual representation for robot manipulation using diverse human video datasets
- VIP (Value-Implicit Pretraining) employs time-contrastive learning to capture temporal dependencies in videos for learning visual goal-conditioned value functions **focused on manipulation tasks**
- Pretraining involves using unlabeled human videos.
- LIV (Language-Image Value Learning) is a control-centric vision-language representation that learns multi-modal vision-language value functions.
- LIV generalizes the prior work VIP by learning multi-modal vision-language value functions and representations using language-aligned videos.
- SayCan investigates integration of large language models with the physical world through learning, using language model for task-grounding and learned affordance function for world-grounding
- Inner Monologue studies the role of grounded environment feedback provided to the LLM for robot planning
- Text2Motion is a language-based planning framework for long-horizon robot manipulation
- VoxPoser builds 3D value maps to ground affordances and constraints into the perceptual space

## Robot task planning using large language models

- Large language models (LLMs) can be used to provide high-level task planning for performing complex long-horizon robot tasks.
### Language instructions for task specification
- SayCan uses LLM for high-level task planning in language with a learned value function to ground instructions in the environment
- Translation from natural language to temporal logic is proposed for imposing temporal specifications in robotic systems
- LLMs are used to translate natural language task description to intermediary task representation used by Task and Motion Planning (TAMP) algorithms

### Code generation using language models for task planning
- ProgPrompt uses LLMs to generate sequences of actions directly with no additional domain knowledge
- Code-as-Policies explores use of code-writing LLMs to generate robot policy code based on natural language commands
- ChatGPT is used to provide design principles for robotics, demonstrating how LLMs can help robotic capabilities rapidly generalize to different form factors

## In-context learning (ICL) for decision-making

- In-context learning operates without parameter optimization, relying on examples included in the prompt
- Chain-of-Thought is a prominent technique within in-context learning for executing sequence of intermediate steps for complex problems
- LLMs possess remarkable pattern recognition abilities through in-context learning
- Chain-of-Thought Predictive Control identifies specific brief sequences within demonstrations to understand hierarchical structure of tasks

## Open-vocabulary robot navigation and manipulation

### Open-vocabulary navigation
- VLN-BERT presents a visual-linguistic transformer-based model for visual navigation using web data
- LM-Nav is a system that utilizes pretrained models of images and language for visual navigation from natural language instructions
- ViNT is a foundation model for visual navigation tasks trained on diverse training data, utilizes a Transformer-based architecture to learn navigational affordances
- AVLMaps (Audio Visual Language Maps) presents 3D spatial map representation for cross-modal information from audio, visual, and language cues

### Open-vocabulary manipulation
- VIMA (VisuoMotor Attention Agent) learns robot manipulation from multi-modal prompts
- RoboCat is a self-improving AI agent that learns to operate different robotic arms and improves from self-generated data
- StructDiffusion enables robots to use partial viewpoint clouds and natural language instructions to construct goal configuration for objects
- MOO (Manipulation of Open-World Objects) leverages pretrained vision-language model to extract object-centric information
- DALL-E-Bot performs zero-shot autonomous rearrangement in scenes using pretrained image diffusion model DALL-E2

### Open-vocabulary grasping
- LERF (Language Embedded Radiance Fields) grounds CLIP embeddings into dense multi-scale 3D field for open-vocabulary grasping
- LERF-TOGO presents zero-shot open-vocabulary grasping framework generating grasp proposals over objects
- F3RM presents few-shot language-guided robot manipulation leveraging NeRF-based distilled feature field
- Splat-MOVER uses language-embedded Gaussian Splatting 3D field for multi-stage open-vocabulary manipulation

---
# Perception

Foundation models enable robots to convert high-dimensional sensory inputs into abstract, structured representations for robot understanding and reasoning.

## Open-vocabulary object detection and 3D classification

### Object detection
- GLIP (Grounded Language-Image Pre-training) integrates object detection and grounding by redefining object detection as phrase grounding
- OWL-ViT is an open-vocabulary object detector using vision transformer architecture with contrastive image-text pre-training
- Grounding DINO combines DINO with grounded pre-training, extending closed-set DINO model to open-set detection

### 3D classification
- PointCLIP transfers CLIP's pre-trained knowledge of 2D images to 3D point cloud understanding
- PointBERT uses transformer-based architecture to extract features from point clouds
- ULIP achieves unified representation of Language, Images, and Point clouds for 3D understanding by pre-training with object triplets

## Open-vocabulary semantic segmentation

- LSeg is a language-driven semantic segmentation model that associates semantically similar labels to similar regions in embedding space
- SAM (Segment Anything Model) introduces framework for promptable segmentation trained using supervised learning with data engines
- FastSAM and MobileSAM achieve comparable performance to SAM at faster inference speeds
- TAM (Track Anything Model) combines SAM and XMem for interactive video object tracking and segmentation

## Open-vocabulary 3D scene and object representations

### Language grounding in 3D scene
- LERF (Language Embedded Radiance Fields) grounds CLIP embeddings into dense multi-scale 3D field for 3D representation
- CLIP-Fields trains implicit scene representation by decoding latent vector to different modality-specific outputs
- VLMaps projects pixel embeddings from LSeg to grid cells in top-down grid map
- Semantic Abstraction decouples visual-semantic reasoning and 3D reasoning for 3D scene understanding
- 3D-LLM proposes using 2D VLMs as backbones to train 3D-LLM that takes 3D representations as inputs

### Scene editing
- CLIP-NeRF uses CLIP to disentangle dependence between shape and appearance in conditional neural radiance fields
- DFF (Distilled Feature Fields) trains distilled feature fields and manipulates them through query-based scene decomposition
- Nerflets represent 3D scene as combination of local neural radiance fields for more controlled editing
- ROSIE uses Imagen editor to modify training images for data augmentation during policy learning

### Object representations
- NDFs (Neural Descriptor Fields) learn correspondences between objects without dense annotation
- F3RM builds scene representations supporting finding corresponding object regions
- Correspondences between objects extracted directly from DINO features without training

## Learned affordances

- Affordance Diffusion synthesizes complex interactions of articulated hand with given object
- VRB (Vision-Robotic Bridge) trains visual affordance model on internet videos of human behavior

## Predictive models

- World models predict how state of world changes given particular agent actions
- GAIA-1 model generates predictions of driving video conditioned on arbitrary combinations of video, action, and text
- Video prediction models combined with goal-conditioned policies to solve manipulation tasks
- COMPASS constructs comprehensive multimodal graph to capture relational information across diverse modalities

## Challenges and perspectives

- Existing affordance models trained on large-scale data,largely due to data quality and quantity limitations and embodied gaps

---
# Embodied AI

Recent research shows success of LLMs can be extended to embodied AI domains.

## Key developments
- Statler endows LLMs with explicit representation of world state as "memory" maintained over time
- Dasgupta et al. combine pattern recognition and adaptation abilities in system with Planner, Actor, and Reporter
- EmbodiedGPT utilizes prefix adapters to augment language model's capacity for embodied tasks
- MineDojo is framework for developing generalist agents in Minecraft with thousands of open-ended language-prompted tasks
- Voyager introduces LLM-powered embodied lifelong learning agent in Minecraft using GPT-4
- GITM (Ghost in the Minecraft) leverages LLM to break down goals into sub-goals
- Reward design can be simplified by utilizing LLM as proxy reward function
- ELLM (Exploring with LLMs) rewards agent for achieving goals suggested by language model
- VPT (Video PreTraining) presents video pretraining where agent learns to act by watching unlabeled online videos
- Dynalang learns multi-modal world model to predict future text and image representations

## Generalist AI
- Gato is generalist agent working as multi-modal, multi-task, multi-embodiment generalist policy
- RRL learns behaviors directly from proprioceptive inputs and can learn from visual inputs using pre-trained visual representations

## Simulators
- Gibson emphasizes real-world perception for embodied agents
- iGibson and BEHAVIOR-1K support simulation of diverse household tasks with high simulation realism
- Habitat consists of Habitat-Sim and Habitat-API for research in Embodied AI
- Habitat-Lab is high-level library for embodied AI providing modular framework
- Habitat 3.0 expands capabilities for co-habitat for humans, avatars and robots
- RoboTHOR serves as platform for development and evaluation of embodied AI agents
- VirtualHome models complex activities occurring in typical household

---
# Challenges and Future Directions

## Overcoming data scarcity in training foundation models for robotics

### Scaling robot learning using unstructured play data and unlabeled videos of humans
- Use of teleoperated human-provided play data instead of fully annotated expert demonstrations
- Play data is unstructured, unlabeled, cheap to collect, but rich
- Very small percentage (as little as 1%) of language-annotated data needed for training

### Data augmentation using inpainting
- Use generative AI such as text-to-image diffusion models for data augmentation
- ROSIE uses Imagen editor to modify training images
- GenAug generates images with in-category and cross-category object substitutions
- CACTI pipeline includes step inpainting different plausible objects via Stable-Diffusion

### Overcoming 3D data scarcity for training 3D foundation models
- Primary obstacle in developing foundational 3D VLM models is scarcity of 3D data paired with language descriptions
- New datasets or data generation methods needed

### Synthetic data generation via high-fidelity simulation
- High-fidelity simulation via gaming engines can provide efficient means to collect data
- TartanAir dataset collected in simulation with various conditions and multi-modal sensor data

### Sim-to-real transfer
- Robotics policies trained in simulated environments can be transferred to real world
- Sim-to-real gap poses significant challenge for foundation models

### Data augmentation using VLMs
- DIAL uses VLM to label offline datasets for language-conditioned policy learning

### Robot physical skills limitations
- Robot physical skills limited to distribution of skills observed within robot data
- Approach involves using motion data from videos of humans performing various tasks

## Real-time performance (high inference time of foundation models)

- Inference time of foundation models still needs improvement for reliable real-time deployment
- Foundation models often stored and run in remote data centers, accessed through APIs requiring network connectivity
- Network reliability should be taken into account before integrating foundation model into robot's autonomy stack
- Potential solution is distillation of large foundation models into smaller-sized specialized models

## Limitations in multimodal representation

- Question remains whether single multimodal model can accommodate all modalities
- When paired data between modality and text is available, can embed that modality into text directly
- Some modalities lack sufficient data and need to be converted to other modalities first

## Uncertainty quantification

### Instance-level uncertainty quantification
- Quantify uncertainty in output of foundation model for particular input
- Instance-level uncertainty quantification can inform robot's decisions at runtime

### Distribution-level uncertainty quantification
- Quantify uncertainty in correctness of foundation model on distribution of possible future inputs
- Allows deciding whether given model is sufficiently reliable to deploy

### Calibration
- Estimates of uncertainty should be calibrated
- Important to distinguish between Frequentist and Bayesian interpretations of probabilities

### Distribution shift
- Important challenge in performing calibrated uncertainty quantification
- More subtle cause in robotics arises from closed-loop deployment of model

### Case study: uncertainty quantification for language-instructed robots
- KNOWNO performs both instance-level and distribution-level uncertainty quantification using conformal prediction
- Ensures statistically guaranteed level of task success

## Safety evaluation

### Pre-deployment safety tests
- Rigorous pre-deployment testing crucial for ensuring safety
- Foundation models often commit errors in ways hard to predict a priori
- Deployment cycle involves thorough red-teaming by human evaluators

### Runtime monitoring and out-of-distribution detection
- Robots should perform runtime monitoring during operation
- Can take form of failure prediction or out-of-distribution (OOD) detection

### Performance evaluation
- Greater reliance on statistical performance evaluation methods independent of policy's complexity

## Using existing foundation models "plug-and-play" vs. building new foundation models for robotics

### Incorporating tactile and audio sensing
- Tactile and audio sensing critical for human manipulation but less commonly utilized
- Early efforts toward developing unified and grounded representations for touch and audio

### Foundation models for high-level reasoning and task planning
- Development of robot-specific models for planning
- Planning problems often require reasoning over discrete decisions and continuous actions

## End-to-end vs. modular systems

- Open question of how much structure should be imposed on robotic system
- End-to-end approaches may yield stronger performance in long term
- Modular approaches could be key to improved generalization capabilities and better sample efficiency

## High variability in real-world robotic settings

- Robot platforms inherently diverse with different physical characteristics
- Real-world environments diverse and uncertain
- Key factor is to pretrain large models that are task-agnostic, cross-embodiment, and open-ended

## Benchmarking and reproducibility in robotics settings

- Necessity of real-world hardware experiments creates challenges for reproducibility
- Many works relied on non-physics-based simulators
- Combination of open hardware, benchmarking in physics-based simulators, and promoting transparency can alleviate challenges

## Envisioning the impact of foundation models in robotics

Ultimate goal is to develop foundation models that enable robots to:
- Safely and efficiently perform wide range of everyday tasks with high success rate
- Operate through simple interfaces such as natural language text input
- Household robots navigate indoor spaces, accurately grasp and manipulate objects, perform various chores
- Autonomous vehicles achieve human-level contextual reasoning
- Open-world navigation capabilities for exploration purposes
- Humanoid robots deployed in open-world environments with human-level navigation, manipulation, and dexterity