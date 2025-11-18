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