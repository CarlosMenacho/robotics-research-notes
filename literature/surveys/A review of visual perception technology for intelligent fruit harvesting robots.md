
# Abstract

- Visual perception technology plays a crucial role in fruit picking robots, involving precise fruit identiﬁcation, localization, and grasping operations

--- 
# Introduction

- With the rapid advancements in artiﬁcial intelligence, robotics, and computer vision technologies, fruit harvesting robots have gradually become a focal point of research.
## Development status of intelligent fruit harvesting robots
- Modern intelligent fruit harvesting robots are now able to obtain real-time environmental information through devices such as cameras, LiDAR, and depth sensors, and identify the type, location, and status of fruits using image processing and pattern recognition technologies.
- bulk harvesting methods are efﬁcient, they cause signiﬁcant damage to the fruits and are difﬁcult to distinguish based on ripeness,
- Selective harvesting robots typically install the end effector on a robotic arm and use computer vision to identify ripe fruits, guiding the robotic arm and end effector

## The importance of visual perception technology in fruit harvesting

- It facilitates the accurate identiﬁcation and localization of target fruits through image processing and object detection, ensuring the efﬁcient and precise execution of harvesting tasks.
- Complex environment, adaptation is crucial for perception. 
- DL, 3D reconstruction, and Image segmentation techniques enhances its robustness
- Visual servo system, close-loop control improve manipulation and minimize fruit damage
- Fruit overlap, variation in fruit madurity, ilumination, plant anatomy recognition impact directy to the harvesting robot. 

---
# Common camera types for harvesting robots

## Monocular camera
- Lower costs
- combines DL applications
- Does not capture depth information, only 2D
- Primarily for object detection, yield estimation
- Some proposals, i) depth estimation using DL , ii) depth estimation using Markov random fields and geometric constrains
- Depth estimations requires high computation resources. 
## Stereo Camera
- They compute object depth by leveraging the principle of parallax.
- Deliver more accurate spatial localization in complex environments
- Some limitations: 
	- High dependency on texture features
	- Poor performance in texture-poor regions or sub-optimal lighting conditions
	- Hardware configuration is inherently more complex than monocular
	- demand high calibration and sync
## RGB-D Cameras
- Integrates RGB camera with a depth sensor (color and depth data). This is a popular choice
- Some sensors includes Time of Flight technology
- Offers high-accuracy at close range  and rapid  depth acquisition
- low reflecting surfaces or lacking texture may decrease accuracy
## Events Camera
- Have demonstrated significant potential in agricultural applications and complex environments (lighting changes)
- event cameras operate using an asynchronous imaging mechanism that records data only when changes in pixel brightness occur (microsecond)
- For instance, event cameras can produce stable outputs under highly variable lighting conditions
## Camera installation position

- The installation position of the camera directly determines the perception ability of the picking robot toward the fruits.
- **Eye-To-hand** means the camera is installed at fixed position on the robotic arm (robot's base, workbench or another). The camera's position does not change with the movement of the robotic arm 
- **Eye-In-Hand** installed on the robot's end-effector, each robot's movement directly affects the camera's view. This method is better at handling target localization and manipulations. 
# Object detection technology in fruit picking 

- there are a variety of fruits in (morphology, size and color)
- Two types of detection: Feature-based machine learning, deep learning based 

## Traditional object detection technology

- Classical methods using color-based, threshold segmentation, texture features, and shape features
- Widely used in scenarios with simple backgrounds and high contrast between the fruit and its surroundings.
- Color-based detection is effective but its performance deteriorates significantly in complex backgrounds or encountering similar colors
- As same as before, morphological detectors (e.g. canny, edges, Hough) robustness are limited in complex scenarios, specially with occluded fruits 
- Feature-based + classifier framework for object detection (SVM, KNN, RF), Constituted the mainstream approach in object detection before DL. 
	- Satisfactory in simple scenarios
	- Manually designed features
## Object detection technology based on Deep Learning DL

- Fruit harvesting confronts multiple challenges, including object recognition in complex environments, identiﬁcation and localization of diverse fruit types
- Detection are mainly using CNNs or variations. YOLO, DETR, R-CNN. 

### Two-stage object detection methods
- image classification tasks, these networks inherently lacked the capability to directly output positional information. 
- R-CNN generates region with deep feature extraction
- Mask R-CNN introduces segmentation
### One-stage object detection
- YOLO gained widespread adoption in agricultural scenarios. 
- Further developments from YOLOv9 to YOLOv12 introduced architectural innovations such as reversible branches, the GELAN backbone, and modules like C2f-faster and Area Attention 
- their effectiveness in agricultural environments depends on task-specific factors such as target size, occlusion level, and real-time requirements 
### Transformer-based object detection methods

- Adopted due to their ability to model global dependencies via self-attention mechanisms. Representative models include DETR 
- particularly suitable for complex agricultural environments with background clutter or occlusion. 
- Still facing several challenges in practical applications
	- High computational cost
	- slow convengence
	- a strong dependence on large-scale labeled datasets
- Transformed-based fruit detection framework was proposed but struggles with small objects and localization boundaries
--- 
# Data labeling methods and localization techniques for fruit picking

- The localization of picking points determines whether the fruit can be successfully harvested, making it one of the core aspects ofthe fruit picking process 
- The goal is to ensure the accurate identificatio and localization of picking points through efficien and precise labeling methods and localization technologies
- theres different types of labeling for this task
	- picking point calculation for different fruits
- Many works also propose the use of instance segmentation for fruits picking
- Also, pose estimation methods to calculate picking point 

--- 
# Robot mobility and global environment perception technologies
## Visual perception and navigation
- Visual perception is one of the core technologies enabling fruit harvesting robots to achieve autonomous navigation and environmental understanding 
- Some works uses a combination of eye-in-hand stereo vision with SLAM 
- 3D mapping also proposed for detecting palms oil trees
- SLAM combined with semantic segmentation net, was proposed to improve point cloud representation and enhancing real-time processing 
## Path planning for mobile robots
- a work proposes a global and local planning strategy using Traveling Salesman Tree TSP and Informed Rapidly-exploring Random Tree IRRT*
- another, full coverage path planning method based on multi-objective constrains
- Hybrid path planning approach was proposed combining inner spiral and improved nested methods
## Task scheduling

- Task scheduling is vital for enhancing the efﬁciency of multi-tasking harvesting robots
- an author proposed Multi-agent Reinforcement Learning MARL based scheduling method that dynamically adjust task allocation based on real-time environment 
- another work, developed Mixed-Integer Linear Programming  MILP optimizing task coordination and substantially improve strawberry harvesting

---
# Optimal viewpoint planning for fruit picking 

- environmental factors such as exposure, backlighting, shadows, occlusions, and vibrations may cause changes in the fruit’s position or lead to recognition failures. 
- Lightning could impact recognition
- Vibration or mechanical movement can also shift the fruit's position
## Geometry-based viewpoint planning method
- Focuses on selecting the optimal viewpoint by calculating the spatial relationships between the environment and the target object 
- Use depth cameras or LIDARs
- The visual system then identifies the position of the target fruit and analyzes the feasibility of viewpoint selection based on the geometric relationship between the fruit and the environment. 
- A work proposes 3D point cloud mappeing based on octrees 
- RVP constructed a voxel map of the fruit region
- These methods have high computational complexity
## Information-based and optimization-based viewpoint planning methods

-  **Information-based** Evaluate the characteristics of different viewpoints to select the ones that provide the maximum perceptual information or optimize task execution. 
- Widely applied in complex scenarios, several works are based: 
	- Estimating missing information through shape complations
	- Muti-viewpoint semantic perception decisions to determine the best viewpoint in tomato harvesting
- Most of the methods require evaluating multiple viewpoints, resulting in a large computational load
- **Optimization-based** uses optimizations algorithm  to select viewpoints
- these methods evaluate the quality of viewpoints by setting objective functions. As example:
	- Improved YOLOv5 and combined it with the ant colony algorithm are used to opimize the harvesting
	- RL to define a reward function for optimizing harvesting strategies using multi-arm
	- Generating candidates of viewpoints and scored them to select the best perspective
	
## Learning-based viewpoint planning methods
- Uses machine learning and DL methods to train models that learn how to select the optimal viewpoint based on occlusion conditions. 
- Particularly well in complex and dynamic environments
- this inclues DL RL, and others. Some previous works:
	- using few-shot RL to jointly train the Next Best View and the Next Best Point
	- YOLOv8 for real-time detection and drone to perform fruit picking (target points -> drone's speed)
- Geometric, information-based, optimization, and learning methods each have their advantages, adapting to different scenarios and requirements

### Summary
- Geometric methods are precise but complex and dependent on the specific conditions
- Information-based methods optimize viewpoints  but computationally intensive
- Optimization-based methods are effective but burdensome in complex environments 
- Learning methods are highly adaptable but rely on training data and resources

---
# Discussion
## Technical challenges and limitaions
- Stereo cameras provide depth estimation, but calibration in key
- Time-of-Flight offer excellent depth estimation but expensive and computationally intensive
- DL methods such as YOLO are improved, but requires powerful computational resources, large training data and depth data fusion
## Picking accuracy and efficiency

- Picking accuracy is crucial, we need to minimize the fruit damage
- Visual perception assist in pinpointing the picking location
- Real-time data processing is required
## Future development directions
- Future research could explore sensor fusion (visual, tactile, and force data) for perception
- Unsupervised learning may reduce the reliance on large datasets  and improve robot's adaptability
- DL + Visual servoing + Path planning  + Control, can be optimized 

