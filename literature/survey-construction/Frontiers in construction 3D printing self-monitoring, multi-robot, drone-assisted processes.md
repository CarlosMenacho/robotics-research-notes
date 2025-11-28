
This comprehensive review examines the state-of-the-art applications of robotic systems in construction automation, with particular emphasis on 3D concrete printing (3DCP). The authors systematically analyze mechanical systems, digital frameworks, and sensory integration to identify current capabilities and research gaps in fully automated construction processes.

## Research Context

The construction industry faces declining productivity (20% reduction) compared to manufacturing (doubled productivity) since the introduction of robotic fabrication technologies. This disparity, combined with labor shortages in developed nations, necessitates investigation of robot-assisted construction technologies to address stagnating industry performance.

## Methodology

The paper employs systematic literature review methodology, analyzing publications across three primary domains: (1) robot-assisted systems and hardware, (2) digital modeling and control frameworks, and (3) quality inspection techniques. Publication trends indicate accelerating research interest, with significant growth in robot-assisted 3DCP, digital modeling, and quality inspection since 2016.

## Principal Findings

### 1. Mechanical Systems Architecture

**Multi-Robot and Mobile Platforms**: Studies demonstrate that collaborative multi-robot systems achieve significant efficiency gains (62.4% reduction in man-hours, 31.5% cost reduction) while maintaining structural compliance with building standards. However, mobile platform performance remains constrained by motion accuracy, base stability, and terrain irregularity.

**Kinematic Classifications**: Cartesian systems offer simplicity and cost-effectiveness with large build volumes but limited geometric flexibility. Articulated systems provide additional degrees of freedom enabling complex overhang structures and improved interlayer adhesion, though at higher operational complexity and cost.

### 2. Unmanned Aerial Vehicle (UAV) Integration

UAV-based construction systems demonstrate potential for circumventing build volume limitations and accessing challenging terrain. Current implementations include masonry assembly and collaborative fiber-winding with ground-based robots. Critical limitations include payload constraints and positioning accuracy requirements for structural applications.

### 3. Sensor Systems and Process Monitoring

**Inspection Technologies**: Laser scanning systems provide high-accuracy 3D point clouds with extended measurement distances and environmental robustness. Photogrammetry-based methods offer cost advantages but require controlled environmental conditions and complex multi-camera configurations.

**Real-Time Control Integration**: Research demonstrates closed-loop feedback systems using Time-of-Flight sensors, industrial cameras, and Hall Effect sensors to enable dynamic adjustment of extrusion parameters. However, communication latency (sampling frequencies 0.1-0.2s) and synchronization between sensing and control systems remain critical challenges.

**Artificial Intelligence Applications**: Deep learning approaches (VGG-16, ResNet architectures) show high classification accuracy for real-time quality assessment, enabling simultaneous adjustment of filament width and material fluidity during fabrication.

### 4. Digital Design and Control Framework

**Computational Design Tools**: Parametric design environments (Grasshopper, Rhinoceros) facilitate integration of fabrication constraints, topology optimization, and finite element analysis within CAD workflows. However, existing frameworks inadequately address 3DCP-specific constraints including fresh concrete rheology, time-dependent material properties, and multi-robot coordination.

**Path Planning Strategies**: Three primary slicing methodologies identified: (1) horizontal slicing (compatible with 3-axis systems but prone to cantilever instabilities), (2) constant layer thickness (requires 6-axis systems, maintains interlayer contact), and (3) curved-layer approaches (accommodates non-planar surfaces). Toolpath generation must reconcile geometric complexity with robot kinematic constraints and collision avoidance.

**Control Systems**: Robot programming employs manufacturer-specific languages (KRL, RAPID) or standardized GCode. However, GCode limitations in data acquisition and feedback integration restrict applicability in complex multi-robot scenarios.

## Critical Limitations

1. **Structural Performance**: Insufficient investigation of mechanical properties in multi-robot fabricated structures, particularly regarding interlayer bonding and interface strength at filament intersections.
2. **System Integration**: Disconnect between mechanical capabilities (mobile/multi-robot systems) and digital design frameworks, which fail to exploit additional structural freedoms.
3. **Material-Process Relationships**: Absence of validated correlations between fresh concrete rheological properties and real-time extrusion control parameters.
4. **Environmental Robustness**: Limited validation of mobile and vision-based systems under actual construction site conditions (lighting variability, terrain irregularity, environmental disturbances).
5. **Operational Complexity**: Multi-robot systems demand extensive interdisciplinary expertise spanning robotics, materials science, and construction practice, creating barriers to practical implementation.

## Research Directions

The authors recommend:

- Development of integrated design frameworks incorporating real-time sensor feedback and material property evolution
- Systematic investigation of mechanical performance in multi-robot fabricated structures
- Advancement of multi-process construction combining 3DCP, discrete assembly, and reinforcement placement
- Enhancement of mobile robot autonomy for complex terrain navigation and human-robot collaboration
- Expansion of UAV-based construction for remote and challenging environments

## Conclusion

While laboratory demonstrations establish technical feasibility of robotic construction achieving substantial efficiency improvements, significant research gaps persist between controlled experiments and practical large-scale implementation. Successful translation requires holistic frameworks integrating mechanical systems, sensing technologies, material science, and digital design methodologies. The interdisciplinary nature of these challenges necessitates collaborative research efforts to realize fully automated construction systems.