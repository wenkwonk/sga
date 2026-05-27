### TECH STACK

### 1. Perception & Keypoint Extraction Layer
* **YOLOv11-Pose (Ultralytics):** Utilized for localized, top-down object detection and human pose estimation. It extracts continuous 17-point human skeletal configurations (nodes) at 30 frames per second directly from high-wide broadcast feeds.
* **ByteTrack:** An advanced multi-object tracking (MOT) algorithm used to maintain persistent bounding box identities (`Player_ID`) for both offensive and defensive players across sequential frames, mitigating identity switches during heavy body collisions.

### 2. Spatial Normalization Layer
* **OpenCV (Homography Alignment):** Maps 2D screen pixels to absolute, physical coordinates on a virtual 2D basketball court grid. By using static court boundary lines (the 3-point arc, key, and baseline) as ground-truth anchor anchors, the system eliminates variable camera panning, tilting, and zooming scales.
* **Kinetic Analytics Engine:** Computes real-world derivative metrics frame-over-frame, including linear acceleration, joint displacement vectors, and angular velocity of closing limbs.

### 3. Relational & Graph Data Layer
* **PyTorch Geometric (PyG):** Used to compile the raw skeleton data into structured, multi-relational graphs. Skeletons are treated as spatial graphs where physical joints are nodes and biological limbs are edges. 
* **Dynamic Interaction Grids:** Generates proximity edges between opposing players' graphs when spatial metrics indicate cross-body defensive tracking, isolating the exact boundary space where contact fouls occur.

### 4. Cognitive Sequence Inference Layer
* **Spatio-Temporal Graph Convolutional Networks (ST-GCN):** The core deep-learning classification engine. Built on pure **PyTorch**, the ST-GCN evaluates how the spatial graphs evolve over time (sequence length of ~90 frames). It extracts temporal patterns (e.g., sudden deceleration upon arm contact or illegal sliding footwork) to flag infractions.
* **Classification Head:** A Softmax inference layer that transforms multi-dimensional spatiotemporal embeddings into definitive rule infraction probabilities (e.g., `Shooting Foul: 94.2%`, `Clean Play: 5.8%`).