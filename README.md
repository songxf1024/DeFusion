# Image Matching Based on Decision Level Fusion of Handcrafted and Deep Features
This is the implementation of the paper "[Image Matching and Localization Based on Fusion of Handcrafted and Deep Features](https://ieeexplore.ieee.org/document/10225672)".

**Overall Architecture**  
<div align="center">
  <img src="https://github.com/user-attachments/assets/ac34bc7e-f615-4866-82e0-7340ffbb7d29" width="500px" />
</div>    
The system consists of four parts: a preprocessing step using specific characteristics, a CAR-HyNet for deep features extraction, a DLF and matching method combining handcrafted and deep features, and a target localization based on the proposed image matching scheme.    

---

**Architecture of CA-SandGlass (CoordAtt SandGlass)**
<div align="center">
  <img src="https://user-images.githubusercontent.com/111047002/233269098-4d400991-3686-45d5-a5e2-27a5a2c3adb4.png" width="500px" />
</div>
We notice that traditional convolutional operations are limited in capturing local positional relationships. To address this, we introduce a CoordAtt module to embed positional information into channel attention, allowing for capturing long-range dependencies for a more accurate description of features. Note that SandGlass is a lightweight module that focuses on feature information at different scales. Considering that CoordAtt focuses on long-range dependencies, we combine these two techniques to allow the network to generate more comprehensive and discriminative feature representations. In addition, considering that the residual connection needs to be built on high-dimensional features, we further combine these two modules to form the CA-SandGlass module. To improve the performance of the network effectively, we add CoordAtt to the second and the third FRN. Our experiments show that incorporating the CoordAtt module at these locations yields superior performance.

---

**Architecture of CAR_HyNet (Coordinate Attention Residual Network)**
<div align="center">
  <img src="https://user-images.githubusercontent.com/111047002/233264270-b07aa08f-d685-4587-9439-4a102916d08c.png" width="800px" />
</div>
The original HyNet structure is the same as L2-Net, which consists of six feature extraction layers and one output layer. In contrast, we propose the addition of two layers of CA-SandGlass to increase nonlinearity for better fitting ability. Another important improvement we introduce in CAR-HyNet is to leverage the full RGB three channels as inputs. Compared with grayscale images, color images contain much richer information at a negligible computational cost. The absence of color information in processing can result in incorrect matching, particularly in regions with identical grayscale and shape but different colors.

> We extract handcrafted features using the RootSIFT algorithm on grayscale images. To achieve fine-grained control over the correspondence of feature points during the fusion process and reduce the computational resources of deep learning, we use handcrafted features as the prior knowledge for extracting deep features. We input patches into CAR-HyNet to extract 128-D deep features. This approach fully leverages the feature points extracted by RootSIFT as prior knowledge for CAR-HyNet, resulting in enhanced rotation invariance and eventually generating deep features for fusion.

---

**Architecture of DLF (Decision Level Fusion)**
<div align="center">
  <img src="https://user-images.githubusercontent.com/111047002/233268987-e67f7d1d-e0ce-41c5-9aa9-47ec8fbce5bd.png" width="600px" />
</div>
For two input images to be matched, named Image1 and Image2, we first extract their RootSIFT feature descriptors D1RootSIFT and D2RootSIFT and then extract CAR-HyNet feature descriptors D1CAR-HyNet and D2CAR-HyNet with rotation and scale invariance, respectively. Since CAR-HyNet takes RootSIFT as its prior knowledge, handcrafted features and deep features form a one-to-one mapping relationship. We calculate the Euclidean distance between the feature descriptors of each feature point in the two images under RootSIFT and CAR-HyNet, respectively. Therefore, each feature point can obtain its two nearest neighboring points, and the distances are dmi,RootSIFT and dmi,CAR-HyNet , where m=1,2 indicates the first and second closest neighbors. For the two nearest neighboring points of each feature point, we find the distances of these two points at the corresponding positions of the CAR-HyNet feature points by traversing the RootSIFT feature points in turn. We then use the NNDR method to determine whether the matching is successful from the two feature extraction algorithms.

---

**Inverse Perspective Transformation**
<div align="center">
  <img src="https://user-images.githubusercontent.com/111047002/233270721-421d5ceb-c976-4d2a-a88c-4595e86d3397.jpg" width="600px" />
</div>
In most cases, there is a certain degree of perspective transformation between images that need to be matched, especially in aerial scenes where the drone may be at a tilt angle. Note that two images in space can be transformed by a transformation matrix. Also, note that the attitude information of the UAV and camera is available. Therefore, we propose to correct the oblique image to a bird’s eye view using attitude-based IPT to improve feature point extraction performance as well as the matching rate. More importantly, this approach does not incur high latency from simulating the viewpoint since it performs the transformation and matching only once.

---

# Citation
If you use this repository in your work, please cite our paper:
```bash
@ARTICLE{10225672,
  author={Song, Xianfeng and Zou, Yi and Shi, Zheng and Yang, Yanfeng},
  journal={IEEE Sensors Journal}, 
  title={Image Matching and Localization Based on Fusion of Handcrafted and Deep Features}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JSEN.2023.3305677}
}
```
