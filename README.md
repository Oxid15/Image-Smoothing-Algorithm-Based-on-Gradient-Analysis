# Image Smoothing Algorithm Based on Gradient Analysis
This repository contains C++ and Python 3.6 implementation of an image smoothing algorithm that was proposed in this [publication](https://ieeexplore.ieee.org/document/9117646) in IEEE conference "2020 Ural Symposium on Biomedical Engineering, Radioelectronics and Information Technology (USBEREIT)".  

![example1](/images/example1.jpg)  
  
## General idea
In this paper image smoothing algorithm based on gradient analysis is proposed. Our algorithm uses filtering and to achieve edge-preserving smoothing it uses two components of gradient vectors: their magnitudes (or lengths) and directions. Our method discriminates between two types of boundaries in given neighborhood: regular and irregular ones.
![boundaries](/images/boundaries.png)  
Regular boundaries have small deviations of gradient angles and the opposite for irregular ones. To measure closeness of angles cosine of doubled difference is used. As additional measure that helps to discriminate the types of boundaries inverted gradient values were used.  
![gradients](/images/gradients.png)  
When gradient magnitudes are inverted bigger values refer to textures (insignificant changes in gradient) and smaller refer to strong boundaries. So textures would have bigger weights and hence they would appear smoother. We also propose to filter image of gradient magnitudes with median filter to enhance visual quality of results. The method proposed in this paper is easy to implement and compute and it gives good results in comparison with other techniques like bilateral filter.  
  
## Examples
![example2](/images/example2.jpg)  
## Comparison
Here is the comparison with other smoothing algorithms.  
a) - original image  
b) - guided filter  
c) - bilateral filter  
d) - our filter  
![comparison](/images/comparison.png)
## Edge detection
Here is the output of Canny edge detector that was applied on the image with and without preprocessing with our filter.
![edges](/images/edge_detection.png)
