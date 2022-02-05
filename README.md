# Image Smoothing Algorithm Based on Gradient Analysis
This repository contains C++ and Python 3 implementation of an image smoothing algorithm that was proposed in this [publication](https://ieeexplore.ieee.org/document/9117646).  

![example1](/images/example.png)  


## How to use code
Libraries used:
  - opencv 4.3.0 if you want Mat type support

Here is the simple example of filter usage with opencv Mat images:

```cpp
#include "FilterBasedOnGradientAnalysis.cpp"

int main()
{
    cv::Mat img = cv::imread("your_input_file_name", cv::IMREAD_COLOR);       //read image using opencv from file into Mat type
    
    int kernelSize = 3;                                                       //set kernelSize = 3 for filtering with 3x3 kernel
    int runsNumber = 2;                                                       //set number of runs: parameter n is 1 by default
    Filter<float, uint8_t> filter;                                            //create the instance of filter
    cv::Mat output = filter(img, kernelSize, n=runsNumber);                   //smooth image
    
    cv::imwrite("your_output_file_name", output);                             //write the result
    return 0;
}
```
Here is example with python:
```python
import filter_based_on_gradient_analysis as fga
import cv2

img = cv2.imread('your_input_file_name', cv2.IMREAD_COLOR)  # read images using opencv from file
kernel_size = 3                                             # set kernel_size = 3 for filtering with 3x3 kernel
runs_number = 2                                             # set number of runs: parameter n is 1 by default
output = fga.smooth(img, kernel_size, n=runs_number)        # smooth image
cv2.imwrite('your_output_file_name', output)                # write the result
```

## General idea
Our algorithm uses filtering and to achieve edge-preserving smoothing it uses two components of gradient vectors: their magnitudes (or lengths) and directions. Our method discriminates between two types of boundaries in given neighborhood: regular and irregular ones.
![boundaries](/images/boundaries.png)  
Regular boundaries have small deviations of gradient angles and the opposite for irregular ones. To measure closeness of angles cosine of doubled difference is used. As additional measure that helps to discriminate the types of boundaries inverted gradient values were used.  
![gradients](/images/gradients.png)  
When gradient magnitudes are inverted bigger values refer to textures (insignificant changes in gradient) and smaller refer to strong boundaries. So textures would have bigger weights and hence they would appear smoother. We also propose to smooth image of gradient magnitudes with median filter to enhance visual quality of results. The method proposed in this paper is easy to implement and compute and it gives good results in comparison with other techniques like bilateral filter.  

## Citation

If you used the code or want to reference this method in your work, please cite:

`@inproceedings{gudkov2020image,
  title={Image smoothing algorithm based on gradient analysis},
  author={Gudkov, Vladimir and Moiseev, Ilia},
  booktitle={2020 Ural Symposium on Biomedical Engineering, Radioelectronics and Information Technology (USBEREIT)},
  pages={403--406},
  year={2020},
  organization={IEEE}
}`

