//opencv included in Source.cpp if you need to change include path, 
//you should change it there
#include "../src/FilterBasedOnGradientAnalysis.cpp"


int main()
{
    cv::Mat img = cv::imread("your_input_file_name", cv::IMREAD_COLOR);       //read image using opencv from file into Mat type

    int kernelSize = 3;                                                       //set kernelSize = 3 for filtering with 3x3 kernel
    int runsNumber = 2;                                                       //set number of runs: parameter n is 1 by default
    Filter<float, uint8_t, uint32_t> filter;                                            //create the instance of filter
    cv::Mat output = filter(img, kernelSize, runsNumber);                   //smooth image

    cv::imwrite("your_output_file_name", output);                             //write the result
    return 0;
}