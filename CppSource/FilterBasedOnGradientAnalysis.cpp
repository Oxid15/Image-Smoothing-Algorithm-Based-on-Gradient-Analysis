//Vladimir Gudkov, Ilia Moiseev
//South Ural State University, Chelyabinsk, Russia, 2020
//Image smoothing Algorithm Based on Gradient Analysis

/*
Functions and classes in this file use C++ templates.
The names of template arguments are the tips that were made to help user
in choosing the data types that are recommended to use (in parenthesis):
	Tf - floating point number   (float or double)
	Ts - signed integer          (int32_t)
	Tu - unsigned integer        (uint8_t)
*/

#include "stdint.h"
#include <cmath>
#include"\opencv\build\include\opencv2\opencv.hpp"

using namespace cv;

#define MIN(a,b) ((a) < (b) ? (a) : (b))

template<typename Tf, typename Ts>
Tf angle(Ts* vector) { return atan2(vector[1], vector[0]); }

template<typename Tf, typename Ts>
Tf module(Ts* vector) { return sqrt(vector[0] * vector[0] + vector[1] * vector[1]); }

template<typename Tu, typename Ts>
void grad(int x, int y, int c, Tu*** image, Ts* grad)
{
	grad[0] = Ts(image[y][x - 1][c] - image[y][x + 1][c]);
	grad[1] = Ts(image[y + 1][x][c] - image[y - 1][x][c]);
}

template<typename Tu, typename Ts>
void computeGrads(Tu*** src, Ts**** dst, uint32_t height, uint32_t width, uint32_t colors)
{
	/*
		The function to compute the gradient vectors.
		Takes the image from src and leaves gradient vectors as [x, y] in dst.
		
		Recieves:
		Tu*** src - source image, array with the shape (height x width x colors)
		Tf*** dst - destination array, where angles will be stored; has the same shape as src
		uint32_t height, width, colors - dimensions of the src image
	*/
	for (uint32_t c = 0; c < colors; c++)
		for (uint32_t i = 1; i < height - 1; i++)
			for (uint32_t j = 1; j < width - 1; j++)
				grad(j, i, c, src, dst[i][j][c]);
}

template<typename Tf, typename Tu, typename Ts>
void computeAngles(Tu*** src, Tf*** dst, uint32_t height, uint32_t width, uint32_t colors)
{
	/*
		The function to compute the angles of the gradients [1/2].
		Takes the image from src and leaves angles in dst. Computes gradients. If the gradients
		are already precomputed, use computeAngles [2/2]

		The angles are given by standard atan2 function, evaluated from gradient vector.
		
		Recieves:
		Tu*** src - source image, array with the shape (height x width x colors)
		Tf*** dst - destination array, where angles will be stored; has the same shape as src
		uint32_t height, width, colors - dimensions of the src image
	*/
	for (uint32_t c = 0; c < colors; c++)
		for (uint32_t i = 1; i < height - 1; i++)
			for (uint32_t j = 1; j < width - 1; j++)
			{
				Ts* gradient = new Ts[2];
				grad<Tu, Ts>(j, i, c, src, gradient);
				dst[i][j][c] = angle<Tf, Ts>(gradient);
				delete gradient;
			}
}

template<typename Tf, typename Tu, typename Ts>
void computeAngles(Tu*** src, Tf*** dst, Ts**** grads, uint32_t height, uint32_t width, uint32_t colors)
{
	/*
		The function to compute the angles of the gradients [2/2]. 
		Takes the image from src and gradients from grads, leaves angles in dst. 
		Uses precomputed gradients. If the gradients
		aren't precomputed, use computeAngles [1/2]

		The angles are given by standard atan2 function, evaluated from gradient vector.
		
		Recieves:
		Tu*** src - source image, array with the shape (height x width x colors)
		Tf*** dst - destination array, where angles will be stored; has the same shape as src
		Ts**** grads - array of precomputed gradients
		uint32_t height, width, colors - dimensions of the src image
	*/
	for (uint32_t c = 0; c < colors; c++)
		for (uint32_t i = 1; i < height - 1; i++)
			for (uint32_t j = 1; j < width - 1; j++)
				dst[i][j][c] = angle<Tf, Ts>(grads[i][j][c]);
}

template<typename Tf, typename Tu, typename Ts>
void computeModules(Tu*** src, Tf*** dst, uint32_t height, uint32_t width, uint32_t colors)
{
	/*
		The function to compute the modules [1/2]. 
		Takes the image from src and leaves modules in dst. Computes gradients. If the gradients
		are already precomputed, use computeModules [2/2]

		The modules are euclidean norm of gradients.
		
		Recieves:
		Tu*** src - source image, array with the shape (height x width x colors)
		Tf*** dst - destination array, where modules will be stored; has the same shape as src
		uint32_t height, width, colors - dimensions of the src image
	*/
	for (uint32_t c = 0; c < colors; c++)
		for (uint32_t i = 1; i < height - 1; i++)
			for (uint32_t j = 1; j < width - 1; j++)
			{
				Ts* gradient = new Ts[2];
				grad<Tu, Ts>(j, i, c, src, gradient);
				dst[i][j][c] = module<Tf, Ts>(gradient);
				delete gradient;
			}
}

template<typename Tf, typename Tu, typename Ts>
void computeModules(Tu*** src, Tf*** dst, Ts**** grads, uint32_t height, uint32_t width, uint32_t colors)
{
	/*
		The function to compute the modules [2/2]. 
		Takes the image from src and gradients from grads, leaves modules in dst. 
		Uses precomputed gradients. If the gradients
		aren't precomputed, use computeModules [1/2]

		The modules are euclidean norm of gradients.
		
		Recieves:
		Tu*** src - source image, array with the shape (height x width x colors)
		Tf*** dst - destination array, where modules will be stored; has the same shape as src
		Ts**** grads - array of precomputed gradients
		uint32_t height, width, colors - dimensions of the src image
	*/
	for (uint32_t c = 0; c < colors; c++)
		for (uint32_t i = 1; i < height - 1; i++)
			for (uint32_t j = 1; j < width - 1; j++)
				dst[i][j][c] = module<Tf, Ts>(grads[i][j][c]);
}

//The class that implements filtering. To use it create an instance of Filter class 
//and then call operator () from it
template<typename Tf, typename Tu, typename Ts>
class Filter
{
private:
	void filter(Tu*** src, Tf*** dst, Tf*** modules, Tf*** angles,
		uint32_t ksize, uint32_t height, uint32_t width, uint32_t colors)
	{
		for (uint32_t c = 0; c < colors; c++)
			for (uint32_t i = 0; i < height; i++)
				for (uint32_t j = 0; j < width; j++)
				{
					int up = i - ksize / 2;
					int left = j - ksize / 2;
					int down = i + ksize / 2 + 1;
					int right = j + ksize / 2 + 1;
					double sumWeights = .0;
					double result = .0;
					for (int k = up; k < down; k++)
					{
						if (k < 0 || k >= height) continue;
						for (int l = left; l < right; l++)
						{
							if (l < 0 || l >= width) continue;
							if (modules[k][l][c] == .0) continue;

							Tf weight = .0;
							if ((k != i) || (l != j))
							{
								Tf a = 1. / modules[k][l][c];
								Tf beta = 2. * (angles[i][j][c] - angles[k][l][c]);
								weight = (cos(beta) + 1.) * a;
							}
							else
								//weight of central pixel
								weight = 1.;

							result += weight * src[k][l][c];
							sumWeights += weight;
						}
					}
					if (sumWeights != 0)
						dst[i][j][c] = Tf(result / sumWeights);
					else
						//pixel remains without changes if sum of weights = 0
						dst[i][j][c] = Tf(src[i][j][c]);
				}
	}

public:
	/*
		Operator () filters src with kernel of size ksize, then leaves result in dst image.

		Recieves:
		Tu*** src - source image, array with the shape (height x width x colors)
		Tf*** dst - destination image with the same shape as src
		Tf**** grads - the array of gradients with the shape (n, m, 3, 2)
		Tf*** modules - the array of gradient modules with the same shape as src
		Tf*** modules - the array of gradient modules with the same shape as src
		uint32_t ksize - size of filtering kernel (odd values expected)
		uint32_t n - number of sequential runs
		uint32_t height, width, colors - dimensions of the src image
	*/
	void operator()(Tu*** src, Tf*** dst, Ts**** grads, Tf*** modules, Tf*** angles,
		uint32_t ksize, uint32_t n, uint32_t height, uint32_t width, uint32_t colors)
	{
		Tu*** srcProxy = new Tu**[height];
		for (uint32_t i = 0; i < height; i++)
		{
			srcProxy[i] = new Tu*[width];
			for (uint32_t j = 0; j < width; j++)
			{
				srcProxy[i][j] = new Tu[colors];
				for (uint32_t c = 0; c < colors; c++)
					srcProxy[i][j][c] = src[i][j][c];
			}	
		}

		for (int iter_num = 0; iter_num < n; iter_num++)
		{
			computeGrads<Tu, Ts>(srcProxy, grads, height, width, colors);
			computeModules<Tf, Tu, Ts>(srcProxy, modules, grads, height, width, colors);
			computeAngles<Tf, Tu, Ts>(srcProxy, angles, grads, height, width, colors);
			this->filter(srcProxy, dst, modules, angles, ksize, height, width, colors);

			for (uint32_t i = 0; i < height; i++)
			{
				for (uint32_t j = 0; j < width; j++)
				{
					for (uint32_t c = 0; c < colors; c++)
					{
						srcProxy[i][j][c] = (Tu)MIN(round(dst[i][j][c]), 255);
					}
				}
			}
			std::cout << iter_num << ' ';
		}

		for (uint32_t i = 0; i < height; i++)
		{
			for (uint32_t j = 0; j < width; j++)
			{
				delete srcProxy[i][j];
			}
			delete srcProxy[i];
		}
		delete srcProxy;
	}

	Mat operator()(Mat src, uint32_t ksize, uint32_t n=1)
	{
		/*
			Easy to use interface for filtering using opencv Mat data type.
			Recieves:
				Mat src - image to be filtered
				uint32_t ksize - kernel size
				uint32_t n - number of sequential runs
			Returns:
				Mat - filtered image
			Usage example:
				Mat input = imread(<input path>, IMREAD_COLOR);

				Filter<float, uint8_t> filter;
				Mat output = filter(input, <kernel size>);

				imwrite(<output path>, output);
		*/
		//allocating memory
		uint32_t height = src.rows;
		uint32_t width = src.cols;
		uint32_t colors = src.channels();

		Tu*** image = new Tu**[height];
		Tf*** output = new Tf**[height];
		Tu*** intoutput = new Tu**[height];
		Tf*** modules = new Tf**[height];
		Tf*** angles = new Tf**[height];
		Ts**** grads = new Ts***[height];
		for (uint32_t i = 0; i < height; i++)
		{
			image[i] = new Tu*[width];
			output[i] = new Tf*[width];
			intoutput[i] = new Tu*[width];
			modules[i] = new Tf*[width];
			angles[i] = new Tf*[width];
			grads[i] = new Ts**[width];
			for (uint32_t j = 0; j < width; j++)
			{
				image[i][j] = src.ptr<Tu>(i, j);

				output[i][j] = new Tf[colors];
				intoutput[i][j] = new Tu[colors];
				modules[i][j] = new Tf[colors];
				angles[i][j] = new Tf[colors];
				grads[i][j] = new Ts*[colors];
				for (uint32_t c = 0; c < colors; c++)
				{
					output[i][j][c] = .0;
					intoutput[i][j][c] = 0;
					modules[i][j][c] = .0;
					angles[i][j][c] = .0;
					grads[i][j][c] = new Ts[2];
					for (uint32_t k = 0; k < 2; k++)
						grads[i][j][c][k] = 0;
				}
			}
		}

		//filtering with regular function
		this->operator()(image, output, grads, modules, angles, ksize, n, height, width, colors);

		//deallocating memory that gradients, modules and angles were using
		for (uint32_t i = 0; i < height; i++)
		{
			for (uint32_t j = 0; j < width; j++)
			{
				for (uint32_t c = 0; c < colors; c++)
					delete grads[i][j][c];
				delete grads[i][j];
				delete modules[i][j];
				delete angles[i][j];
			}
			delete modules[i];
			delete angles[i];
			delete grads[i];
		}
		delete modules;
		delete angles;
		delete grads;

		//converting to integer, flattening and returning new Mat image
		//while deallocating all used memory
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				for (int c = 0; c < colors; c++)
					intoutput[i][j][c] = (Tu)MIN(round(output[i][j][c]), 255);
				delete output[i][j];
			}
			delete output[i];
		}
		delete output;

		Mat matOutput = Mat::zeros(height, width, CV_8UC3);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				for (int c = 0; c < colors; c++)
				{
					matOutput.ptr<Tu>(i, j)[c] = intoutput[i][j][c];
				}
				delete intoutput[i][j];
			}
			delete intoutput[i];
		}
		delete intoutput;

		return matOutput;
	}
};