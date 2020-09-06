//Vladimir Gudkov, Ilia Moiseev
//South Ural State University, Chelyabinsk, Russia, 2020
//Image smoothing Algorithm Based on Gradient Analysis
/*
Functions and classes in this file use the template system.
The names of template arguments are the tips that were made to help user
in choosing the data types that are recommended to use (in parenthesis):
	Tf - floating point number   (float or double)
	Ts - signed integer          (int32_t)
	Tu - unsigned integer        (uint8_t)
*/

#include "stdint.h"
#include <cmath>

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
	for (uint32_t c = 0; c < colors; c++)
		for (uint32_t i = 1; i < height - 1; i++)
			for (uint32_t j = 1; j < width - 1; j++)
				grad(j, i, c, src, dst[i][j][c]);
}

template<typename Tf, typename Tu, typename Ts>
void computeAngles(Tu*** src, Tf*** dst, uint32_t height, uint32_t width, uint32_t colors)
{
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
	for (uint32_t c = 0; c < colors; c++)
		for (uint32_t i = 1; i < height - 1; i++)
			for (uint32_t j = 1; j < width - 1; j++)
				dst[i][j][c] = angle<Tf, Ts>(grads[i][j][c]);
}

template<typename Tf, typename Tu, typename Ts>
void computeModules(Tu*** src, Tf*** dst, uint32_t height, uint32_t width, uint32_t colors)
{
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
	for (uint32_t c = 0; c < colors; c++)
		for (uint32_t i = 1; i < height - 1; i++)
			for (uint32_t j = 1; j < width - 1; j++)
				dst[i][j][c] = module<Tf, Ts>(grads[i][j][c]);
}

//The class that implements filtering. To use it create an instance of Filter class 
//and then call operator ()
template<typename Tf, typename Tu>
class Filter
{
public:
	/*
		Operator () filters src with kernel of size ksize, then leaves result in dst image.

		Recieves:
		Tu*** src - source image, array with the shape (height x width x colors)
		Tf*** dst - destination image with the same shape as src
		Tf*** modules - the array of gradient modules with the same shape as src
		Tf*** modules - the array of gradient modules with the same shape as src
		uint32_t ksize - size of filtering kernel (odd values expected)
		uint32_t height, width, colors - dimensions of the src image
	*/
	void operator()(Tu*** src, Tf*** dst, Tf*** modules, Tf*** angles,
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

	/*
		Easy to use interface for filtering using opencv Mat data type.
		Recieves:
			Mat src - image to be filtered
			uint32_t ksize - kernel size
		Returns:
			Mat - filtered image
		Usage example:
			Mat input = imread(<input path>, IMREAD_COLOR);

			Filter<float, uint8_t> filter;
			Mat output = filter(input, <kernel size>);

			imwrite(<output path>, output);
	*/
	Mat operator()(Mat src, uint32_t ksize)
	{
		//allocating memory
		uint32_t height = src.rows;
		uint32_t width = src.cols;
		uint32_t colors = src.channels();

		uint8_t*** image = new uint8_t**[height];
		float*** output = new float**[height];
		uint8_t*** intoutput = new uint8_t**[height];
		float*** modules = new float**[height];
		float*** angles = new float**[height];
		int32_t**** grads = new int32_t***[height];
		for (uint32_t i = 0; i < height; i++)
		{
			image[i] = new uint8_t*[width];
			output[i] = new float*[width];
			intoutput[i] = new uint8_t*[width];
			modules[i] = new float*[width];
			angles[i] = new float*[width];
			grads[i] = new int32_t**[width];
			for (uint32_t j = 0; j < width; j++)
			{
				image[i][j] = src.ptr<uint8_t>(i, j);

				output[i][j] = new float[colors];
				intoutput[i][j] = new uint8_t[colors];
				modules[i][j] = new float[colors];
				angles[i][j] = new float[colors];
				grads[i][j] = new int32_t*[colors];
				for (uint32_t c = 0; c < colors; c++)
				{
					output[i][j][c] = .0;
					intoutput[i][j][c] = 0;
					modules[i][j][c] = .0;
					angles[i][j][c] = .0;
					grads[i][j][c] = new int32_t[2];
					for (uint32_t k = 0; k < 2; k++)
						grads[i][j][c][k] = 0;
				}
			}
		}

		computeGrads<uint8_t, int32_t>(image, grads, height, width, colors);
		computeModules<float, uint8_t, int32_t>(image, modules, grads, height, width, colors);
		computeAngles<float, uint8_t, int32_t>(image, angles, grads, height, width, colors);

		//filtering with regular function
		this->operator()(image, output, modules, angles, ksize, height, width, colors);
		//

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
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				for (int c = 0; c < colors; c++)
					intoutput[i][j][c] = (uint8_t)MIN(output[i][j][c], 255);
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
					matOutput.ptr<uint8_t>(i, j)[c] = intoutput[i][j][c];
				}
				delete intoutput[i][j];
			}
			delete intoutput[i];
		}
		delete intoutput;

		return matOutput;
	}
};