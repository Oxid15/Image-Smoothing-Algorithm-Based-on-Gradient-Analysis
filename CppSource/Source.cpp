//Vladimir Gudkov, Ilia Moiseev
//South Ural State University, Chelyabinsk, Russia, 2020
//Image smoothing Algorithm Based on Gradient Analysis

#include "stdint.h"
#include <cmath>

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

template<typename Tf, typename Tu>
class Filter
{
public:
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
}; 