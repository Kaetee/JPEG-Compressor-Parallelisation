#pragma once
#include <array>

#define JPEG_BLOCK_WIDTH 8 // JPEG standard divides images into 8x8 blocks
#define JPEG_BLOCK_SIZE JPEG_BLOCK_WIDTH * JPEG_BLOCK_WIDTH

class jpeg_values {
public:
	//output[0] =       float(((0.299 * values[0][i] + 0.587 * values[1][i] + 0.114 * values[2][i])));
	//output[1] = 128 + float(((-0.16874 * values[0][i] - 0.33126 * values[1][i] + 0.5 * values[2][i])));
	//output[2] = 128 + float(((0.5 * values[0][i] - 0.41869 * values[1][i] -0.08131 * values[2][i])));

	// By the power of Wikipedia!
	// https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
	// (A more accurate version of the one found on http://www.equasys.de/colorconversion.html)
	// ((Sneaky suspicion the FLOAT type is canceling out any extra accuracy))
	static const float YCbCr_Matrix[3][3];

	// Look above
	static const float YCbCr_Vector[3];

	static const std::array<std::array<int, 8>, 8> Luminance_Matrix;

	static const std::array<std::array<int, 8>, 8> Chrominance_Matrix;

	static const std::array<std::array<std::array<int, 8>, 8>, 3> qTables;

	static const int vSampFactor[3];
	static const int hSampFactor[3];
};