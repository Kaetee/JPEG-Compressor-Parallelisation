#pragma once
#include <array>
#include <vector>
#include "math_extra.h"
#include "jpeg_values.h"

using namespace std;

class image {
private:
	array<vector<float>, 3> values;
	int width;
	int height;
	int blockCountX;
	int blockCountY;

public:
	image() { }

	image(int size) {
		values[0].resize(size);
		values[1].resize(size);
		values[2].resize(size);
	}

	image(array<vector<float>, 3> pixels, int width, int height) : width(width), height(height) {
		values[0].resize(pixels.size());
		values[1].resize(pixels.size());
		values[2].resize(pixels.size());

		values = pixels;

		blockCountX = ((width % 8 != 0) ? (int)(floor(static_cast<long double>(width / 8.0)) + 1) * 8 : width);
		blockCountY = ((height % 8 != 0) ? (int)(floor(static_cast<long double>(height / 8.0)) + 1) * 8 : height);

		blockCountX = min(blockCountX, width / 8);
		blockCountY = min(blockCountY, height / 8);
	}

	void getPixels(array<vector<float>, 3> &output) {
		output = values;
	}

	vector<float> &operator[] (int x) {
		return values[x];
	}

	int Width() { return width; }
	int Height() { return height; }
	int BlockCountX() { return blockCountX; }
	int BlockCountY() { return blockCountY; }

	// Convert the stored arrays from RGB to YCbCr format
	// No function for backward conversion exists - this only an encoder, not a decoder.
	void toYCbCr() {
		for (int i = 0; i < values[0].size(); i++) {
			float current[3]{ values[0][i], values[1][i], values[2][i] };
			float output[3];

			math_extra::mat_vec_multiply(jpeg_values::YCbCr_Matrix, current, output);

			//output[0] = float(((0.299 * values[0][i] + 0.587 * values[1][i] + 0.114 * values[2][i])));
			//output[1] = float(((-0.16874 * values[0][i] - 0.33126 * values[1][i] + 0.5 * values[2][i])));
			//output[2] = float(((0.5 * values[0][i] - 0.41869 * values[1][i] -0.08131 * values[2][i])));

			math_extra::vec_add(jpeg_values::YCbCr_Vector, output, output);

			values[0][i] = output[0];
			values[1][i] = output[1];
			values[2][i] = output[2];
		}
	}

	// Subsampling function.
	// Subsampling is implemented but never used due to report page count restrictions,
	// but feel free to use this function for any testing.
	// NOTE: The block-generating and compression function do not compensate for subsampling.
	// The extra loops were removed to save processing time (as they'd only iterate once anyway).
	// They're trivial to implement however. See the extra "appendix.java" file for a "working" implementation.
	bool subSample(int J, int a, int b) {
		if (a == 0)
			return false;
		if (J % a != 0)
			return false;
		if (b != 0)
			if (a % b != 0)
				return false;

		vector<float> Cb;
		vector<float> Cr;
		int samples_x = 4 / a;
		int samples_y = (b == 0) ? 2 : 1;

		for (int row = 0; row < height;) {
			for (int column = 0; column < width;) {
				float sum_Cb = 0.0;
				float sum_Cr = 0.0;

				for (int y = 0; y < samples_y; y++) {
					for (int x = 0; x < samples_x; x++) {
						sum_Cb += values[1][column + (row * width)];
						sum_Cr += values[2][column + (row * width)];
					}
				}

				sum_Cb /= static_cast<float>(samples_x * samples_y);
				sum_Cr /= static_cast<float>(samples_x * samples_y);

				Cb.push_back(sum_Cb);
				Cr.push_back(sum_Cr);

				column += samples_x;
			}

			row += samples_y;
		}

		values[1] = Cb;
		values[2] = Cr;

		return true;
	}
};