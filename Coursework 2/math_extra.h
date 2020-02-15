#pragma once

#define PI 3.141592653589793

class math_extra {
public:
	/*
	output[0] = mat[0][0] * vec[0];
	output[1] = mat[1][0] * vec[0];
	output[2] = mat[2][0] * vec[0];

	output[0] += mat[0][1] * vec[1];
	output[1] += mat[1][1] * vec[1];
	output[2] += mat[2][1] * vec[1];

	output[0] += mat[0][2] * vec[2];
	output[1] += mat[1][2] * vec[2];
	output[2] += mat[2][2] * vec[2];
	*/
	// Multiply a vector and a matrix together.
	// I left the working out out of the formula above - never know who might look at this
	static void mat_vec_multiply(const float mat[3][3], const float vec[3], float output[3]) {
		// Initialise output vector to 0
		for (int i = 0; i < 3; i++)
			output[i] = 0;

		// Multiply each vector component by it's matrix column
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				output[j] += mat[j][i] * vec[i];
			}
		}
	}

	// Add two vectors together. This is a trivial task,
	// but I split it into a separate function so it's
	// easier to adapt it for OpenCL / CUDA if need be
	static void vec_add(const float v1[3], const float v2[3], float output[3]) {
		for (int i = 0; i < 3; i++) {
			output[i] = v1[i] + v2[i];
		}
	}
};