#pragma once
#include <vector>
#include <array>
#include <functional>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "jpeg_values.h"
#include "math_extra.h"
#include "image.h"
#include <stdlib.h>

using namespace std;

// Base DCT transform class. Performs DCT transforms on a given array in a hugely embedded loop, on one CPU thread
class dct_function {
public:
	// Define the DCT function as a type - this way they can be "grabbed" and stored in an array, allowing
	// automated testing
	typedef void(*dct_func)(image&, array<vector<array<float, 64>>, 3>&, array<vector<array<double, 64>>, 3>&, int*);
	// The main function of a DCT class. Performs DCT transforms on an array, and takes integer arguments
	// These can be used to determine thread count, timeout, anything really.
	static void run(image &img, array<vector<array<float, 64>>, 3> &blocks, array<vector<array<double, 64>>, 3> &DCT, int *args);

	// The getter for the run([...]) method. Read above.
	static dct_func getRun() { return run; }
};

// Performs DCT transforms on the GPU using CUDA. Does so on the inner loop (per-block)
class dct_cuda_inner : dct_function {
public:
	static void run(image & img, array<vector<array<float, 64>>, 3>& blocks, array<vector<array<double, 64>>, 3>& DCT, int *args);

	static dct_func getRun() { return run; }
};

// Performs DCT transforms on the GPU using CUDA. Does so on the outer loop (per-image)
class dct_cuda_outer : dct_function {
public:
	static void run(image & img, array<vector<array<float, 64>>, 3>& blocks, array<vector<array<double, 64>>, 3>& DCT, int *args);

	static dct_func getRun() { return run; }
};

// Performs DCT on CPU threads using OpenMP, set to static scheduling
class dct_openmp_static : dct_function {
public:
	static void run(image & img, array<vector<array<float, 64>>, 3>& blocks, array<vector<array<double, 64>>, 3>& DCT, int *args);

	static dct_func getRun() { return run; }
};

// Performs DCT on CPU threads using OpenMP, set to dynamic scheduling
class dct_openmp_dynamic : dct_function {
public:
	static void run(image & img, array<vector<array<float, 64>>, 3>& blocks, array<vector<array<double, 64>>, 3>& DCT, int *args);

	static dct_func getRun() { return run; }
};