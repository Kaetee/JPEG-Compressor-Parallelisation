#pragma once
#include "dct_function.h"

__global__ void calcDCT(const float *block, double *output);

void calculateDCTCUDAInner(array<float, 64> &block, array<double, 64> &output) {
	auto data_size_block = sizeof(float) * 64;
	auto data_size_output = sizeof(double) * 64;
	unsigned int width = 8;
	unsigned int height = 8;

	float *buffer_Block;
	double *buffer_Output;

	cudaMalloc((void**)&buffer_Block, data_size_block);
	cudaMalloc((void**)&buffer_Output, data_size_output);

	cudaMemcpy(buffer_Block, &block[0], data_size_block, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(4, 4);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

	calcDCT <<<numBlocks, threadsPerBlock >>>(buffer_Block, buffer_Output);

	cudaDeviceSynchronize();

	cudaMemcpy(&output[0], buffer_Output, data_size_output, cudaMemcpyDeviceToHost);

	cudaFree(buffer_Block);
	cudaFree(buffer_Output);
}

void dct_cuda_inner::run(image & img, array<vector<array<float, 64>>, 3>& blocks, array<vector<array<double, 64>>, 3>& DCT, int *args) {
	DCT[0].resize(blocks[0].size());
	DCT[1].resize(blocks[1].size());
	DCT[2].resize(blocks[2].size());

	int rows, index, components;
	for (rows = 0; rows < img.BlockCountY(); rows++) {
		for (int columns = 0; columns < img.BlockCountX(); columns++) {

			index = columns + (rows * img.BlockCountX());
			for (components = 0; components < 3; components++) {
				calculateDCTCUDAInner(blocks[components][index], DCT[components][index]);
			}
		}
	}
}

__global__ void calcDCT(const float *block, double *output) {
	unsigned int v = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	output[u + (v * JPEG_BLOCK_WIDTH)] = 0;

	//*
	for (int y = 0; y < JPEG_BLOCK_WIDTH; y++) {
		for (int x = 0; x < JPEG_BLOCK_WIDTH; x++) {
			output[u + (v * JPEG_BLOCK_WIDTH)] +=
				double(block[x + (y * JPEG_BLOCK_WIDTH)] *
					cosf(double(u) * PI * (2.0 * double(x) + 1.0) / (16.0)) *
					cosf(double(v) * PI * (2.0 * double(y) + 1.0) / (16.0)));
		}
	}
	//*/
	double Cu = (u == 0) ? (1.0 / sqrt(2.0)) : 1.0;
	double Cv = (v == 0) ? (1.0 / sqrt(2.0)) : 1.0;

	output[u + (v * JPEG_BLOCK_WIDTH)] *= 0.25 * Cu * Cv;
}