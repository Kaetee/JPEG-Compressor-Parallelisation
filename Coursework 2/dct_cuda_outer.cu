#pragma once
#include "dct_function.h"

__global__ void calcDCT(unsigned int blockCountX, unsigned int blockCountY, unsigned int height, const float *blocks, double *output);

void dct_cuda_outer::run(image &img, array<vector<array<float, 64>>, 3> &blocks, array<vector<array<double, 64>>, 3> &DCT, int *args) {
	DCT[0].resize(blocks[0].size());
	DCT[1].resize(blocks[1].size());
	DCT[2].resize(blocks[2].size());

	float* data = new float[(64 * blocks[0].size() * 3)];
	double* output = new double[(64 * blocks[0].size() * 3)];

	for (int components = 0; components < 3; components++) {
		for (int i = 0; i < blocks[components].size(); i++) {
			for (int j = 0; j < 64; j++) {
				float x = blocks[components][i][j];
				data[j * blocks[0].size() * 3 + i * 3 + components] = x;
			}
		}
	}

	auto data_size_block = sizeof(float) * 64 * blocks[0].size() * 3;
	auto data_size_output = sizeof(double) * 64 * blocks[0].size() * 3;

	data_size_block = sizeof(float) * 64 * blocks[0].size() * 3;
	data_size_output = sizeof(double) * 64 * blocks[0].size() * 3;
	unsigned int blockCountY = img.BlockCountY();
	unsigned int blockCountX = img.BlockCountX();
	unsigned int height = blocks[0].size();

	float *buffer_Block;
	double *buffer_Output;

	cudaMalloc((void**)&buffer_Block, data_size_block);
	cudaMalloc((void**)&buffer_Output, data_size_output);

	cudaMemcpy(buffer_Block, data, data_size_block, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(blockCountX / threadsPerBlock.x, blockCountY / threadsPerBlock.y);

	calcDCT <<<numBlocks, threadsPerBlock >>>(blockCountX, blockCountY, height, buffer_Block, buffer_Output);

	cudaDeviceSynchronize();

	cudaMemcpy(output, buffer_Output, data_size_output, cudaMemcpyDeviceToHost);


	for (int components = 0; components < 3; components++) {
		for (int i = 0; i < blocks[components].size(); i++) {
			for (int j = 0; j < 64; j++) {
				DCT[components][i][j] = output[j * blocks[0].size() * 3 + i * 3 + components];
			}
		}
	}

	cudaFree(buffer_Block);
	cudaFree(buffer_Output);

	delete []data;
	delete []output;
}

__global__ void calcDCT(unsigned int blockCountX, unsigned int blockCountY, unsigned int height, const float *blocks, double *output) {
	unsigned int rows = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned int cols = (blockIdx.x * blockDim.x) + threadIdx.x;



	int index = cols + (rows * blockCountX);
	for (int components = 0; components < 3; components++) {

		for (int v = 0; v < JPEG_BLOCK_WIDTH; v++) {
			for (int u = 0; u < JPEG_BLOCK_WIDTH; u++) {
				int uVar = u + (v * JPEG_BLOCK_WIDTH);

				output[uVar * height * 3 + index * 3 + components] = 0;
				for (int y = 0; y < JPEG_BLOCK_WIDTH; y++) {
					for (int x = 0; x < JPEG_BLOCK_WIDTH; x++) {

						int xVar = x + (y * JPEG_BLOCK_WIDTH);
						output[uVar * height * 3 + index * 3 + components] +=
							double(blocks[xVar * height * 3 + index * 3 + components]) *
							cosf(double(u) * PI * (2.0 * double(x) + 1.0) / (16.0)) *
							cosf(double(v) * PI * (2.0 * double(y) + 1.0) / (16.0));
					}
				}
				double Cu = (u == 0) ? (1.0 / sqrtf(2.0)) : 1.0;
				double Cv = (v == 0) ? (1.0 / sqrtf(2.0)) : 1.0;

				output[uVar * height * 3 + index * 3 + components] *= 0.25 * Cu * Cv;
			}
		}
	}
}