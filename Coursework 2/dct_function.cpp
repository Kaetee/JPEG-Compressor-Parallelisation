#pragma once
#include "dct_function.h"
// Default DCT calculation
void dct_function::run(image & img, array<vector<array<float, 64>>, 3>& blocks, array<vector<array<double, 64>>, 3>& DCT, int *args) {
	DCT[0].resize(blocks[0].size());
	DCT[1].resize(blocks[1].size());
	DCT[2].resize(blocks[2].size());

	// Go through every block by Y and X, then every pixel component
	int rows, index, components;
	// <-- OpenMP Static Test -->
	// <-- OpenMP Dynamic Test -->
	// <-- CUDA Image-Level test -->
	for (rows = 0; rows < img.BlockCountY(); rows++) {
		for (int columns = 0; columns < img.BlockCountX(); columns++) {
			index = columns + (rows * img.BlockCountX());
			for (components = 0; components < 3; components++) {

				// <-- CUDA Block-Level Test -->
				for (int v = 0; v < JPEG_BLOCK_WIDTH; v++) {
					for (int u = 0; u < JPEG_BLOCK_WIDTH; u++) {
						DCT[components][index][u + (v * JPEG_BLOCK_WIDTH)] = 0;
						for (int y = 0; y < JPEG_BLOCK_WIDTH; y++)
							for (int x = 0; x < JPEG_BLOCK_WIDTH; x++)
								// Append new information
								DCT[components][index][u + (v * JPEG_BLOCK_WIDTH)] += double(blocks[components][index][x + (y * JPEG_BLOCK_WIDTH)]) *
									cos(double(u) * PI * (2.0 * double(x) + 1.0) / (16.0)) * cos(double(v) * PI * (2.0 * double(y) + 1.0) / (16.0));

						double Cu = (u == 0) ? (1.0 / sqrt(2.0)) : 1.0;
						double Cv = (v == 0) ? (1.0 / sqrt(2.0)) : 1.0;
						// Save DCT
						DCT[components][index][u + (v * JPEG_BLOCK_WIDTH)] *= 0.25 * Cu * Cv;
					}
				}
			}
		}
	}
}