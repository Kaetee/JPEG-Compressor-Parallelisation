//	This is the main processing file for JPEG compression and testing.
//
// ***********************************************************************************************************************
// *|																													|*
// *| PLEASE NOTE::																										|*
// *|	 Part of the functionality requires the installation of "LodePNG" via the NuGet Manager to function correctly.	|*
// *|																													|*
// ***********************************************************************************************************************

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <lodepng.h>
#include <fstream>
#include "huffman.h"
#include <chrono>
#include <omp.h>
#include <thread>
#include "image.h"
#include "dct_function.h"

using namespace std;
using namespace chrono;

// Pauses execution until user input is detected
// It's only one line, but naming it "pause" helps with in-code clarity
void pause() {
	cin.get();
}

typedef void(*dct_func)(image&, array<vector<array<float, 64>>, 3>&, array<vector<array<double, 64>>, 3>&, int*);

// Quantises an array by dividing it by the corresponding value on a Quantisation table (JPEG standard)
void quantize(array<double, 64> &block, array<int, 64> &output, array<array<int, 8>, 8> table) {
	for (int y = 0; y < JPEG_BLOCK_WIDTH; y++) {
		for (int x = 0; x < JPEG_BLOCK_WIDTH; x++) {
			output[x + (y * JPEG_BLOCK_WIDTH)] = int(round(static_cast<long double>(block[x + (y * JPEG_BLOCK_WIDTH)] / float(table[y][x]))));
		}
	}
}



// Sprinkle of forward declaration magicc
struct image;
void makeBlocks(image &img, array<vector<array<float, 64>>, 3> &blocks);



// Loads image into array, 4 bytes per pixel (RGBA, RGBA, RGBA (...))
void loadPNG(string filename, unsigned int &width, unsigned int &height, array<vector<float>, 3> &pixels) {
	vector<unsigned char> image;

	unsigned error = lodepng::decode(image, width, height, filename);

	if (error) {
		cout << "Decoding error:: " << error << ":" << lodepng_error_text(error) << endl;
		pause();
		throw std::exception();
	}

	// Make sure pixels are large enough for the array
	pixels[0].resize(height * width);
	pixels[1].resize(height * width);
	pixels[2].resize(height * width);

	// Easier to think in height and width. Also, go through 4 pixels but only save 3 (discard the Alpha channel)
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			pixels[0][j + (i * width)] = float(image[(j * 4) + (i * width * 4)]);
			pixels[1][j + (i * width)] = float(image[(j * 4) + (i * width * 4) + 1]);
			pixels[2][j + (i * width)] = float(image[(j * 4) + (i * width * 4) + 2]);
		}
	}
}

// Write the two EOI characters to the phile
void writeEOI(ofstream &os) {
	char EOI[2]{ char(0xFF), char(0xD9) };
	os << EOI[0];
	os << EOI[1];

}

// Initialise the JPEG file
void writeHeaders(ofstream &os, huffman huf, array<array<int, 64>, 2> quantum, int imageHeight, int imageWidth) {
	string comment = "Comment.";
	// This is the ZigZag pattern. It runs faster when in array form than calculating on the fly
	int jpegNaturalOrder[64] = { 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40,
		48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
		58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };
	int i, j, index, offset, length;
	array<int, 64> tempArray;
	int qTableNumber[3]{ 0, 1, 1 };

	// the SOI marker
	char SOI[2] = { char(0xFF), char(0xD8) };
	os.write(reinterpret_cast<const char *>(SOI), sizeof(SOI));

	// The order of the following headers is quiet inconsequential.
	// the JFIF header
	char JFIF[18];
	JFIF[0] = char(0xff);
	JFIF[1] = char(0xe0);
	JFIF[2] = char(0x00);
	JFIF[3] = char(0x10);
	JFIF[4] = char(0x4a);
	JFIF[5] = char(0x46);
	JFIF[6] = char(0x49);
	JFIF[7] = char(0x46);
	JFIF[8] = char(0x00);
	JFIF[9] = char(0x01);
	JFIF[10] = char(0x00);
	JFIF[11] = char(0x00);
	JFIF[12] = char(0x00);
	JFIF[13] = char(0x01);
	JFIF[14] = char(0x00);
	JFIF[15] = char(0x01);
	JFIF[16] = char(0x00);
	JFIF[17] = char(0x00);
	// Write the information array to the file
	os.write(reinterpret_cast<const char *>(JFIF), sizeof(JFIF));

	// Comment Header
	length = comment.length();
	vector<char> COM;
	COM.resize(length + 4);
	COM[0] = char(0xFF);
	COM[1] = char(0xFE);
	COM[2] = char((length >> 8) & 0xFF);
	COM[3] = char(length & 0xFF);

	for (int i = 0; i < length; i++)
		COM[4 + i] = char(comment[i]);

	os.write((const char *)&COM[0], COM.size() * sizeof(char));
	// I mean, it's all just the JPEG standard

	// The DQT header
	// 0 is the luminance index and 1 is the chrominance index
	char DQT[134];
	DQT[0] = char(0xFF);
	DQT[1] = char(0xDB);
	DQT[2] = char(0x00);
	DQT[3] = char(0x84);
	offset = 4;

	for (i = 0; i < 2; i++) {
		DQT[offset++] = char((0 << 4) + i);
		tempArray = quantum[i];

		for (j = 0; j < 64; j++)
			DQT[offset++] = char(tempArray[jpegNaturalOrder[j]]);
	}

	os.write(reinterpret_cast<const char *>(DQT), sizeof(DQT));

	// Start of Frame Header
	char SOF[19];
	SOF[0] = char(0xFF);
	SOF[1] = char(0xC0);
	SOF[2] = char(0x00);
	SOF[3] = char(17);

	// Image information so file doesn't crash and hate you
	SOF[4] = char(8);
	SOF[5] = char((imageHeight >> 8) & 0xFF);
	SOF[6] = char((imageHeight) & 0xFF);
	SOF[7] = char((imageWidth >> 8) & 0xFF);
	SOF[8] = char((imageWidth) & 0xFF);
	SOF[9] = char(3);

	index = 10;
	for (i = 0; i < 3; i++) {
		SOF[index++] = char(i + 1);
		SOF[index++] = char((jpeg_values::hSampFactor[i] << 4) + jpeg_values::vSampFactor[i]);
		SOF[index++] = char(qTableNumber[i]);
	}

	os.write(reinterpret_cast<const char *>(SOF), sizeof(SOF));

	// The DHT Header
	vector<char> DHT1, DHT2, DHT3, DHT4;
	int bytes, temp, oldindex, intermediateindex;
	length = 2;
	index = 4;
	oldindex = 4;

	
	DHT1.resize(17);
	DHT4.resize(4);
	DHT4[0] = char(0xFF);
	DHT4[1] = char(0xC4);
	for (i = 0; i < 4; i++) {
		bytes = 0;
		DHT1[index++ - oldindex] = char(huf.bits[i][0]);

		for (j = 1; j < 17; j++) {
			temp = (huf.bits[i][j]);
			DHT1[index++ - oldindex] = char(temp);
			bytes += temp;
		}

		intermediateindex = index;
		DHT2.resize(bytes);
		for (j = 0; j < bytes; j++) {
			DHT2[index++ - intermediateindex] = char(huf.val[i][j]);
		}

		DHT3.resize(index);

		for (int i = 0; i < oldindex; i++)
			DHT3[0 + i] = DHT4[i];

		for (int i = 0; i < 17; i++)
			DHT3[oldindex + i] = DHT1[i];

		for (int i = 0; i < bytes; i++)
			DHT3[oldindex + 17 + i] = DHT2[i];

		DHT4 = DHT3;
		oldindex = index;
	}

	DHT4[2] = char(((index - 2) >> 8) & 0xFF);
	DHT4[3] = char((index - 2) & 0xFF);

	os.write(reinterpret_cast<const char *>(&DHT4[0]), DHT4.size() * sizeof(char));

	// Start of Scan Header
	char SOS[14];
	SOS[0] = char(0xFF);
	SOS[1] = char(0xDA);
	SOS[2] = char(0x00);
	SOS[3] = char(12);
	SOS[4] = char(3);

	index = 5;
	for (i = 0; i < SOS[4]; i++) {
		SOS[index++] = char(i + 1);
		SOS[index++] = char((qTableNumber[i] << 4) + qTableNumber[i]);
	}

	SOS[index++] = char(0);
	SOS[index++] = char(63);
	SOS[index++] = char((0 << 4) + 0);

	os.write(reinterpret_cast<const char *>(SOS), sizeof(SOS));

	os.flush();
}

// Splits the 3 1D pixel arrays into blocks
void makeBlocks(image &img, array<vector<array<float, 64>>, 3> &blocks) {
	blocks[0].resize(img[0].size() / (JPEG_BLOCK_SIZE));
	blocks[1].resize(img[1].size() / (JPEG_BLOCK_SIZE));
	blocks[2].resize(img[2].size() / (JPEG_BLOCK_SIZE));

	//cout << "Creating Blocks..." << endl;

	// Go through pixel components, then blocks
	for (int component = 0; component < 3; component++) {
		for (int y = 0; y < img.BlockCountY(); y++) {
			for (int x = 0; x < img.BlockCountX(); x++) {
				array<float, 64> block;
				for (int i = 0; i < JPEG_BLOCK_WIDTH; i++) {
					for (int j = 0; j < JPEG_BLOCK_WIDTH; j++) {

						// Translate the block index into a pixel index
						int index_x = j + (x * JPEG_BLOCK_WIDTH);
						int index_y = i + (y * JPEG_BLOCK_WIDTH);
						int index = index_x + (index_y * JPEG_BLOCK_WIDTH * img.BlockCountX());

						block[j + (i * JPEG_BLOCK_WIDTH)] = img[component][index] - 128;

					}
				}

				blocks[component][x + (y * img.BlockCountX())] = block;
			}
		}
	}
	//cout << "Blocks Created." << endl;

	//cout << "Count:: " << blocks[1].size() << endl;
	//cout << "Count:: " << blocks[2].size() << endl;
}

// Quantize function that works on multiple blocks. For running testse
void quantizeBlocks(image &img, array<vector<array<double, 64>>, 3> &blocks, array<vector<array<int, 64>>, 3> &quantized) {
	quantized[0].resize(blocks[0].size());
	quantized[1].resize(blocks[1].size());
	quantized[2].resize(blocks[2].size());

	int rows, columns, index, components;
	for (rows = 0; rows < img.BlockCountY(); rows++) {
		for (columns = 0; columns < img.BlockCountX(); columns++) {
			// Index of the block in the array
			index = columns + (rows * img.BlockCountX());
			for (components = 0; components < 3; components++) {
				quantize(blocks[components][index], quantized[components][index], jpeg_values::qTables[components]);
			}
		}
	}
}

void writeCompressedData(ofstream &os, image &img, array<vector<array<int, 64>>, 3> quantizedBlocks, huffman &huf) {
	// This initial setting of MinBlockWidth and MinBlockHeight is done to
	// ensure they start with values larger than will actually be the case.
	int dcTableNumber[3]{ 0, 1, 1 };
	int acTableNumber[3]{ 0, 1, 1 }; // index tables for AC and DC tables

	int lastDCValue[3]{ 0, 0, 0 };
	int rows, columns, components, index;
	for (rows = 0; rows < img.BlockCountY(); rows++) {
		//cout << "[" << rows << "/" << img.BlockCountY() << "]" << endl;
		for (columns = 0; columns < img.BlockCountX(); columns++) {

			index = columns + (rows * img.BlockCountX());
			for (components = 0; components < 3; components++) {
				huf.huffmanBlockEncoder(os, quantizedBlocks[components][index], lastDCValue[components], dcTableNumber[components], acTableNumber[components]);
				lastDCValue[components] = quantizedBlocks[components][index][0];
			}
		}
	}
}

// Encode using Huffman Encoding
void huffmanEncode(ofstream &os, image &img, huffman &huf, array<vector<array<int, 64>>, 3> &quantizedBlocks) {
	writeCompressedData(os, img, quantizedBlocks, huf);
}

void saveImage(ofstream &os, image &img, huffman &huf) {
	array<array<int, 64>, 2> quantum;
	for (int y = 0; y < 8; y++) {
		for (int x = 0; x < 8; x++) {
			quantum[0][x + y * 8] = jpeg_values::Luminance_Matrix[y][x];
			quantum[1][x + y * 8] = jpeg_values::Chrominance_Matrix[y][x];
		}
	}

	//cout << "Writing Headers" << endl << endl;
	writeHeaders(os, huf, quantum, img.Height(), img.Width());

	//cout << "Writing Image Data" << endl << endl;
	huf.flushBuffer(os);

	//cout << "Writing EOI" << endl << endl;
	writeEOI(os);
}

image loadImage(string filename) {
	unsigned int width, height;
	array<vector<float>, 3> pixels;
	loadPNG(filename, width, height, pixels);
	image img(pixels, width, height);
	return img;
}

// Run tests and time them.
void runTest(string &filename, array<duration<double>, 7> &times, dct_func dctFunction, int &dctParameter) {
	cout << "-----     Process File[" << filename << "]     -----" << endl << endl;
	time_point<system_clock> start;
	time_point<system_clock> end;

	start = chrono::system_clock::now();
	ofstream os(string("output_") + filename + string(".jpg"), std::ios_base::binary | std::ios_base::out);
	image img(loadImage(filename + string(".png")));
	array<vector<array<int, 64>>, 3> quantized;
	huffman huff(img.Width(), img.Height());
	end = chrono::system_clock::now();
	array<vector<array<float, 64>>, 3> blocks;
	array<vector<array<double, 64>>, 3> DCT;
	times[0] = end - start;

	// YCbCr conversion
	start = chrono::system_clock::now();
	img.toYCbCr();
	end = chrono::system_clock::now();
	times[1] = end - start;
	// Splitting
	start = chrono::system_clock::now();
	makeBlocks(img, blocks);
	end = chrono::system_clock::now();
	times[2] = end - start;
	// DCT transforms
	start = chrono::system_clock::now();
	dctFunction(img, blocks, DCT, &dctParameter);
	end = chrono::system_clock::now();
	times[3] = end - start;
	// Quantisation
	start = chrono::system_clock::now();
	quantizeBlocks(img, DCT, quantized);
	end = chrono::system_clock::now();
	times[4] = end - start;
	// Huffman encoding
	start = chrono::system_clock::now();
	huffmanEncode(os, img, huff, quantized);
	end = chrono::system_clock::now();
	times[5] = end - start;
	// You're done, save the file!
	start = chrono::system_clock::now();
	saveImage(os, img, huff);
	end = chrono::system_clock::now();
	times[6] = end - start;

	os.close();
	cout << "-----     File[" << filename << "] Finished     -----" << endl << endl;
}
// Tests the a function for profiler. Closes because profiler is profiling!
void profileTest(string filename, dct_func func, int* args) {
	image img(loadImage(filename + string(".png")));
	array<vector<array<int, 64>>, 3> quantized;
	huffman huff(img.Width(), img.Height());
	array<vector<array<float, 64>>, 3> blocks;
	array<vector<array<double, 64>>, 3> DCT;
	ofstream os(string("output_") + filename + string(".jpg"), std::ios_base::binary | std::ios_base::out);

	img.toYCbCr();
	makeBlocks(img, blocks);
	func(img, blocks, DCT, args);
	quantizeBlocks(img, DCT, quantized);
	huffmanEncode(os, img, huff, quantized);
	saveImage(os, img, huff);

	exit(0);
}

int main() {
	//string filename = "960.png";
	// Opening output jpeg file in binary mode.
	// Sidenode: If your JPEG output looks identical to a regular jpeg output but has extra lines in,
	// Windows treats char(10) as CRLF in regular ofstream mode. Changing to binary makes it treat
	// char(10) as LF.

	profileTest("1440", dct_function::getRun(), 0);

	vector<dct_func> dctFunctions{
		dct_openmp_dynamic::getRun(),
		dct_cuda_outer::getRun(),
		dct_openmp_static::getRun(),
		dct_openmp_static::getRun(),
		dct_openmp_static::getRun(),
		dct_openmp_static::getRun(),
		dct_openmp_static::getRun(),
		dct_cuda_inner::getRun(),
		dct_function::getRun()
	};

	vector<string> dctFunctionNames{
		"OpenMP Dynamic",
		"CUDA Outer",
		"OpenMP Static (2 threads)",
		"OpenMP Static (4 threads)",
		"OpenMP Static (8 threads)",
		"OpenMP Static (16 threads)",
		"OpenMP Static (32 threads)",
		"CUDA Inner",
		"Default"
	};

	vector<int> dctParameters{
		8,
		0,
		2,
		4,
		8,
		16,
		32,
		0,
		0
	};

	vector<string> filenames{
		"768",
		"960",
		"1200",
		"1440"
	};

	string stagesConsole[7]{
		"Setup......",
		"YCbCring...",
		"Splitting..",
		"DCTing.....",
		"Quantizing.",
		"Huffman....",
		"Saving....."
	};

	string stagesCSV[7]{
		"Setup",
		"YCbCring",
		"Splitting",
		"DCTing",
		"Quantizing",
		"Huffman",
		"Saving"
	};

	ofstream excel("sheet.csv");

	vector<array<duration<double>, 7>> times;
	times.resize(filenames.size());
	
	
	for (int j = 0; j < 10; j++) {
		for (int i = 0; i < dctFunctions.size(); i++) {
			for (int file = 0; file < filenames.size(); file++) {
				runTest(filenames[file], times[file], dctFunctions[i], dctParameters[i]);
			}

			excel << dctFunctionNames[i] << ",[File],";

			for (int time = 0; time < 7; time++)
				excel << stagesCSV[time] << ",";
			excel << endl;

			for (int file = 0; file < filenames.size(); file++) {
				cout << "File:: " << filenames[file] << endl;
				excel << "," << filenames[file] << ",";

				for (int time = 0; time < 7; time++) {
					cout << ".....Time[" << stagesConsole[time] << "]:: " << times[file][time].count() << "s" << endl;
					excel << times[file][time].count() << ",";
				}
				excel << endl;

				cout << endl;
			}

			excel << endl;
			excel.flush();
		}
	}

	cout << endl;

	cout << "End:: " << endl;

	pause();
	return 0;
}