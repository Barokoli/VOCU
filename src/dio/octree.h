/*
 * octree.h
 *
 *  Created on: 03.11.2016
 *      Author: sebastian
 */

#ifndef OCTREE_H_
#define OCTREE_H_

#include <cuda_runtime.h>
#include "extern/helper_cuda.h"
#include "octree_kernel.cuh"

class Octree{
	public:
	void init_from_random(int sqr_size);
	bool calculated;
private:
	void *data_pointer;
};

void Octree::init_from_random(int sqr_size){
	std::cout << "Initializing Octree from random values. Size: (" << sqr_size << "," << sqr_size << "," << sqr_size << ")\nOn Thread:" << std::this_thread::get_id() << std::endl;

	unsigned int num_threads = 32;
	unsigned int mem_size = sizeof(float) * num_threads;
	float *h_idata = (float *) malloc(mem_size);

	// initalize the memory
	for (unsigned int i = 0; i < num_threads; ++i)
	{
		h_idata[i] = (float) i;
	}

	// allocate device memory
	float *d_idata;
	checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
	// copy host memory to device
	checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size,
							   cudaMemcpyHostToDevice));

	// allocate device memory for result
	float *d_odata;
	checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

	// setup execution parameters
	dim3  grid(1, 1, 1);
	dim3  threads(num_threads, 1, 1);

	// execute the kernel
	testKernel<<< grid, threads, mem_size >>>(d_idata, d_odata);

	// check if kernel execution generated and error
	getLastCudaError("Kernel execution failed");

	// allocate mem for the result on host side
	float *h_odata = (float *) malloc(mem_size);
	// copy result from device to host
	checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * num_threads,
							   cudaMemcpyDeviceToHost));

	std::cout << "init finished" << std::endl;
	calculated = true;
}

#endif /* OCTREE_H_ */
