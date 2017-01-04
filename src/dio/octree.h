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
#include <sstream>

class Octree{
	public:
	void init_from_random(DataIO *dio);
	bool calculated;
private:
	void *data_pointer;
};

void Octree::init_from_random(DataIO *dio){
	std::cout << "Initializing Octree from random values. Size: (" << CHUNK_SIZE << "," << CHUNK_SIZE << "," << CHUNK_SIZE << ")\nOn Thread:" << std::this_thread::get_id() << std::endl;

	//First Stage: Fill Highest raw Layer, Group next Layer
	dim3  grid(CHUNK_SIZE>>4,CHUNK_SIZE>>4,CHUNK_SIZE>>4);
	dim3  threads(8, 8, 8);
	k_octree_fill_blocks<<< grid, threads >>>(dio->tmp_bulkstorage.d_data, dio->random_numbers.d_data,1,64,0,0,0);

	if(cudaSuccess != cudaGetLastError()){
		std::cout << "Failed kernel." << std::endl;
	}

	//Second Stage: Build Tree Raw
	int level;
	int Off = CHUNK_SIZE_3;
	for(level = CHUNK_SIZE>>2; level > 0; level>>=1){
		threads = level >= 8? dim3(8,8,8) : dim3(threads.x>>1,threads.y>>1,threads.z>>1);
		grid = level >= 8? dim3(level>>3,level>>3,level>>3) : dim3(1,1,1);
		k_build_tree<<< grid, threads >>>(dio->tmp_bulkstorage.d_data,Off);
		std::cout << "offset:" << Off << "  grid: "<< grid.x << " threads:" << threads.x << std::endl;
		Off += level*level*level*8;
	}

	std::cout << "init finished" << std::endl;
	calculated = true;
}

#endif /* OCTREE_H_ */
