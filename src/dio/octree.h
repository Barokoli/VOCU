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
	void init_from_random(DataIO *dio,int max_threads_per_block);
	bool calculated;
private:
	void *data_pointer;
};

void Octree::init_from_random(DataIO *dio, int max_threads_per_block){
	std::cout << "Initializing Octree from random values. Size: (" << CHUNK_SIZE << "," << CHUNK_SIZE << "," << CHUNK_SIZE << ")\nOn Thread:" << std::this_thread::get_id() << std::endl;

	//First Stage: Fill Highest raw Layer, Group next Layer
	dim3  grid(CHUNK_SIZE>>4,CHUNK_SIZE>>4,CHUNK_SIZE>>4);
	dim3  threads(8, 8, 8);
	k_octree_fill_blocks<<< grid, threads >>>(dio->tmp_bulkstorage.d_data, dio->random_numbers.d_data,1,64,0,0,0);

	if(cudaSuccess != cudaGetLastError()){
		std::cout << "Failed kernel." << std::endl;
	}

	//Second Stage: Build Tree Raw | Split up smaller levels of the reduce?
	int level;
	int Off = CHUNK_SIZE_3;
	for(level = CHUNK_SIZE>>2; level > 0; level>>=1){
		threads = level >= 8? dim3(8,8,8) : dim3(threads.x>>1,threads.y>>1,threads.z>>1);
		grid = level >= 8? dim3(level>>3,level>>3,level>>3) : dim3(1,1,1);
		k_build_tree<<< grid, threads >>>(dio->tmp_bulkstorage.d_data,Off);
		std::cout << "offset:" << Off << "  grid: "<< grid.x << " threads:" << threads.x << std::endl;
		Off += level*level*level*8;
	}

	//Third Stage: Scan and Pack
	grid = dim3(dio->tmp_bulkstorage.size/max_threads_per_block/2,1,1);
	threads = dim3(max_threads_per_block,1,1);
	//Sharedmemory +1 for exclusive scan. Could be inclusive or first 0 used? Result is partialy packed with element counts at end
	k_blelloch_scan_and_pack<<< grid, threads, (1+max_threads_per_block*4)*sizeof(int) >>>(dio->tmp_bulkstorage.d_data,rec_chunk_size());

	dio->tmp_bulkstorage.memcpy_dth();
	int sum_elements = 0;
	for(int i = max_threads_per_block*2-1; i < dio->tmp_bulkstorage.size; i += max_threads_per_block*2){
		if(dio->tmp_bulkstorage.h_data[i]&META_MASK > 0){
			sum_elements += max_threads_per_block*2;
		}else{
			sum_elements += dio->tmp_bulkstorage.h_data[i];
			dio->tmp_bulkstorage.h_data[i] = sum_elements;
		}
	}
	dio->tmp_bulkstorage.memcpy_htd();
	//the bulkstorage now consists of variable sized (multiple of max_threads*2) chunks of continuous data. In front of every Chunk is an offset address.
	//Pack all remaining holes.

	std::cout << "init finished" << std::endl;
	calculated = true;
}

#endif /* OCTREE_H_ */
