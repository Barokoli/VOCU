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
#include <chrono>

class Octree{
	public:
	Memory<int>* init_from_random(DataIO *dio,int max_threads_per_block);
	void free();
	bool calculated;
	Memory<int> data;
};

Memory<int>* Octree::init_from_random(DataIO *dio, int max_threads_per_block){
	std::cout << "Initializing Octree from random values. Size: (" << CHUNK_SIZE << "," << CHUNK_SIZE << "," << CHUNK_SIZE << ")\nOn Thread:" << std::this_thread::get_id() << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

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
		Off += level*level*level*8;
	}

	/*dio->tmp_bulkstorage.memcpy_dth();
	printf("last data:%i",dio->tmp_bulkstorage.h_data[rec_chunk_size()]);

	std::stringstream stream;
	stream << "ResultData:(";
	for(int i = max((int)rec_chunk_size()-100,0); i < rec_chunk_size();i++){
		stream << std::hex << dio->tmp_bulkstorage.h_data[i] << ";";
	}
	std::cout << stream.str() << ")" << std::endl;*/

	//Third Stage: Scan and Pack
	grid = dim3(dio->tmp_bulkstorage.size/max_threads_per_block/2,1,1);
	threads = dim3(max_threads_per_block,1,1);
	//Sharedmemory +1 for exclusive scan. Could be inclusive or first 0 used? Result is partialy packed with element counts at end
	k_blelloch_scan_and_pack<<< grid, threads, (1+max_threads_per_block*4)*sizeof(int) >>>(dio->tmp_bulkstorage.d_data,dio->tmp_scan_storage.d_data,rec_chunk_size());

	if(cudaSuccess != cudaGetLastError()){
		std::cout << "Failed kernel." << std::endl;
	}

	dio->tmp_scan_storage.memcpy_dth();
	int sum_elements = 0;
	int last_elements = 0;
	for(int i = 0; i < dio->tmp_scan_storage.size; i += max_threads_per_block*2){
		last_elements = sum_elements;
		sum_elements += dio->tmp_scan_storage.h_data[i];
		dio->tmp_scan_storage.h_data[i] = last_elements;
	}
	dio->tmp_scan_storage.memcpy_htd();

	/*std::stringstream stream;
	stream << "ResultData:(";
	for(int i = 80000; i < 80100;i++){
		stream << std::hex << dio->tmp_scan_storage.h_data[i] << ";";
	}
	std::cout << stream.str() << ")" << std::endl;*/

	//the bulkstorage now consists of variable sized (multiple of max_threads*2) chunks of continuous data. In front of every Chunk is an offset address.
	//Pack all remaining holes.
	std::cout << "compressed data size:" << sum_elements << std::endl;
	new_cuda_mem<int>(&data,sum_elements);

	//inherit from last call
	//grid = dim3(dio->tmp_bulkstorage.size/max_threads_per_block/2,1,1);
	//threads = dim3(max_threads_per_block,1,1);

	std::cout << "grid.x: " << grid.x << "  threads.x: " << threads.x << std::endl;
	k_copy_packed_mem<<< grid, threads >>>(dio->tmp_bulkstorage.d_data,dio->tmp_scan_storage.d_data,data.d_data);
	data.memcpy_dth();

	/*std::stringstream stream;
	stream << "ResultData:(";
	for(int i = max((int)data.size-100,0); i < data.size;i++){
		stream << std::hex << data.h_data[i] << ";";
	}
	std::cout << stream.str() << ")" << std::endl;*/

	if(cudaSuccess != cudaGetLastError()){
		std::cout << "Failed kernel." << std::endl;
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "init finished in:" << elapsed.count()*1000 << "ms" << std::endl;
	calculated = true;

	return &data;
}

void Octree::free(){
	data.mem_free();
}

#endif /* OCTREE_H_ */
