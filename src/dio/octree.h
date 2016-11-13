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
	void init_from_random(DataIO *dio);
	bool calculated;
private:
	void *data_pointer;
};

void Octree::init_from_random(DataIO *dio){
	std::cout << "Initializing Octree from random values. Size: (" << CHUNK_SIZE << "," << CHUNK_SIZE << "," << CHUNK_SIZE << ")\nOn Thread:" << std::this_thread::get_id() << std::endl;

	dim3  grid(CHUNK_SIZE>>1, CHUNK_SIZE>>1, CHUNK_SIZE>>1);
	dim3  threads(8, 8, 8);
	k_octree_fill_blocks<<< grid, threads >>>(dio->tmp_bulkstorage.d_data, dio->random_numbers.d_data,1,64,0,0,0);


	//std::cout << "chunk Size." << (aworld.ChunkSize[0]>>1) << std::endl;
	/*
	 * OpenCL referenz
	 * std::size_t glob_size [3] = { aworld.ChunkSize[0]>>1, aworld.ChunkSize[1]>>1, aworld.ChunkSize[2]>>1 };//2 -> 8
	std::size_t local_size [3] = { 8,8,8 };//todo: MaxWorkGroupSize 4->8
	std::cout << "Global size:" << glob_size[0] << "   local size: " << local_size[0] <<  std::endl;
	clFinish(queue);
	CheckErrorCL (clEnqueueNDRangeKernel (queue, aworld.chunkInitKernel1, 3, nullptr, glob_size, local_size,
										  0, nullptr, nullptr),__LINE__);


	 //clEnqueueReadBuffer(queue, aworld.tmpChunkDataCLMem, CL_TRUE, 0, sizeof(uint)*aworld.tmpChunkDataSize, aworld.tmpChunkData, 0, NULL, NULL);//TODO:Needed?
	//clFinish(queue);
	//std::cout << "some random chunk data: " << aworld.tmpChunkData[2097153] << std::endl;
	//writeOut(aworld.tmpChunkData,aworld.tmpChunkDataSize);

	int AbsSize = (int)(aworld.ChunkSize[0]*aworld.ChunkSize[1]*aworld.ChunkSize[2]);
	SetKernelArgMem(aworld.chunkInitKernel2,0,aworld.tmpChunkDataCLMem);
	int dispachSize = AbsSize;
	int dspchSize = 0;
	//Debug.Log("Abssize : "+AbsSize);
	for(dspchSize = (int)aworld.ChunkSize[0]>>2; dspchSize>0; dspchSize>>=1){
		//Debug.Log("i ="+i+"; dispachSize ="+dispachSize+";");
		SetKernelArgValue(aworld.chunkInitKernel2,1,dispachSize,sizeof(int));
		glob_size[0] = (int)(dspchSize);
		glob_size[1] = (int)(dspchSize);
		glob_size[2] = (int)(dspchSize);
		std::cout << "chunk GPU init in progress." << std::endl;
		clFinish(queue);
		CheckErrorCL (clEnqueueNDRangeKernel (queue, aworld.chunkInitKernel2, 3, nullptr, glob_size, nullptr, 0, nullptr, nullptr),__LINE__);
		dispachSize += dspchSize*dspchSize*dspchSize*8;
	}

	clEnqueueReadBuffer(queue, aworld.tmpChunkDataCLMem, CL_TRUE, 0, sizeof(uint)*aworld.tmpChunkDataSize, aworld.tmpChunkData, 0, NULL, NULL);
	clFinish(queue);

	std::cout << "chunk GPU init finished." << std::endl;

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
							   cudaMemcpyDeviceToHost));*/

	std::cout << "init finished" << std::endl;
	calculated = true;
}

#endif /* OCTREE_H_ */
