/*
 * renderer.cuh
 *
 *  Created on: 07.01.2017
 *      Author: sebastian
 */

#ifndef RENDERER_CUH_
#define RENDERER_CUH_

#include "dio/misc.h"

__device__ static uint EvaluateBlock(int relX,int relY,int relZ,int *chunk,int lastI,float *blockSize);

__global__ void k_render_to_buffer(uint *buffer,int *octree,int octree_size){
	int u = threadIdx.x+blockIdx.x*blockDim.x;
	int v = threadIdx.y+blockIdx.y*blockDim.y;
	int idx = threadIdx.x+blockIdx.x*blockDim.x+(threadIdx.y+blockIdx.y*blockDim.y)*gridDim.x*blockDim.x;
	float bs = 0.0;
	if(u < CHUNK_SIZE && v < CHUNK_SIZE){
		buffer[idx] = EvaluateBlock(u,50,v,octree,octree_size,&bs) ;//> 127 ? 0xFFFFFF : 0x000000;//v+(u<<16):v+(u<<16);//0xFFFFFF : 0x000000;
	}
}

__device__ static uint EvaluateBlock(int relX,int relY,int relZ,int *chunk,int lastI,float *blockSize){
    float4 bPos = make_float4(0,0,0,0);
    uint block = chunk[lastI-1];

    uint off = 0;

    uint lvl = CHUNK_SIZE>>1;

    while((block&0xC0000000) != 0xC0000000){
        off |= (uint)(relX>=(bPos.x+lvl)? 1:0);
        off |= (uint)(relY>=(bPos.y+lvl)? 2:0);
        off |= (uint)(relZ>=(bPos.z+lvl)? 4:0);
        block = chunk[(block&0x3FFFFFFF)+off];
        bPos = make_float4((off&1)*lvl+bPos.x,((off&2)>>1)*lvl+bPos.y,((off&4)>>2)*lvl+bPos.z,0.0);
        //*blockSize = (float)lvl;
        lvl = lvl >> 1;
        off = 0;
    }

    return block&0x3FFFFFFF;
}

#endif /* RENDERER_CUH_ */
