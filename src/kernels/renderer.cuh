/*
 * renderer.cuh
 *
 *  Created on: 07.01.2017
 *      Author: sebastian
 */

#ifndef RENDERER_CUH_
#define RENDERER_CUH_

__global__ void k_render_to_buffer(uint *buffer){
	int idx = threadIdx.x+blockIdx.x*blockDim.x;
	buffer[idx] = 0xFFFFFF;
}

#endif /* RENDERER_CUH_ */
