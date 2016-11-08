/*
 * octree_kernel.cuh
 *
 *  Created on: 08.11.2016
 *      Author: sebastian
 */

#ifndef OCTREE_KERNEL_CUH_
#define OCTREE_KERNEL_CUH_


__global__ void testKernel(float *g_idata, float *g_odata)
{
    // shared memory
    // the size is determined by the host application
    extern  __shared__  float sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // read in input data from global memory
    sdata[tid] = g_idata[tid];
    __syncthreads();

    // perform some computations
    sdata[tid] = (float) num_threads * sdata[tid];
    __syncthreads();

    // write data to global memory
    g_odata[tid] = sdata[tid];
}



#endif /* OCTREE_KERNEL_CUH_ */
