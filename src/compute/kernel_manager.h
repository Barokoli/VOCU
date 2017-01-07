/*
 * kernel_manager.h
 *
 *  Created on: 08.11.2016
 *      Author: sebastian
 */

#ifndef KERNEL_MANAGER_H_
#define KERNEL_MANAGER_H_

#include <cuda_runtime.h>
#include "extern/helper_cuda.h"
#include <cuda_gl_interop.h>

class KernelManager {
public:
	void init_cuda_device(int argc, char **argv);
	int devID;
	int devCount;
	int max_threads;
};

void KernelManager::init_cuda_device(int argc, char **argv){
	/*http://stackoverflow.com/questions/5689028/how-to-get-card-specs-programatically-in-cuda*/
	const int kb = 1024;
	const int mb = kb * kb;

	std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;

	cudaGetDeviceCount(&devCount);

	for(int i = 0; i < devCount; ++i)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor << std::endl;
		std::cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << std::endl;
		std::cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << std::endl;
		std::cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << std::endl;
		std::cout << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;

		std::cout << "  Warp size:         " << props.warpSize << std::endl;
		std::cout << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
		std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
		std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << std::endl;
		std::cout << std::endl;
	}
	devID = findCudaDevice(argc, (const char **)argv);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, devID);

	max_threads = props.maxThreadsPerBlock;

	unsigned int glDeviceCount = 16;
	int glDevices[16];

	cudaGLGetDevices(&glDeviceCount, glDevices, glDeviceCount, cudaGLDeviceListAll);
	printf("OpenGL is using CUDA device(s): ");
	for (unsigned int i = 0; i < glDeviceCount; ++i) {
		printf("%s%d", i == 0 ? "" : ", ", glDevices[i]);
	}
	printf("\n");

	//cudaGLGetDevices(&devID,&props);
	cudaChooseDevice(&devID,&props);
}

#endif /* KERNEL_MANAGER_H_ */
