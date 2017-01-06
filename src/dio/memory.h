/*
 * memory.h
 *
 *  Created on: 04.11.2016
 *      Author: sebastian
 */

#ifndef MEMORY_H_
#define MEMORY_H_

#include "extern/helper_cuda.h"

void log_cuda_mem(void);

template <class T>
class Memory{
public:
	size_t size;
	int gl_id;
	T *h_data;
	T *d_data;

	void memcpy_dth(void);
	void memcpy_htd(void);
	void mem_free(void);

	~Memory(){
		if(h_data)
			free(h_data);
		if(d_data)
			checkCudaErrors(cudaFree(d_data));
	}
};

template <class T>
void new_cuda_mem(Memory<T> *mem,size_t size){
	mem->size = size;
	mem->h_data = (T *) malloc(size * sizeof(T));
	//std::cout << size * sizeof(T) << std::endl;
	checkCudaErrors(cudaMalloc((void **) &(mem->d_data), size * sizeof(T)));
}

template <class T>
void Memory<T>::memcpy_dth(){
	checkCudaErrors(cudaMemcpy((void*) h_data,(const void*) d_data, (size_t) size * sizeof(T), cudaMemcpyDeviceToHost));
}

template <class T>
void Memory<T>::memcpy_htd(){
	checkCudaErrors(cudaMemcpy((void*) d_data, (const void*) h_data, (size_t) size * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
void Memory<T>::mem_free(){
	std::cout << "free mem: " << size << std::endl;
	if(h_data)
		free(h_data);
	if(d_data)
		checkCudaErrors(cudaFree(d_data));
	h_data = NULL;
	d_data = NULL;
}

void log_cuda_mem(){
	size_t free_byte ;
	size_t total_byte ;
	cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

	if ( cudaSuccess != cuda_status ){
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
		exit(1);
	}
	double free_db = (double)free_byte ;
	double total_db = (double)total_byte ;
	double used_db = total_db - free_db ;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

#endif /* MEMORY_H_ */
