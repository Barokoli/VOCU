/*
 * memory.h
 *
 *  Created on: 04.11.2016
 *      Author: sebastian
 */

#ifndef MEMORY_H_
#define MEMORY_H_

#include "extern/helper_cuda.h"

template <class T>
class Memory{
public:
	int size;
	int gl_id;
	T *h_data;
	T *d_data;

	void memcpy_dth(void);
	void memcpy_htd(void);

	~Memory(){
		if(h_data)
			free(h_data);
		if(d_data)
			checkCudaErrors(cudaFree(d_data));
		std::cout << "Memory freed" << std::endl;
	}
};

template <class T>
void new_cuda_mem(Memory<T> *mem,int size){
	mem->size = size;
	mem->h_data = (T *) malloc(size);

	checkCudaErrors(cudaMalloc((void **) &(mem->d_data), size*sizeof(T)));
}

template <class T>
void Memory<T>::memcpy_dth(){
	cudaMemcpy(&h_data, &d_data, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template <class T>
void Memory<T>::memcpy_htd(){
	cudaMemcpy(&d_data, &h_data, size * sizeof(T), cudaMemcpyHostToDevice);
}

#endif /* MEMORY_H_ */
