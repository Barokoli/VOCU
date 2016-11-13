/*
 * misc.h
 *
 *  Created on: 12.11.2016
 *      Author: sebastian
 */

#ifndef MISC_H_
#define MISC_H_

#define CHUNK_SIZE 512
#define CHUNK_SIZE_3 CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE

#include <math.h>
#include "compute/kernel_manager.h"

int toCube(int n,KernelManager k_manager){
    return (int)ceil((float)n/(float)k_manager.max_threads)*k_manager.max_threads;
}

int rec_chunk_size(){
	int acc = 0;
	for(int x = CHUNK_SIZE_3; x > 0; x = x>>3){
		acc += x;
	}
	return acc;
}

int rec_chunk_size_cubed(KernelManager k_manager){
	return toCube(rec_chunk_size(),k_manager);
}

#endif /* MISC_H_ */
