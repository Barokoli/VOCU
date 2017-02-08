/*
 * dio.h
 *
 *  Created on: 04.11.2016
 *      Author: sebastian
 */

#ifndef DIO_H_
#define DIO_H_

#include <iostream>
#include <stdlib.h>
#include <thread>
#include <mutex>
#include <queue>
#include <curand.h>
#include <curand_kernel.h>
#include "memory.h"
#include "task.h"
#include "misc.h"

using namespace std;

class DataIO {
public:
	static const int max_cpu_threads = 4;
	Task * enqueue(function<void()>);
	Memory<float> random_numbers;
	Memory<int> tmp_bulkstorage; //32 -> 64 bit pointer?
	Memory<int> tmp_scan_storage;

	void rng(int,unsigned long long seed);
	void work_on_queue(int me);
	void wait(bool *cond,Task *task);
	void init_dio(KernelManager k_manager);
	void free(void);
	int manage_threads();

	~DataIO(){
		std::cout << "destroying dio" << std::endl;
		for(int i = 0; i < max_cpu_threads;i++){
			if(thread_list[i].joinable()){
				thread_list[i].join();
			}
		}
	}
private:
    mutex queue_mtx;
    mutex thread_mtx[max_cpu_threads];

    queue<Task> task_queue;
    thread thread_list[max_cpu_threads];
    void work(void);
};

void DataIO::init_dio(KernelManager k_manager){
	rng(262144,time(0));
	new_cuda_mem<int>(&tmp_bulkstorage,rec_chunk_size_cubed(k_manager));
	new_cuda_mem<int>(&tmp_scan_storage,rec_chunk_size_cubed(k_manager));
}

void DataIO::work_on_queue(int me){
	thread_mtx[me].lock();
	queue_mtx.lock();
	cout << "DIO: executing queue." << endl;
	Task * f = &task_queue.front();
	task_queue.pop();
	queue_mtx.unlock();
	f->exe();
	thread_mtx[me].unlock();
}

Task * DataIO::enqueue(function<void()> func){
	Task t = Task(func);
	task_queue.push(t);
	t.tid = manage_threads();
	return &task_queue.front();
}

int DataIO::manage_threads(){
	if(!task_queue.empty()){
		for(int i = 0; i < max_cpu_threads;i++){
			if(!thread_list[i].joinable()){
				cout << "Starting thread." << endl;
				thread_list[i] = thread(&DataIO::work_on_queue,this,i);
				return i;
			}
		}
	}
	return -1;
}

/*Wait till end of task. Calling thread freezes.*/
void DataIO::wait(bool *cond,Task *t){
	cout << "started waiting for data" << endl;
	if(!*cond){
		t->work_mtx.lock();
		cout << "waiting for data" << endl;
		thread_mtx[t->tid].lock();
		thread_mtx[t->tid].unlock();
	}
	cout << "data finished" << endl;
}

void DataIO::rng(int size, unsigned long long seed){
	curandGenerator_t gen;

	clock_t begin = clock();
	new_cuda_mem<float>(&random_numbers,size);

	/* Create pseudo-random number generator */
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	/* Set seed */
	curandSetPseudoRandomGeneratorSeed(gen, seed);
	/* Generate n floats on device */
	curandGenerateUniform(gen, random_numbers.d_data,(size_t) size);
	/* Copy device memory to host */

	cudaDeviceSynchronize();

	random_numbers.memcpy_dth();
	cout << "random number: " << random_numbers.h_data[0] << endl;

	cudaDeviceSynchronize();

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC * 1000;

	cout << size << " Random Numbers created in: " << elapsed_secs << "ms (cpu/gpu time)"<< endl;
}

void DataIO::free(){
	random_numbers.mem_free();
	tmp_bulkstorage.mem_free();
	tmp_scan_storage.mem_free();
}

#endif /* DIO_H_ */
