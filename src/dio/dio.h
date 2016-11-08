/*
 * dio.h
 *
 *  Created on: 04.11.2016
 *      Author: sebastian
 */

#ifndef DIO_H_
#define DIO_H_

#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <curand.h>
#include <curand_kernel.h>
#include "memory.h"
#include "task.h"

using namespace std;

class DataIO {
public:
	static const int max_threads = 4;
	Task * enqueue(function<void()>);

	void rng(int);
	void work_on_queue(int me);
	void wait(bool *cond,Task *task);
	void init_dio();
	int manage_threads();

	~DataIO(){
		std::cout << "destroying dio" << std::endl;
		for(int i = 0; i < max_threads;i++){
			if(thread_list[i].joinable()){
				thread_list[i].join();
			}
		}
	}
private:
    mutex queue_mtx;
    mutex thread_mtx[max_threads];

    queue<Task> task_queue;
    thread thread_list[max_threads];
    Memory<float> random_numbers;
    void work(void);
};

void DataIO::init_dio(){
	rng(512);
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
		for(int i = 0; i < max_threads;i++){
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

void DataIO::rng(int size){
	curandGenerator_t gen;

	cout << "Setting up Random Numbers" << endl;
	new_cuda_mem<float>(&random_numbers,size);

	/* Create pseudo-random number generator */
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	/* Set seed */
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	/* Generate n floats on device */
	curandGenerateUniform(gen, random_numbers.d_data,(size_t) size);
	/* Copy device memory to host */
	random_numbers.memcpy_dth();

	// check if kernel execution generated and error
	getLastCudaError("Random Number Kernel execution failed");



	// allocate mem for the result on host side
	//float *h_odata = (float *) malloc(mem_size);
	// copy result from device to host
	//checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * num_threads,
	//						   cudaMemcpyDeviceToHost));
}

#endif /* DIO_H_ */
