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
#include "memory.h"
#include "task.h"

using namespace std;

class DataIO {
public:
	DataIO(){
		rng(2097152);
	}
	~DataIO(){
		for(int i = 0; i < max_threads;i++){
			if(thread_list[i].joinable()){
				thread_list[i].join();
			}
		}
	}
	Task * enqueue(function<void()>);
	void rng(int);
	void work_on_queue(int me);
	void wait(bool *cond,Task *task);
	int manage_threads();
	static const int max_threads = 4;

private:
    mutex queue_mtx;
    mutex thread_mtx[max_threads];

    queue<Task> task_queue;
    thread thread_list[max_threads];
    Memory random_numbers;
    void work(void);
};

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
}

/*Wait till end of task. main thread freezes.*/
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
	cout << "Setting up Random Numbers" << endl;
}

#endif /* DIO_H_ */
