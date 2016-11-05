/*
 * task.h
 *
 *  Created on: 05.11.2016
 *      Author: sebastian
 */

#ifndef TASK_H_
#define TASK_H_

#include <mutex>

using namespace std;

class Task{
public:
	Task(function<void()> f){
		func = f;
		work_mtx.lock();
		tid = 0;
	}
	~Task(){

	}
	Task(const Task &obj) {
		func = obj.func;
		work_mtx.lock();
		tid = obj.tid;
	}
	mutex work_mtx;
	function<void()> func;
	int tid;
	void exe(){
		work_mtx.unlock();
		func();
	}
};


#endif /* TASK_H_ */
