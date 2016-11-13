#include <string>
#include <iostream>
#include <stdlib.h>
#include "gl/glContext.h"
#include "dio/dio.h"
#include "dio/octree.h"
#include "compute/kernel_manager.h"


Context context;
DataIO dio;
Octree octreeData1;
KernelManager kernel_manager;

void context_input(int in);

int main(int argc, char **argv)
{
	kernel_manager.init_cuda_device(argc, (char **)argv);
	log_cuda_mem();
	dio.init_dio(kernel_manager);
	std::cout << "Initializing VOCU on Thread:" << std::this_thread::get_id() << std::endl;
	context.init_context(1080,720,context_input);
	context.start_render_loop();
	context.terminate_context();
}

void context_input(int in){
	std::cout << "Context key input!" << std::endl;
	switch(in){
	case GLFW_KEY_ESCAPE:
		dio.free();
		break;
	case GLFW_KEY_D :
		Task *task = dio.enqueue([=](){ octreeData1.init_from_random(&dio);});
		//dio.wait(&octreeData1.calculated,task);
		break;
	}
}
