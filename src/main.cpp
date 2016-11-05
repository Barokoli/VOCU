#include <string>
#include <iostream>
#include <stdlib.h>
#include "gl/glContext.h"
#include "dio/dio.h"
#include "dio/octree.h"


Context context;
DataIO dio;
Octree octreeData1;

void context_input(int in);

int main(int argc, char **argv)
{
	char* err;
	std::cout << "Initializing VOCU on Thread:" << std::this_thread::get_id() << std::endl;
	if(!context.init_context(1080,720,err,context_input)){
		std::cout << err << std::endl;
	}
	context.start_render_loop();
	context.terminate_context();
}

void context_input(int in){
	std::cout << "Context key input!" << std::endl;
	if(in == GLFW_KEY_D){
		Task *task = dio.enqueue([=](){ octreeData1.init_from_random(512);});
		//dio.wait(&octreeData1.calculated,task);
	}
}
