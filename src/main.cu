#include <string>
#include <iostream>
#include <stdlib.h>
#include "gl/glContext.h"
#include "dio/dio.h"
#include "dio/octree.h"
#include "compute/kernel_manager.h"
#include "gl/camera.h"


Context context;
DataIO dio;
Octree octreeData1;
Camera main_camera(1080,720);
KernelManager kernel_manager;

void context_input(int key, int action);
void context_poll_input(GLFWwindow *window);

int main(int argc, char **argv)
{
	kernel_manager.init_cuda_device(argc, (char **)argv);
	log_cuda_mem();
	dio.init_dio(kernel_manager);
	std::cout << "Initializing VOCU on Thread:" << std::this_thread::get_id() << std::endl;
	context.init_context(1080,720,context_input,context_poll_input,&main_camera);
	main_camera.init_cuda_mem();
	context.start_render_loop();
	context.terminate_context();
}

void context_poll_input(GLFWwindow *window){
	if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS){
		if(main_camera.drag){
			main_camera.rotateSS(main_camera.dragStart[0]-context.mouse_pos_x,main_camera.dragStart[1]-context.mouse_pos_y);
			main_camera.dragStart[0] = context.mouse_pos_x;
			main_camera.dragStart[1] = context.mouse_pos_y;
		}else{
			main_camera.drag = true;
			main_camera.dragStart[0] = context.mouse_pos_x;
			main_camera.dragStart[1] = context.mouse_pos_y;
    	}
	}
	if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE){
		if(main_camera.drag){
			main_camera.drag = false;
			main_camera.rotateSS(main_camera.dragStart[0]-context.mouse_pos_x,main_camera.dragStart[1]-context.mouse_pos_y);
		}
	}
	if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS){
		main_camera.speed = 2.0f;
	}
	if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_RELEASE){
			main_camera.speed = 1.0f;
		}
	glfwGetCursorPos(window, &context.mouse_pos_x, &context.mouse_pos_y);
	if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS){
		main_camera.position += main_camera.forward * main_camera.speed;
	}
	if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS){
		main_camera.position -= main_camera.right*main_camera.width * main_camera.speed;
	}
	if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS){
		main_camera.position -= main_camera.forward * main_camera.speed;
	}
	if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS){
		main_camera.position += main_camera.right*main_camera.width * main_camera.speed;
	}
	main_camera.updateParams();
}
void context_input(int key, int action){
	std::cout << "Context key input!" << std::endl;
	if(action == GLFW_PRESS){
		switch(key){
			case GLFW_KEY_ESCAPE:{
				main_camera.free();
				octreeData1.free();
				dio.free();
				break;
			}
			case GLFW_KEY_G :{
				Task *task = dio.enqueue([=](){context.objects_to_draw = octreeData1.init_from_random(&dio,kernel_manager.max_threads);});
				main_camera.updateParams();
				//dio.wait(&octreeData1.calculated,task);
				break;
			}
		}
	}
	if(action == GLFW_RELEASE){
		switch(key){
			case GLFW_MOUSE_BUTTON_LEFT :{
				if(main_camera.drag){
					main_camera.drag = false;
					main_camera.rotateSS(main_camera.dragStart[0]-context.mouse_pos_x,main_camera.dragStart[1]-context.mouse_pos_y);
				}
				break;
			}
		}
	}
}
