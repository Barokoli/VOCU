/*
 * glContext.h
 *
 *  Created on: 03.11.2016
 *      Author: sebastian
 */

#ifndef GLCONTEXT_H_
#define GLCONTEXT_H_

#include <GLFW/glfw3.h>
#include <iostream>
#include <thread>

using namespace std;

void error_callback(int error, const char* description);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

class Context{
public:
	GLFWwindow* window;
	bool init_context(int x,int y,char *err);
	bool terminate_context();
	void start_render_loop();
};
bool Context::init_context(int x,int y,char *err){
	if (!glfwInit())
	{
		err = (char*)"GLFW was unable to initialize.";
		return false;
	}

	glfwSetErrorCallback(error_callback);

	window = glfwCreateWindow(x, y, "VOCU", NULL, NULL);
	if (!window){
		err = (char*)"Window or OpenGL context creation failed";
	}
	glfwSetKeyCallback(window, key_callback);

	return true;
}

bool Context::terminate_context(){
	std::cout << "Context Terminated" << std::endl;

	glfwDestroyWindow(window);

	glfwTerminate();
	return true;
}

void Context::start_render_loop(){
	std::cout << "GL Rendering." << std::endl;
	glfwMakeContextCurrent(window);

	while (!glfwWindowShouldClose(window)){
		glfwPollEvents();
	}
	std::cout << "GL Ended." << std::endl;
}

void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

#endif /* GLCONTEXT_H_ */
