/*
 * glContext.h
 *
 *  Created on: 03.11.2016
 *      Author: sebastian
 */

#ifndef GLCONTEXT_H_
#define GLCONTEXT_H_

#include <GLFW/glfw3.h>
#include <OpenGL/gl.h>

void error_callback(int error, const char* description);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

class Context{
public:
	GLFWwindow* window;
	int width,height;
	bool init_context(int x,int y, void (*f)(int));
	bool terminate_context();
	void start_render_loop();
	void draw();
	void (*input_handle)(int);
};
bool Context::init_context(int x,int y, void (*f)(int)){
	width = x;
	height = y;
	input_handle = f;

	if (!glfwInit())
	{
		std::cout << "GLFW was unable to initialize." << std::endl;
		return false;
	}

	glfwSetErrorCallback(error_callback);

	window = glfwCreateWindow(x, y, "VOCU", NULL, NULL);
	if (!window){
		std::cout << "Window or OpenGL context creation failed" << std::endl;
	}
	glfwSetKeyCallback(window,key_callback);
	glfwSetWindowUserPointer(window, this);

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

	double lastTime = glfwGetTime();
	int nbFrames = 0;

	while (!glfwWindowShouldClose(window)){
		double currentTime = glfwGetTime();
		nbFrames++;
		if ( currentTime - lastTime >= 1.0 ){
			printf("%f ms/frame\n", 1000.0/double(nbFrames));
			nbFrames = 0;
			lastTime += 1.0;
		}
		draw();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	std::cout << "GL Ended." << std::endl;
}

void Context::draw(){

	glViewport(0, 0, width, height);
	glClearColor(0.6,0.7,0.8,1);
	glClear(GL_COLOR_BUFFER_BIT);

	return;
}

void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	Context * context = reinterpret_cast<Context *>(glfwGetWindowUserPointer(window));

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    if (action == GLFW_PRESS){
    	context->input_handle(key);
    }
}

#endif /* GLCONTEXT_H_ */
