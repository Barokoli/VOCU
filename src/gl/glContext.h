/*
 * glContext.h
 *
 *  Created on: 03.11.2016
 *      Author: sebastian
 */

#ifndef GLCONTEXT_H_
#define GLCONTEXT_H_

#include <GLFW/glfw3.h>
#include <OpenGL/gl3.h>
#include "shader/load_shader.h"

void error_callback(int error, const char* description);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

class Context{
public:
	GLFWwindow* window;
	int width,height;
	GLuint texture_framebuffer_id;
	GLuint shader_program_id;
	GLuint tex_uniform_loc;
	unsigned char *data;
	GLuint vertexbuffer;
	GLuint uvbuffer;
	GLuint VertexArrayID;
	static const GLfloat g_vertex_buffer_data[18];
	static const GLfloat g_uv_buffer_data[18];
	bool init_context(int x,int y, void (*f)(int));
	bool terminate_context();
	void start_render_loop();
	void draw();
	void (*input_handle)(int);
};

const GLfloat Context::g_vertex_buffer_data[18] = {
		-1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		-1.0f,  -1.0f, 0.0f,

		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f,  1.0f, 0.0f
	};

const GLfloat Context::g_uv_buffer_data[18] = {
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f,  0.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f,  1.0f
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

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	window = glfwCreateWindow(x, y, "VOCU", NULL, NULL);
	if (!window){
		std::cout << "Window or OpenGL context creation failed" << std::endl;
	}

	glfwSetKeyCallback(window,key_callback);
	glfwSetWindowUserPointer(window, this);

	glfwMakeContextCurrent(window);

	//setup framebuffer texture
	data = new unsigned char [width*height*4];
	for(int i = 0; i < width*height*4;i += 3){
	        data[i] = 255;
	    }

	glGenTextures(1, &texture_framebuffer_id);

	// "Bind" the newly created texture : all future texture functions will modify this texture
	glBindTexture(GL_TEXTURE_2D, texture_framebuffer_id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

	shader_program_id = load_shader("src/shader/SimpleVertexShader.vertexshader", "src/shader/SimpleFragmentShader.fragmentshader" );
	tex_uniform_loc = glGetUniformLocation(shader_program_id, "myTextureSampler");

	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	glGenBuffers(1, &uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data), g_uv_buffer_data, GL_STATIC_DRAW);

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glClearColor(0.0f, 0.0f, 0.3f, 0.0f);

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
		glClear( GL_COLOR_BUFFER_BIT );
		draw();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	std::cout << "GL Ended." << std::endl;
}

void Context::draw(){

	glUseProgram(shader_program_id);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_framebuffer_id);
	glUniform1i(tex_uniform_loc, 0);

	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(
						  0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
						  3,                  // size
						  GL_FLOAT,           // type
						  GL_FALSE,           // normalized?
						  0,                  // stride
						  (void*)0            // array buffer offset
						  );

	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glVertexAttribPointer(
						  1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
						  2,                                // size : U+V => 2
						  GL_FLOAT,                         // type
						  GL_FALSE,                         // normalized?
						  0,                                // stride
						  (void*)0                          // array buffer offset
						  );

	glDrawArrays(GL_TRIANGLES, 0, 6);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

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
