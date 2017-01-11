/*
 * glContext.h
 *
 *  Created on: 03.11.2016
 *      Author: sebastian
 */

#ifndef GLCONTEXT_H_
#define GLCONTEXT_H_

#include "dio/memory.h"
#include <GLFW/glfw3.h>
#include <OpenGL/gl3.h>
#include <cuda_gl_interop.h>
#include "shader/load_shader.h"
#include "extern/helper_cuda.h"
#include "renderer.cuh"
#include "gl/camera.h"

void error_callback(int error, const char* description);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void key_poll_callback(GLFWwindow* window);

texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> frame_texture_cuda_ref;

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
	GLuint pixelBufferID;
	GLuint VertexArrayID;
	Memory<int> *objects_to_draw;
	static const GLfloat g_vertex_buffer_data[18];
	static const GLfloat g_uv_buffer_data[18];
	double mouse_pos_x, mouse_pos_y;

	Camera *active_camera;

	struct cudaGraphicsResource *frame_texture_cuda_resource;

	bool init_context(int x,int y, void (*f)(int,int), void (*f2)(GLFWwindow*),Camera *cam);
	bool terminate_context();
	void start_render_loop();
	void draw();
	void (*input_handle)(int,int);
	void (*input_poll_handle)(GLFWwindow *window);
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

bool Context::init_context(int x,int y, void (*f)(int,int),void (*f2)(GLFWwindow*),Camera *cam){
	width = x;
	height = y;
	input_handle = f;
	input_poll_handle = f2;

	active_camera = cam;

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

	glGenBuffers(1, &pixelBufferID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferID);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(uint), nullptr, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&frame_texture_cuda_resource, pixelBufferID,
	                                                     cudaGraphicsMapFlagsWriteDiscard));

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
	//checkCudaErrors(cudaFree(d_texture_frame));

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

	glfwGetCursorPos(window, &mouse_pos_x, &mouse_pos_y);
	key_poll_callback(window);
	//Do Cuda stuff:
	if(objects_to_draw){
		uint *dptr;
		checkCudaErrors(cudaGraphicsMapResources(1, &frame_texture_cuda_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, frame_texture_cuda_resource));
		dim3  grid(36,36,1);
		dim3  threads(30, 20, 1);
		//active_camera->kernelParams.memcpy_htd();
		//objects_to_draw->memcpy_htd();
		k_render_to_buffer<<< grid, threads >>>(dptr,objects_to_draw->d_data,objects_to_draw->size,active_camera->kernelParams.d_data);
		// unmap buffer object
		checkCudaErrors(cudaGraphicsUnmapResources(1, &frame_texture_cuda_resource, 0));

		cudaStreamSynchronize(0);
	}
	//end cuda

	glUseProgram(shader_program_id);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_framebuffer_id);

	//Fill texture from PBO
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

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
    	context->input_handle(key,action);
    }
}
void key_poll_callback(GLFWwindow* window)
{
	Context * context = reinterpret_cast<Context *>(glfwGetWindowUserPointer(window));
	context->input_poll_handle(window);
}

#endif /* GLCONTEXT_H_ */
