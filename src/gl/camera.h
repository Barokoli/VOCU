/*
 * camera.h
 *
 *  Created on: 10.01.2017
 *      Author: sebastian
 */

#ifndef CAMERA_H_
#define CAMERA_H_

#include "extern/Eigen/Dense"
#include "dio/memory.h"

using namespace Eigen;

class Camera{
public:
	Vector3f position;
    Vector3f forward;
    Vector3f right;
    Vector3f up;
    Vector3f screen_position;
    Vector3f look_at;

    int width,height;
    float speed;
    float tiltSpeed;

    float fov;
    float clipNear;
    float clipFar;

    bool drag;
    float dragStart[2];

    Memory<float> kernelParams;
    Camera();
    Camera(int,int);
    Vector3f toFov(Vector3f,bool);
    void rotateSS (float deltaX,float deltaY);
    void updateParams();
    void init_cuda_mem();
    void free();
};
Camera::Camera(){

}
Camera::Camera (int screenX,int screenY){
	width = screenX;
	height = screenY;

	speed = 1;
	tiltSpeed = 4;

    position = Vector3f(20,20,250);

    look_at = Vector3f(64,64,60);

    forward = look_at-position;

    forward.normalize();

    //forward = Vector3f(1,1,-30);
    right = forward.cross(Vector3f(0,0,1));
    up = right.cross(forward);

    std::cout << "Look direction vector: ("<< (float)forward[0] << ","<< (float)forward[1] << ","<< (float)forward[2] << ")" << std::endl;
    //cross cross
    fov = 60*0.0174533;

    forward.normalize();
    right.normalize();
    up.normalize();


    right = toFov(right,true);
    up = toFov(up,false);

    screen_position = forward - right*0.5*width - up*0.5*height;

	clipNear = 0.01;
	clipFar = 1000;
	std::cout << "up direction vector: ("<< (float)up[0] << ","<< (float)up[1] << ","<< (float)up[2] << ")" << std::endl;
	std::cout << "right direction vector: ("<< (float)right[0] << ","<< (float)right[1] << ","<< (float)right[2] << ")" << std::endl;
	std::cout << "forward direction vector: ("<< (float)forward[0] << ","<< (float)forward[1] << ","<< (float)forward[2] << ")" << std::endl;
	std::cout << "screen_position direction vector: ("<< (float)screen_position[0] << ","<< (float)screen_position[1] << ","<< (float)screen_position[2] << ")" << std::endl;

	/*new_cuda_mem<float>(&kernelParams,18);
	updateParams();*/
}
Vector3f Camera::toFov(Vector3f vctr,bool w){
  if(w){
    return vctr*((float)2.0*tan(fov*0.5)/width);
  }else{
	  //std::cout << (float)2.0*tan(((float)height/width)*fov*0.5)/height << std::endl;
	return vctr*((float)2.0*tan(((float)height/(float)width)*fov*0.5)/(float)height);
  }
}

void Camera::init_cuda_mem(){
	new_cuda_mem<float>(&kernelParams,18);
	updateParams();
}

void Camera::rotateSS(float deltaX,float deltaY){
	forward = forward*(1-(tiltSpeed*deltaX/(4.0f*width)+tiltSpeed*deltaY/(4.0f*height)))+(-right*deltaX)+up*deltaY;

	forward.normalize();

	right = forward.cross(Vector3f(0,0,1));
	up = right.cross(forward);

	right.normalize();
	up.normalize();

	right = toFov(right,true);
	up = toFov(up,false);

	screen_position = forward - right*0.5*width - up*0.5*height;

	updateParams();
}

void Camera::updateParams(){
	kernelParams.h_data[0] = position[0];
	kernelParams.h_data[1] = position[1];
	kernelParams.h_data[2] = position[2];

	kernelParams.h_data[3] = forward[0];
	kernelParams.h_data[4] = forward[1];
	kernelParams.h_data[5] = forward[2];

	kernelParams.h_data[6] = up[0];
	kernelParams.h_data[7] = up[1];
	kernelParams.h_data[8] = up[2];

	kernelParams.h_data[9] = right[0];
	kernelParams.h_data[10] = right[1];
	kernelParams.h_data[11] = right[2];

	kernelParams.h_data[12] = screen_position[0];
	kernelParams.h_data[13] = screen_position[1];
	kernelParams.h_data[14] = screen_position[2];

	kernelParams.h_data[15] = fov;
	kernelParams.h_data[16] = clipNear;
	kernelParams.h_data[17] = clipFar;

	kernelParams.memcpy_htd();
}

void Camera::free(){
	kernelParams.mem_free();
}


#endif /* CAMERA_H_ */
