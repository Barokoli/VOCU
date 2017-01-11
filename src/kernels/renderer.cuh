/*
 * renderer.cuh
 *
 *  Created on: 07.01.2017
 *      Author: sebastian
 */

#ifndef RENDERER_CUH_
#define RENDERER_CUH_

#include "dio/misc.h"
#include "extern/helper_math.h"

__device__ static uint EvaluateBlock(int relX,int relY,int relZ,int *chunk,int lastI,float *blockSize);
__device__ static uint to_uint(char4 col);
__device__ static float4 cast_ray(float4 origin, float4 dir, int *chunk,int chunkL);

__global__ void k_render_to_buffer(uint *buffer,int *octree,int octree_size,float *cam_info){
	int u = threadIdx.x+blockIdx.x*blockDim.x;
	int v = threadIdx.y+blockIdx.y*blockDim.y;
	int idx = threadIdx.x+blockIdx.x*blockDim.x+(threadIdx.y+blockIdx.y*blockDim.y)*gridDim.x*blockDim.x;

	float4 cam_pos = make_float4(cam_info[0],cam_info[1],cam_info[2],0.0f);
	float4 cam_forward = make_float4(cam_info[3],cam_info[4],cam_info[5],0.0f);
	float4 cam_up = make_float4(cam_info[6],cam_info[7],cam_info[8],0.0f);
	float4 cam_right = make_float4(cam_info[9],cam_info[10],cam_info[11],0.0f);
	float4 screen_pos = make_float4(cam_info[12],cam_info[13],cam_info[14],0.0f);
	float fov = cam_info[15];
	float clipping_near = cam_info[16];
	float clipping_far = cam_info[17];

	float4 camera_ray_dir = screen_pos + cam_right * (float)u + cam_up * (float)v;

	//char4 color = make_char4((char)(cam_pos.x*255),(char)(cam_pos.y*255),(char)(camera_ray_dir.z*255.0f),0);

	float4 ray = cast_ray(cam_pos,camera_ray_dir,octree,octree_size);
	//float4 ray;
	if(ray.x < 0){
		ray.x = 0;
	}
	char4 color = make_char4((char)ray.x,(char)ray.x,(char)ray.x,0);

	float tmp = CHUNK_SIZE;

	if(u < CHUNK_SIZE && v < CHUNK_SIZE){
		uint b = EvaluateBlock(50,u,v,octree,octree_size,&tmp);
		color = make_char4((char)log2((tmp)*31),(char)log2((tmp)*31),(char)(log2(tmp)*31),0);
		if(b > 0 & b < 255){
			color.x = 0x0000FF;
		}
		//color = make_char4((char)) ;//> 127 ? 0xFFFFFF : 0x000000;//v+(u<<16):v+(u<<16);//0xFFFFFF : 0x000000;
	}
	buffer[idx] = to_uint(color);//EvaluateBlock(cam_pos.x,cam_pos.y,cam_pos.z,octree,octree_size,&tmp);//to_uint(color);//cam_pos.x>0?0xFFFF00:0xFF0000;//to_uint(color);
}

//Returns Ray information: Length, surface normal
__device__ static float4 cast_ray(float4 origin, float4 dir, int *chunk,int chunkL){
	float depth = 0;
	float rx,ry,rz;//,cx,cy,cz;
	float mag;
	int4 blockOrigin = make_int4((int)origin.x,(int)origin.y,(int)origin.z,1);
	float4 lastNormal;// = (float4)(0.0f,0.0f,0.0f,0.0f);
	float currBlockSize = (float)(CHUNK_SIZE>>1);//TODO: Hardcoded BlockSize
	while(EvaluateBlock((int)blockOrigin.x,(int)blockOrigin.y,(int)blockOrigin.z,chunk,chunkL,&currBlockSize)<127){
		lastNormal = make_float4(0,0,0,0);
		rx = dir.x!=0.0f? (float)(
								   dir.x>0.0f?
								   ( -(float)fracf(origin.x) + (currBlockSize)-(blockOrigin.x%(int)currBlockSize) )/(dir.x):
								   (fracf(origin.x) == 0.0f? -currBlockSize/dir.x:(float)-(fracf(origin.x)+(blockOrigin.x%(int)currBlockSize))/(dir.x))

								   )
							:100.0f;
				ry = dir.y!=0.0f? (float)(
								   dir.y>0.0f?
										 (-(float)fracf(origin.y)+(currBlockSize)-(blockOrigin.y%(int)currBlockSize))/(dir.y):
										 (fracf(origin.y) == 0.0f? -currBlockSize/dir.y:(float)-(fracf(origin.y)+(blockOrigin.y%(int)currBlockSize))/(dir.y))
								   )
							:100.0f;
				rz = dir.z!=0.0f? (float)(
								   dir.z>0.0f?
										 (-(float)fracf(origin.z)+(currBlockSize)-(blockOrigin.z%(int)currBlockSize))/(dir.z):
										 (fracf(origin.z) == 0.0f? -currBlockSize/dir.z:(float)-(fracf(origin.z)+(blockOrigin.z%(int)currBlockSize))/(dir.z))
								   )
							:100.0f;

		rx = rx>0.0f?rx:10000.0f;
		ry = ry>0.0f?ry:10000.0f;
		rz = rz>0.0f?rz:10000.0f;

		mag = rx<ry?(rx<rz?rx:rz):(ry<rz?ry:rz);
		origin = origin+dir*mag;
		depth += mag;

		if(rx<ry){
		  if(rx < rz){
			  //origin = origin+dir*rx;
			  blockOrigin = make_int4((dir.x > 0.0f ? floor(origin.x-0.5f)+1 : floor(origin.x+0.5f)-1),floor(origin.y),floor(origin.z),0);
			  lastNormal.x += dir.x > 0.0f? -1.0f:1.0f;
			  //depth += rx;
		  }else{
			  //origin = origin+dir*rz;
			  blockOrigin = make_int4(floor(origin.x),floor(origin.y),(dir.z > 0.0f ? floor(origin.z-0.5f)+1 : floor(origin.z+0.5f)-1),0);
			  lastNormal.z += dir.z > 0.0f? -1.0f:1.0f;
			  //depth += rz;
		  }
		}else{
		  if(ry<rz){
			  //origin = origin+dir*ry;
			  blockOrigin = make_int4(floor(origin.x),(dir.y > 0.0f ? floor(origin.y-0.5f)+1 : floor(origin.y+0.5f)-1),floor(origin.z),0);
			  lastNormal.y += dir.y > 0.0f? -1.0f:1.0f;
			  //depth += ry;
		  }else{
			  //origin = origin+dir*rz;
			  blockOrigin = make_int4(floor(origin.x),floor(origin.y),(dir.z > 0.0f ? floor(origin.z-0.5f)+1 : floor(origin.z+0.5f)-1),0);
			  lastNormal.z += dir.z > 0.0f? -1.0f:1.0f;
			  //depth += rz;
		  }
		}

		if(!(blockOrigin.x > 0 && blockOrigin.x < CHUNK_SIZE &&
			  blockOrigin.y > 0 && blockOrigin.y < CHUNK_SIZE &&
			  blockOrigin.z > 0 && blockOrigin.z < CHUNK_SIZE)){
		  return make_float4(-1,dir.x,dir.y,dir.z);
		}
	}

	return make_float4(depth,lastNormal.x,lastNormal.y,lastNormal.z);
}

__device__ static uint EvaluateBlock(int relX,int relY,int relZ,int *chunk,int lastI,float *blockSize){
    float4 bPos = make_float4(0,0,0,0);
    uint block = chunk[lastI-1];

    uint off = 0;

    uint lvl = CHUNK_SIZE>>1;

    while((block&0xC0000000) != 0xC0000000){
        off |= (uint)(relX>=(bPos.x+lvl)? 1:0);
        off |= (uint)(relY>=(bPos.y+lvl)? 2:0);
        off |= (uint)(relZ>=(bPos.z+lvl)? 4:0);
        block = chunk[(block&0x3FFFFFFF)+off];
        bPos = make_float4((off&1)*lvl+bPos.x,((off&2)>>1)*lvl+bPos.y,((off&4)>>2)*lvl+bPos.z,0.0);
        *blockSize = (float)(lvl);
        lvl = lvl >> 1;
        off = 0;
    }

    return block&0x3FFFFFFF;
}

#endif /* RENDERER_CUH_ */
