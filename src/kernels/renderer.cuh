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

__device__ static int EvaluateBlock(int relX,int relY,int relZ,int *chunk,int lastI,float *blockSize);
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
	camera_ray_dir = normalize(camera_ray_dir);

	float4 ray = cast_ray(cam_pos,camera_ray_dir,octree,octree_size);//make_float4(1,camera_ray_dir.x*255,camera_ray_dir.y*255,camera_ray_dir.z*255);//
	//float4 ray;
	if(ray.x < 0){
		ray.x = 0;
	}
	char4 color = make_char4((char)ray.x,(char)ray.x,(char)ray.x,0);
	//char4 color = make_char4((char)ray.y*100,(char)ray.z*100,(char)ray.w*100,0);

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
	float mag;
	int4 blockOrigin = make_int4((int)origin.x,(int)origin.y,(int)origin.z,1);
	float4 lastNormal = make_float4(0.0f,0.0f,0.0f,0.0f);
	float currBlockSize = (float)(CHUNK_SIZE>>1);//TODO: Hardcoded BlockSize

	float4 deltaDist;
	deltaDist.x = 1.0f/fabs(dir.x);
	deltaDist.y = 1.0f/fabs(dir.y);
	deltaDist.z = 1.0f/fabs(dir.z);

	float4 sideDist;
	sideDist.x = (copysignf((float)blockOrigin.x-(float)origin.x,dir.x) + copysignf(0.5f,dir.x) + 0.5f)*deltaDist.x;
	sideDist.y = (copysignf((float)blockOrigin.y-(float)origin.y,dir.y) + copysignf(0.5f,dir.y) + 0.5f)*deltaDist.y;
	sideDist.z = (copysignf((float)blockOrigin.z-(float)origin.z,dir.z) + copysignf(0.5f,dir.z) + 0.5f)*deltaDist.z;

	float4 stepSize = make_float4(1.0f,1.0f,1.0f,0);

	int3 b1,b2,mask;

	for (int i = 0; i < MAX_RAY_STEPS; i++) {
		if (EvaluateBlock((int)blockOrigin.x,(int)blockOrigin.y,(int)blockOrigin.z,chunk,chunkL,&currBlockSize)>127) continue;

		b1.x = sideDist.x < sideDist.y;
		b1.y = sideDist.y < sideDist.z;
		b1.z = sideDist.z < sideDist.x;

		b2.x = sideDist.x <= sideDist.z;
		b2.y = sideDist.y <= sideDist.x;
		b2.z = sideDist.z <= sideDist.y;

		mask.x = b1.x && b2.x;
		mask.y = b1.y && b2.y;
		mask.z = b1.z && b2.z;

		stepSize.x = currBlockSize-fmod((float)blockOrigin.x,currBlockSize)+1;
		stepSize.y = currBlockSize-fmod((float)blockOrigin.y,currBlockSize)+1;
		stepSize.z = currBlockSize-fmod((float)blockOrigin.z,currBlockSize)+1;

		sideDist.x += mask.x * deltaDist.x;// * stepSize.x;
		sideDist.y += mask.y * deltaDist.y;// * stepSize.y;
		sideDist.z += mask.z * deltaDist.z;// * stepSize.z;

		blockOrigin.x += copysignf(mask.x,dir.x);// * (int)stepSize.x;
		blockOrigin.y += copysignf(mask.y,dir.y);// * (int)stepSize.y;
		blockOrigin.z += copysignf(mask.z,dir.z);// * (int)stepSize.z;

		depth += 1;//length(make_float3(mask.x * deltaDist.x,mask.y * deltaDist.y,mask.z * deltaDist.z));

		if(!(blockOrigin.x > 0 && blockOrigin.x < CHUNK_SIZE &&
			  blockOrigin.y > 0 && blockOrigin.y < CHUNK_SIZE &&
			  blockOrigin.z > 0 && blockOrigin.z < CHUNK_SIZE)){
		  return make_float4(-1,mask.x,mask.y,mask.z);
		}
	}

	return make_float4(depth,mask.x,mask.y,mask.z);
}

__device__ static int EvaluateBlock(int relX,int relY,int relZ,int *chunk,int lastI,float *blockSize){
    float4 bPos = make_float4(0,0,0,0);
    int block = chunk[lastI-1];

    int off = 0;

    int lvl = CHUNK_SIZE>>1;

    while((block&0xC0000000) != 0xC0000000){
        off |= (int)(relX>=(bPos.x+lvl)? 1:0);
        off |= (int)(relY>=(bPos.y+lvl)? 2:0);
        off |= (int)(relZ>=(bPos.z+lvl)? 4:0);
        block = chunk[(block&0x3FFFFFFF)+off];
        bPos = make_float4((off&1)*lvl+bPos.x,((off&2)>>1)*lvl+bPos.y,((off&4)>>2)*lvl+bPos.z,0.0);
        *blockSize = (float)(lvl);
        lvl = lvl >> 1;
        off = 0;
    }

    return block&0x3FFFFFFF;
}

#endif /* RENDERER_CUH_ */
