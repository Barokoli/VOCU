/*
 * octree_kernel.cuh
 *
 *  Created on: 08.11.2016
 *      Author: sebastian
 */

#ifndef OCTREE_KERNEL_CUH_
#define OCTREE_KERNEL_CUH_
#define EMPTY_MASK		0x00000000
#define META_MASK 		0xC0000000
#define INV_META_MASK	0x3FFFFFFF
#define NODE_MASK 		0x80000000

int GetBlock(int x, int y, int z, float* rn, int noise_count, int noise_size);

__global__ void k_octree_fill_blocks(int *bulk_storage, float *rn,int noise_count,int noise_size,int x_off,int y_off,int z_off){
	const int3 global_size = make_int3(blockDim.x*gridDim.x,
			blockDim.y*gridDim.y,
			blockDim.z*gridDim.z);

	//1D Block ID
	const int blockId = blockIdx.x
			+ blockIdx.y * gridDim.x
			+ gridDim.x * gridDim.y * blockIdx.z;

	//1D Coords 8th of Octree (skips every second in each dimension) Grid-space
	const int global_id = blockId * (blockDim.x * blockDim.y * blockDim.z)
			+ (threadIdx.z * (blockDim.x * blockDim.y))
			+ (threadIdx.y * blockDim.x)
			+ threadIdx.x;

	//3D Coords 8th
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int size = global_size.x*global_size.y*global_size.z;
	//Octree Abs 1D id
	int id = (x+y*global_size.x+z*global_size.x*global_size.y)*8;

	int a,b,c,d,e,f,g,h;//block pairs of 8

	//Octree space Global 1D ID | Conversion from 3D Grid-space to 1D index
	int globId =	(
						(x>>1)
						+ ((y>>1)*global_size.x)>>1
						+ ((z>>1)*global_size.x*global_size.y)>>2
					)*8
					+ (x%2+((y%2)<<1)+((z%2)<<2));

	a = GetBlock(x*2+x_off  ,y*2+y_off  ,z*2+z_off  ,rn,noise_count,noise_size);
	b = GetBlock(x*2+x_off+1,y*2+y_off  ,z*2+z_off  ,rn,noise_count,noise_size);
	c = GetBlock(x*2+x_off  ,y*2+y_off+1,z*2+z_off  ,rn,noise_count,noise_size);
	d = GetBlock(x*2+x_off+1,y*2+y_off+1,z*2+z_off  ,rn,noise_count,noise_size);
	e = GetBlock(x*2+x_off  ,y*2+y_off  ,z*2+z_off+1,rn,noise_count,noise_size);
	f = GetBlock(x*2+x_off+1,y*2+y_off  ,z*2+z_off+1,rn,noise_count,noise_size);
	g = GetBlock(x*2+x_off  ,y*2+y_off+1,z*2+z_off+1,rn,noise_count,noise_size);
	h = GetBlock(x*2+x_off+1,y*2+y_off+1,z*2+z_off+1,rn,noise_count,noise_size);

	if(a == b &&a == c &&a == d &&a == e &&a == f &&a == g &&a == h){
		bulk_storage[id  ] =(int) EMPTY_MASK;
		bulk_storage[id+1] =(int) EMPTY_MASK;
		bulk_storage[id+2] =(int) EMPTY_MASK;
		bulk_storage[id+3] =(int) EMPTY_MASK;
		bulk_storage[id+4] =(int) EMPTY_MASK;
		bulk_storage[id+5] =(int) EMPTY_MASK;
		bulk_storage[id+6] =(int) EMPTY_MASK;
		bulk_storage[id+7] =(int) EMPTY_MASK;

		bulk_storage[(size<<3)+ globId] = (int) (META_MASK)|(INV_META_MASK&a);
	}else{
		bulk_storage[id  ] =(int) (META_MASK)|(INV_META_MASK&a);
		bulk_storage[id+1] =(int) (META_MASK)|(INV_META_MASK&b);
		bulk_storage[id+2] =(int) (META_MASK)|(INV_META_MASK&c);
		bulk_storage[id+3] =(int) (META_MASK)|(INV_META_MASK&d);
		bulk_storage[id+4] =(int) (META_MASK)|(INV_META_MASK&e);
		bulk_storage[id+5] =(int) (META_MASK)|(INV_META_MASK&f);
		bulk_storage[id+6] =(int) (META_MASK)|(INV_META_MASK&g);
		bulk_storage[id+7] =(int) (META_MASK)|(INV_META_MASK&h);

		bulk_storage[(size<<3)+ globId] =(int) (NODE_MASK)|(INV_META_MASK&id);
	}
}

//0-127 = leer | 128-255 = voll
__device__ int GetBlock(int x, int y, int z, float* rn, int noise_count, int noise_size){

    int xf,yf,zf,xpf,ypf,zpf,nSsqr;
    nSsqr = noise_size*noise_size;
    float xv,yv,zv;
    float value = 0.0f;
    float3 noise_layers = make_float3(20.0f,4.0f,1.0f);
    float3 noise_layer_weight = make_float3(40.0f,15.0f,1.2f);

    //Layer the different noises to get an interesting surface
    for(int noiseLayer = 0; noiseLayer < 3; noiseLayer++){
    	//Trilinear filtering
		xv = (float)x/((float)noise_size*noise_layers.x);//*7.654321f;
		yv = (float)y/((float)noise_size*noise_layers.x);//*7.654321f;
		zv = (float)z/((float)noise_size*noise_layers.x);//*7.654321f;
		xf = floor(xv);
		yf = floor(yv);
		zf = floor(zv);
		xv -= xf;
		yv -= yf;
		zv -= zf;
		xf = xv*(float)noise_size;
		yf = yv*(float)noise_size;
		zf = zv*(float)noise_size;
		xpf = xf<(noise_size-1)?xf+1:0;
		ypf = yf<(noise_size-1)?yf+1:0;
		zpf = zf<(noise_size-1)?zf+1:0;
		float a,b,c,d,e,f,g,h;
		a = rn[xf +yf *noise_size+zf *nSsqr];
		b = rn[xpf+yf *noise_size+zf *nSsqr];
		c = rn[xf +ypf*noise_size+zf *nSsqr];
		d = rn[xpf+ypf*noise_size+zf *nSsqr];
		e = rn[xf +yf *noise_size+zpf*nSsqr];
		f = rn[xpf+yf *noise_size+zpf*nSsqr];
		g = rn[xf +ypf*noise_size+zpf*nSsqr];
		h = rn[xpf+ypf*noise_size+zpf*nSsqr];
		value += noise_layer_weight.x*(zv*(a*xv+b*(1.0f-xv))+(1.0f-zv)*(c*yv+d*(1.0f-yv)));
		value = ((float)((float)(127-z)/64.0f)-1.0f)+value*0.2;
    }

    //Clamping
    value = value >= 0.5f?1.0f:value;
    value = value < 0.5f?-1.0f:value;

    return (int)(127.5f*(value+1.0f));
}


__global__ void testKernel(float *g_idata, float *g_odata)
{
    // shared memory
    // the size is determined by the host application
    extern  __shared__  float sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // read in input data from global memory
    sdata[tid] = g_idata[tid];
    __syncthreads();

    // perform some computations
    sdata[tid] = (float) num_threads * sdata[tid];
    __syncthreads();

    // write data to global memory
    g_odata[tid] = sdata[tid];
}



#endif /* OCTREE_KERNEL_CUH_ */