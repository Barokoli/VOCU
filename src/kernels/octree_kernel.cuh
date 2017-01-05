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
#define FILL_MASK 		0x40000000
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

int GetBlock(int x, int y, int z, float* rn, int noise_count, int noise_size);

//First Stage:
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
	/*int globId =	(
						(x>>1)
						+ ((y>>1)*global_size.x)>>1
						+ ((z>>1)*global_size.x*global_size.y)>>2
					)*8
					+ (x%2+((y%2)<<1)+((z%2)<<2));*/

	int globId = ((((int)(x*0.5f)) + (int)((int)y*0.5f)*global_size.x*0.5f + (int)((int)z*0.5f)*global_size.x*global_size.y*0.25f))*8 + (x%2+((y%2)*2)+((z%2)*4));


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
		//Trilinear Interpolation
		value += noise_layer_weight.x*(
					zv*(
							yv*(xv*a+(1.0f-xv)*b)
							+
							(1.0f-yv)*(xv*c+(1.0f-xv)*d)
						)
					+
					(1.0f-zv)*(
							yv*(xv*e+(1.0f-xv)*f)
							+
							(1.0f-yv)*(xv*g+(1.0f-xv)*h)
						)
				);
				//(zv*(a*xv+b*(1.0f-xv))+(1.0f-zv)*(c*yv+d*(1.0f-yv)));
		value = ((float)((float)(127-z)/64.0f)-1.0f)+value*0.2;
    }

    //Clamping
    value = value >= 0.5f?1.0f:value;
    value = value < 0.5f?-1.0f:value;

    return (int)(127.5f*(value+1.0f));
}

//Second Stage
__global__ void k_build_tree(int* bulk_storage, int Off){//init with 8th of kernels
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

    int a,b,c,d,e,f,g,h;

    int id = (x+y*global_size.x+z*global_size.x*global_size.y)*8;

    int globId = ((((int)(x*0.5f)) + (int)((int)y*0.5f)*global_size.x*0.5f + (int)((int)z*0.5f)*global_size.x*global_size.y*0.25f))*8 + (x%2+((y%2)*2)+((z%2)*4));

    int cubeSize = global_size.x*global_size.y*global_size.z;

    a = bulk_storage[Off+id  ];
    b = bulk_storage[Off+id+1];
    c = bulk_storage[Off+id+2];
    d = bulk_storage[Off+id+3];
    e = bulk_storage[Off+id+4];
    f = bulk_storage[Off+id+5];
    g = bulk_storage[Off+id+6];
    h = bulk_storage[Off+id+7];

    if(a == b &&a == c &&a == d &&a == e &&a == f &&a == g &&a == h&&(a&FILL_MASK)!=0){ //TODO:META_MASK?
    	bulk_storage[Off+id  ] = (int) EMPTY_MASK;
    	bulk_storage[Off+id+1] = (int) EMPTY_MASK;
    	bulk_storage[Off+id+2] = (int) EMPTY_MASK;
    	bulk_storage[Off+id+3] = (int) EMPTY_MASK;
    	bulk_storage[Off+id+4] = (int) EMPTY_MASK;
    	bulk_storage[Off+id+5] = (int) EMPTY_MASK;
    	bulk_storage[Off+id+6] = (int) EMPTY_MASK;
    	bulk_storage[Off+id+7] = (int) EMPTY_MASK;

    	bulk_storage[Off+(cubeSize<<3)+globId] = (int) (META_MASK)|(INV_META_MASK&a);
    }else{
    	bulk_storage[Off+(cubeSize<<3)+globId] = (int) (NODE_MASK)|(INV_META_MASK&(Off+id));
    }
}

//based on: http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
__global__ void k_blelloch_scan_and_pack(int* bulk_storage,int nullMem){
	extern __shared__ int shared_mem[];

	int thread_id_2 = threadIdx.x*2;
	//1D Index
	int block_off = blockIdx.x*blockDim.x;
	int glob_id = threadIdx.x+block_off;
	int offset = 1;
	int sum;
	int dim = blockDim.x*2;

	if(glob_id*2 >= nullMem){
		bulk_storage[glob_id*2] = 0;
		bulk_storage[glob_id*2+1] = 0;
	}
	//fill shared Memory 2-way Bank conflict (unavoidable?)
	shared_mem[thread_id_2] = (bulk_storage[glob_id*2]&META_MASK)!=0?1:0;
	shared_mem[thread_id_2+1] = (bulk_storage[glob_id*2+1]&META_MASK)!=0?1:0;

	for(int d = blockDim.x; d > 0; d >>= 1){
		__syncthreads();

		if(threadIdx.x < d){
			int ai = (thread_id_2+1)*offset-1;
			int bi = (thread_id_2+2)*offset-1;
			shared_mem[bi] += shared_mem[ai];
		}
		offset <<= 1;
	}

	if(threadIdx.x == 0) {
		shared_mem[dim*2] = shared_mem[dim-1];
		shared_mem[dim-1] = 0;
	}

	for(int d = 1; d <= blockDim.x; d <<= 1){
		offset >>= 1;
		__syncthreads();

		if(threadIdx.x < d){
			int ai = (thread_id_2+1)*offset-1;
			int bi = (thread_id_2+2)*offset-1;

			int t = shared_mem[ai];
			shared_mem[ai] = shared_mem[bi];
			shared_mem[bi] += t;
		}
	}
	__syncthreads();

	//Rearrange Array
	//Either load copy from global mem to shared coalesced write Random or Read Random write coalesced | Writing coalesced is supposed to be quicker due to easier caching
	//fill shared Memory 2-way Bank conflict (unavoidable?)
	shared_mem[dim+thread_id_2] = bulk_storage[block_off+shared_mem[thread_id_2]];
	shared_mem[dim+thread_id_2+1] = bulk_storage[block_off+shared_mem[thread_id_2+1]];

	__syncthreads();
	bulk_storage[block_off+thread_id_2] = shared_mem[dim+thread_id_2];
	bulk_storage[block_off+thread_id_2+1] = shared_mem[dim+thread_id_2+1];

	__syncthreads();
	if(threadIdx.x == 0) {
		bulk_storage[block_off+blockDim.x-1] = shared_mem[dim-1] < (dim-1) ? shared_mem[dim*2] : bulk_storage[block_off+blockDim.x-1];
	}
	//last element is equal to element count if not full
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
