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

__device__ int GetBlock(int x, int y, int z, float* rn, int noise_count, int noise_size);

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
	//return x;
	//return z > 120 ? 0 : 255;
	int xf,yf,zf,xpf,ypf,zpf,nSsqr;
    nSsqr = noise_size*noise_size;
    float xv,yv,zv;
    float value = 0.0f;
    float noise_values[6] = {	35.0f,0.7f,
    							13.0f,0.2f,
    							5.00f,0.1f};

    //Layer the different noises to get an interesting surface
    for(int noiseLayer = 0; noiseLayer < noise_count; noiseLayer++){
    	//Trilinear filtering
		xv = fmod((float)x/((float)noise_values[noiseLayer*2]),(float)noise_size);//*7.654321f;
		yv = fmod((float)y/((float)noise_values[noiseLayer*2]),(float)noise_size);//*7.654321f;
		zv = fmod((float)z/((float)noise_values[noiseLayer*2]),(float)noise_size);//*7.654321f;

		xf = floor(xv);
		yf = floor(yv);
		zf = floor(zv);

		xv = xv-(float)xf;
		yv = yv-(float)yf;
		zv = zv-(float)zf;

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
		value += noise_values[noiseLayer*2+1]*(
						(1.0f-zv)*(
							(1.0f-yv)*(xv*b+(1.0f-xv)*a)
							+
							(yv)*(xv*d+(1.0f-xv)*c)
						)
						+
						zv*(
							(1.0f-yv)*(xv*f+(1.0f-xv)*e)
							+
							(yv)*(xv*h+(1.0f-xv)*g)
						)
					);

		//value += xv*b+(1.0f-xv)*a;
				//(zv*(a*xv+b*(1.0f-xv))+(1.0f-zv)*(c*yv+d*(1.0f-yv)));
		//value = value*0.1f;//((float)((float)(127-z)/64.0f)-1.0f);//+value*0.1;
		//value = a;
    }
    value += 1.0f-((float)z/128.0f);
    //Clamping
    value = value >= 0.506f?1.0f:value;
    value = value < 0.494f?0.0f:value;

    return (int)(127.5f*(value*2.0f));
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
__global__ void k_blelloch_scan_and_pack(int* bulk_storage,int* scan_mem,int nullMem){
	extern __shared__ int shared_mem[];

	int thread_id_2 = threadIdx.x*2;
	//1D Index
	int block_off = blockIdx.x*blockDim.x*2;
	int glob_id_2 = (threadIdx.x*2+block_off);
	int offset = 1;
	int dim = blockDim.x*2;
	int total_sum = 0;

	if(glob_id_2 >= nullMem){
		bulk_storage[glob_id_2] = 0;
		bulk_storage[glob_id_2+1] = 0;
	}

	//fill shared Memory 2-way Bank conflict (unavoidable?)
	shared_mem[thread_id_2] = (bulk_storage[glob_id_2]&META_MASK)?1:0;
	shared_mem[thread_id_2+1] = (bulk_storage[glob_id_2+1]&META_MASK)?1:0;

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
		total_sum = shared_mem[dim-1];
		//shared_mem[dim*2] = shared_mem[dim-1];
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

	//write to scan memory
	scan_mem[glob_id_2] = shared_mem[thread_id_2];
	scan_mem[glob_id_2+1] = shared_mem[thread_id_2+1];

	if(threadIdx.x == 0){
		scan_mem[block_off] = total_sum;
	}

	//Rearrange Array
	//Either load copy from global mem to shared coalesced write Random or Read Random write coalesced | Writing coalesced is supposed to be quicker due to easier caching
	//fill shared Memory 2-way Bank conflict (unavoidable?)
	/*shared_mem[dim+thread_id_2] = bulk_storage[block_off+shared_mem[thread_id_2]];
	shared_mem[dim+thread_id_2+1] = bulk_storage[block_off+shared_mem[thread_id_2+1]];

	__syncthreads();
	bulk_storage[block_off+thread_id_2] = shared_mem[dim+thread_id_2];
	bulk_storage[block_off+thread_id_2+1] = shared_mem[dim+thread_id_2+1];*/

	/*__syncthreads();
	if(threadIdx.x == 0) {
		bulk_storage[block_off+blockDim.x*2-1] = shared_mem[dim-1] < (dim-1) ? shared_mem[dim*2] : bulk_storage[block_off+blockDim.x*2-1];
	}*/
	//last element is equal to element count if not full
}

__global__ void k_copy_packed_mem(int *bulk_storage,int *scan_mem,int *res_mem){
	int block_off = blockIdx.x*blockDim.x*2;
	int local_thread_id = threadIdx.x*2;
	int thread_id = block_off+local_thread_id;
	//int block_size = blockDim.x*2;
	int off = scan_mem[block_off];//bulk_storage[block_off+block_size-1];

	int v1 = bulk_storage[thread_id];
	int v2 = bulk_storage[thread_id+1];

	if(threadIdx.x != 0){
		scan_mem[thread_id] += off;
		scan_mem[thread_id+1] += off;
	}

	__syncthreads();

	if(v1&META_MASK){
		res_mem[scan_mem[thread_id]] = (v1&FILL_MASK) ? v1 : (scan_mem[(v1&INV_META_MASK)]|NODE_MASK);
	}
	if(v2&META_MASK){
		res_mem[scan_mem[thread_id+1]] = (v2&FILL_MASK) ? v2 : (scan_mem[(v2&INV_META_MASK)]|NODE_MASK);
	}
}

#endif /* OCTREE_KERNEL_CUH_ */
