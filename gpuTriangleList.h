#ifndef __GPU_TRIANGLE_LIST_H
#define __GPU_TRIANGLE_LIST_H
#include "kdtypes.h"
#include "lock.h"

//20k triangles.
#define GPU_TRIANGLE_ARRAY_SIZE 20000



struct GPUTriangleArray {

public:
	__host__ GPUTriangleArray()
	{
		//Allocate memory for the TriangleArray on the GPU
		int size = sizeof(uint32)* GPU_TRIANGLE_ARRAY_SIZE;
		cudaMalloc(&triangles,size);

		capacity=GPU_TRIANGLE_ARRAY_SIZE;
		nextAvailable=0;
	}

	__host__ uint32 pushList(uint32 * h_triangles, uint32 length)
	{
		uint32 primBase = nextAvailable;
		uint32 * d_base = triangles+primBase;
		cudaMemcpy(d_base,h_triangles,sizeof(uint32)*length,cudaMemcpyHostToDevice);
		nextAvialable+=length;
		return primBase;
	}

	__device__ uint32* getList(uint32 primBaseIdx)
	{
		return &triangles[primBaseIdx];
	}

	__host__ void copyList(uint32 * h_list, uint32 primBaseIdx, uint32 primLength)
	{
		uint32 * d_list = triangles+primBaseIdx;
		cudaMemcpy(h_list, d_list, primLength*sizeof(uint32),cudaMemcpyDeviceToHost);
	}

	__device__ uint32 allocateList(uint32 primLength)
	{
		uint32 t = nextAvailable;
		nextAvailable += primLength;
		return t;
	}

	__device__ void lock()
	{
		l.lock();
	}

	__device__ void unlock()
	{
		l.unlock();
	}

private:
	uint32  capacity;	
	uint32  nextAvailable;	
	uint32* triangles;
	Lock l;
};

#endif
