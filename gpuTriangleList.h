#ifndef __GPU_TRIANGLE_LIST_H__
#define __GPU_TRIANGLE_LIST_H__

#include "kdtypes.h"
#include "util.h"
#include "lock.h"

//a lot of triangles.
#define GPU_TRIANGLE_ARRAY_SIZE 5000000

struct GPUTriangleArray {

	GPUTriangleArray()
	{
		//Allocate memory for the TriangleArray on the GPU
		int size = sizeof(uint32)* GPU_TRIANGLE_ARRAY_SIZE;
		HANDLE_ERROR(cudaMalloc(&triangles,size));

		capacity=GPU_TRIANGLE_ARRAY_SIZE;
		nextAvailable=0;
	
		l = Lock();
	}

	__host__ void destroy()  {
		HANDLE_ERROR(cudaFree(triangles));
		l.destroy();
	}

	// NOTE:	Need to update nextAvailable if it changed on the device side.
	// 		Only works for initialization.
	__host__ uint32 pushList(uint32 * h_triangles, uint32 length)
	{
		uint32 primBase = nextAvailable;
		uint32 * d_base = triangles+primBase;
		HANDLE_ERROR(cudaMemcpy(d_base,h_triangles,sizeof(uint32)*length,cudaMemcpyHostToDevice));
		nextAvailable+=length;
		return primBase;
	}

	__device__ uint32* getList(uint32 primBaseIdx)
	{
		return &triangles[primBaseIdx];
	}

	__host__ void copyList(uint32 * h_list, uint32 primBaseIdx, uint32 primLength)
	{
		uint32 * d_list = triangles+primBaseIdx;
		HANDLE_ERROR(cudaMemcpy(h_list, d_list, primLength*sizeof(uint32),cudaMemcpyDeviceToHost));
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

	uint32  capacity;	
	uint32  nextAvailable;	
	uint32* triangles;
	Lock l;
};

#endif
