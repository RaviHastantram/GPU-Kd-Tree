#ifndef __GPU_TRIANGLE_LIST_H
#ifdef __GPU_TRIANGLE_LIST_H

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

	__device uint32* getList(uint32 primBaseIdx)
	{
		return &triangles[primBaseIdx];
	}

	__device uint32 allocateList(uint32 primLength)
	{
		uint32 t = nextAvailable;
		nextAvailable += primLength;
		return t;
	}


private:
	uint32  capacity;	
	uint32  nextAvailable;	
	uint32* triangles;
};

#endif
