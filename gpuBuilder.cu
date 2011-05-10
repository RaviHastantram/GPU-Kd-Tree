#include "kdtree.h"
#include "kdtypes.h"
#include "lists.h"
#include "gpuBuilder.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#incluce <cfloat>

using namespace std;

///////////////////////////
// 
// Tree Building
//
///////////////////////////
uint32 getActiveNodes()
{
	uint32 numNodes=0;
	 
	if(cudaMemcpyFromSymbol(&numNodes,
				"d_numActiveNodes",
				sizeof(uint32),
				0,
				cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Copying d_numActiveNodes failed.\n");
	}
	return numNodes;
}

uint32 getThreadsPerNode(int numActiveNodes,int numActiveTriangles)
{
	return 0;
}

uint32 getActiveTriangles()
{
	uint32 numTriangles=0;
	
	if(cudaMemcpyFromSymbol(&numTriangles,
				"d_numActiveTriangles",
				sizeof(uint32),
				0,
				cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Copying d_numActiveTriangles failed.\n");
	}
	return numTriangles;
}

__device__ void computeCost()
{
	__shared__ float mins[MAX_BLOCK_SIZE];
	__shared__ float maxs[MAX_BLOCK_SIZE];
	
	uint32 min=FLT_MAX;
	uint32 max=FLT_MIN;
 
	uint32 dim = blockIdx.x % 3;
	uint32 nodeIdx = blockIdx.x + d_activeOffset;
	GPUNode * node = d_gpuNodes.getNode(nodeIdx);
	
	uint32 * triangleIDs= gpuTriangleList.getList(node->primBaseIdx);

	mins[threadIdx.x]=FLT_MAX;
	maxs[threadIdx.x]=FLT_MIN;

	uint32 currIdx = threadIdx.x;

	while(currIdx<node->primLength)
	{
		uint32 triangleID =  triangleIDs[currIdx];
		Triangle * triangle = d_triangles[triangleID];
		for(uint32 j=0;j<3;j++)
		{
			uint32 pointID = triangle->ids[j];
			Point * point = d_points[pointID];
			if(point->values[dim]<mins[threadIdx.x])
			{
				mins[threadIdx.x]=point->values[dim];
			}
			if(point->values[dim]>maxs[threadIdx.x])
			{
				maxs[threadIdx.x]=point->values[dim];
			}
		}
		currIdx += blockDim;
	}

	__syncthreads();

	if(threadIdx.x==0)
	{
		for(uint32 k=0;k<blockDim;k++)
		{
			if(mins[k]<min)
			{
				min=mins[k];
			}
			if(maxs[k]>max)
			{
				max=maxs[k];
			}
		}
		node->splitValue = 0.5*(min+max);
		node->splitChoice = dim;
	}
}

__device__ void splitNodes()
{
	__shared__ uint32 offL[MAX_BLOCK_SIZE];
	__shared__ uint32 offD[MAX_BLOCK_SIZE];
	__shared__ uint32 offR[MAX_BLOCK_SIZE];
	uint32 nodeIdx = blockIdx.x + d_activeOffset;
	GPUNode * node = d_gpuNodes.getNode(nodeIdx);
	int dim = node->splitChoice;
	float splitValue = node->splitValue;
	
	uint32 * triangleIDs= gpuTriangleList.getList(node->primBaseIdx);

	uint32 currIdx = threadIdx.x;
	float low = FLT_MIN;
	float high = FLT_MAX;
	
	//Need to initialize the offL, offD, offR arrays 
	while(currIdx<node->primLength)
	{
		uint32 triangleID =  triangleIDs[currIdx];
		Triangle * triangle = d_triangles[triangleID];
		for(uint32 j=0;j<3;j++)
		{
			uint32 pointID = triangle->ids[j];
			Point * point = d_points[pointID];
			if(point->values[dim]<low)
			{
				low=point->values[dim];
			}
			if(point->values[dim]>high)
			{
				high=point->values[dim];
			}
		}
		if( low < splitValue && high < splitValue ) offL[currIdx] = 1;
		if( low >= splitValue && high >= splitValue) offR[currIdx] = 1;
		if( low < splitValue && hight >= splitValue ) offD[currIdx] = 1;
		__syncthreads();
	}


}

////////////////////////////////
//
// Data Import/Export
//
///////////////////////////////
void copyToGPU(Mesh *mesh)
{
	//Copy the Points list
	int size = sizeof(Point)*(mesh->numPoints);
	cudaMalloc(&d_points,size);
	cudaMemcpy(d_points,mesh->points,size,cudaMemcpyHostToDevice);

	//Copy the triangle list
	size = sizeof(Triangle)*(mesh->numTriangles);
	cudaMalloc(&d_triangles, size);
	cudaMemcpy(d_triangles,mesh->triangles,size,cudaMemcpyHostToDevice);

	//Copy the mesh
	size = sizeof(Mesh);
	cudaMalloc(&d_mesh,size);
	cudaMemcpy(d_mesh,mesh,size,cudaMemcpyHostToDevice);
}




