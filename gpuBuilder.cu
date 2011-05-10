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

	if(node->nodeDepth>MAX_DEPTH)
	{
		node->splitChoice=SPLIT_NONE;
		node->isLeaf=true;
		return;
	}
	
	uint32 * triangleIDs = gpuTriangleList.getList(node->primBaseIdx);

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
	__shared__ uint32 * leftList;
	__shared__ uint32 * rightList;
	__shared__ uint32 leftPrimBaseIdx;
	__shared__ uint32 rightPrimBaseIdx;

	uint32 triangleChoice;

	uint32 nodeIdx = blockIdx.x + d_activeOffset;
	GPUNode * node = d_gpuNodes.getNode(nodeIdx);
	int dim = node->splitChoice;
	float splitValue = node->splitValue;
	uint32 currIdx = threadIdx.x;
	uint32 * triangleIDs = gpuTriangleList.getList(node->primBaseIdx);
	uint32 leftBase=0, rightBase=0;
	uint32 leftCount=0, rightCount=0;
	
	if(node->isLeaf)
	{
		return;
	}

	if(threadIdx.x==0)
	{
		leftPrimBaseIdx=gpuTriangleList.allocateList(node->primLength);
		rightPrimBaseIdx=gpuTriangleList.allocateList(node->primLength);
		leftList=gpuTriangleList.getList(leftPrimBaseIdx);
		rightList=gpuTriangleList.getList(rightPrimBaseIdx);
	}
	__syncthreads();

	float low = FLT_MIN;
	float high = FLT_MAX;
	
	//Need to initialize the offL, offD, offR arrays 
	while(currIdx<node->primLength)
	{
		offL[threadIdx.x]=0;
		offR[threadIdx.x]=0;
		offD[threadIdx.x]=0;

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

		if( low < splitValue && high < splitValue )
		{
			offL[currIdx] = 1;
			triangleChoice=0;
		}

		if( low >= splitValue && high >= splitValue) 
		{
			offR[currIdx] = 1;
			triangleChoice=1;
		}

		if( low < splitValue && hight >= splitValue ) 
		{
			offD[currIdx] = 1;
			triangleChoice=2;
		}

		__syncthreads();

		if(threadIdx.x==0)
		{
			for(uint32 k=1;k<blockDim;k++)
			{
				offL[k] += offL[k-1];
				offR[k] += offR[k-1];
				offD[k] += offD[k-1];
			}
			leftCount += offL[blockDim-1]+offD[blockDim-1];
			rightCount += offR[blockDim-1]+offD[blockDim-1];
		}

		__syncthreads();

		if(triangleChoice==0)
		{
			leftList[leftBase+offL[threadIdx.x]-1]=triangleID;
		}
		else if(triangleChoice==1)
		{
			rightList[rightBase+offR[threadIdx.x]-1]=triangleID;
		}
		else if(triangleChoice==2)
		{
			leftList[leftBase+offL[blockDim-1]+offD[threadIdx.x]-1]=triangleID;
			rightList[rightBase+offR[blockDim-1]+offD[threadIdx.x]-1]=triangleID;
		}
	
		leftBase += offL[blockDim-1]+offD[blockDim-1];
		rightBase += offR[blockDim-1]+offD[blockDim-1];

		currIdx += blockDim;
	}
	
	if(threadIdx.x==0)
	{
		d_gpuNodes.lock();

		leftNode =  d_gpuNodes.allocateNode();
		rightNode = d_gpuNodes.allocateNode();

		d_gpuNodes.unlock();

		node->leftIdx = leftNode->nodeIdx;
		node->rightIdx = rightNode->nodeIdx;

		leftNode->primBaseIdx=leftPrimBaseIdx;
		leftNode->primLength=leftCount;
		leftNode->nodeDepth=node->nodeDepth+1;
		
		rightNode->primBaseIdx=rightPrimBaseIddx;
		rightNode->primLength=rightPrimBaesIdx;
		rightNode->nodeDepth=node->nodeDepth+1;
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




