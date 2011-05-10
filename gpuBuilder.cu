#include "kdtypes.h"
#include "gpuBuilder.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cfloat>

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

__device__ void computeCost(GPUNodeArray* d_gpuNodes, GPUTriangleArray* gpuTriangleList)
{
	__shared__ float mins[MAX_BLOCK_SIZE];
	__shared__ float maxs[MAX_BLOCK_SIZE];
	
	float min=FLT_MAX;
	float max=FLT_MIN;
 
	uint32 dim = blockIdx.x % 3;
	uint32 nodeIdx = blockIdx.x + d_activeOffset;
	GPUNode * node = d_gpuNodes->getNode(nodeIdx);

	if(node->nodeDepth>MAX_DEPTH)
	{
		node->splitChoice=SPLIT_NONE;
		node->isLeaf=true;
		return;
	}
	
	uint32 * triangleIDs = gpuTriangleList->getList(node->primBaseIdx);

	mins[threadIdx.x]=FLT_MAX;
	maxs[threadIdx.x]=FLT_MIN;

	uint32 currIdx = threadIdx.x;

	while(currIdx<node->primLength)
	{
		uint32 triangleID =  triangleIDs[currIdx];
		Triangle * triangle = &d_triangles[triangleID];
		for(uint32 j=0;j<3;j++)
		{
			uint32 pointID = triangle->ids[j];
			Point * point = &d_points[pointID];
			if(point->values[dim]<mins[threadIdx.x])
			{
				mins[threadIdx.x]=point->values[dim];
			}
			if(point->values[dim]>maxs[threadIdx.x])
			{
				maxs[threadIdx.x]=point->values[dim];
			}
		}
		currIdx += blockDim.x;
	}

	__syncthreads();

	if(threadIdx.x==0)
	{
		for(uint32 k=0;k<blockDim.x;k++)
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

__device__ void splitNodes(GPUNodeArray* d_gpuNodes, GPUTriangleArray* gpuTriangleList, int * counts)
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
	GPUNode * node = d_gpuNodes->getNode(nodeIdx);
	int dim = node->splitChoice;
	float splitValue = node->splitValue;
	uint32 currIdx = threadIdx.x;
	uint32 * triangleIDs = gpuTriangleList->getList(node->primBaseIdx);
	uint32 leftBase=0, rightBase=0;
	uint32 leftCount=0, rightCount=0;

	if(threadIdx.x==0)
	{
		counts[blockIdx.x]=0;
	}
	
	if(node->isLeaf)
	{
		return;
	}

	if(threadIdx.x==0)
	{
		leftPrimBaseIdx=gpuTriangleList->allocateList(node->primLength);
		rightPrimBaseIdx=gpuTriangleList->allocateList(node->primLength);
		leftList=gpuTriangleList->getList(leftPrimBaseIdx);
		rightList=gpuTriangleList->getList(rightPrimBaseIdx);
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
		Triangle * triangle = &d_triangles[triangleID];

		for(uint32 j=0;j<3;j++)
		{
			uint32 pointID = triangle->ids[j];
			Point * point = &d_points[pointID];
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

		if( low < splitValue && high >= splitValue ) 
		{
			offD[currIdx] = 1;
			triangleChoice=2;
		}

		__syncthreads();

		if(threadIdx.x==0)
		{
			for(uint32 k=1;k<blockDim.x;k++)
			{
				offL[k] += offL[k-1];
				offR[k] += offR[k-1];
				offD[k] += offD[k-1];
			}
			leftCount += offL[blockDim.x-1]+offD[blockDim.x-1];
			rightCount += offR[blockDim.x-1]+offD[blockDim.x-1];
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
			leftList[leftBase+offL[blockDim.x-1]+offD[threadIdx.x]-1]=triangleID;
			rightList[rightBase+offR[blockDim.x-1]+offD[threadIdx.x]-1]=triangleID;
		}
	
		leftBase += offL[blockDim.x-1]+offD[blockDim.x-1];
		rightBase += offR[blockDim.x-1]+offD[blockDim.x-1];

		currIdx += blockDim.x;
	}
	
	if(threadIdx.x==0)
	{
		d_gpuNodes->lock();

		GPUNode* leftNode =  d_gpuNodes->allocateNode();
		GPUNode* rightNode = d_gpuNodes->allocateNode();

		d_gpuNodes->unlock();

		node->leftIdx = leftNode->nodeIdx;
		node->rightIdx = rightNode->nodeIdx;

		leftNode->primBaseIdx=leftPrimBaseIdx;
		leftNode->primLength=leftCount;
		leftNode->nodeDepth=node->nodeDepth+1;
		
		rightNode->primBaseIdx=rightPrimBaseIdx;
		rightNode->primLength=rightCount;
		rightNode->nodeDepth=node->nodeDepth+1;
	
		counts[blockIdx.x]=1;
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




