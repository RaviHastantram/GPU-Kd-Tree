#include <iostream>
#include <fstream>
#include <cstdio>
#include <cfloat>
#include <cassert>

#include "kdtypes.h"
#include "gpuBuilder.h"
#include "util.h"
#include "geom.h"
#include "cuPrintf.cu"

#define MAX_ITERATIONS 100000

using namespace std;


Point * d_points=0;
Triangle * d_triangles=0;

uint32 * d_triangleCounts=0;
uint32 * d_nodeCounts=0;

uint32 * d_numActiveNodes=0;
uint32 * d_numActiveTriangles=0;
uint32 * d_activeOffset=0;
uint32 * d_numTotalNodes=0;

///////////////////////////
// 
// Tree Building
//
///////////////////////////

void initializeDeviceVariables()
{
	HANDLE_ERROR( cudaMalloc(&d_numActiveNodes, sizeof(uint32) ) );
	HANDLE_ERROR( cudaMalloc(&d_activeOffset, sizeof(uint32) ) );
	HANDLE_ERROR( cudaMalloc(&d_numTotalNodes, sizeof(uint32) ) );
	HANDLE_ERROR( cudaMalloc(&d_numActiveTriangles, sizeof(uint32) ) );
	HANDLE_ERROR( cudaMalloc(&d_numActiveNodes, sizeof(uint32) ) );

	HANDLE_ERROR( cudaMalloc(&d_triangleCounts, sizeof(uint32) * MAX_BLOCKS) );
	HANDLE_ERROR( cudaMalloc(&d_nodeCounts, sizeof(uint32) * MAX_BLOCKS) );
}

void initializeActiveNodeList(GPUNodeArray gpuNodes, GPUTriangleArray triangleArray, Mesh * m)
{
	GPUNode h_node;
	h_node.nodeIdx=0;
	h_node.isLeaf=false;

	h_node.hostTriangles=new uint32[m->numTriangles];
	for(int i=0;i<m->numTriangles;i++)
	{
		h_node.hostTriangles[i]=i;
	}
	h_node.primBaseIdx=triangleArray.pushList(h_node.hostTriangles,m->numTriangles);
	delete [] h_node.hostTriangles;

	assert(h_node.primBaseIdx==0);

	h_node.primLength=m->numTriangles;

	h_node.nodeDepth=0;

	gpuNodes.pushNode(&h_node);
}

uint32 countActiveNodes(uint32 numBlocks, uint32 * d_numActiveNodes, uint32 * d_nodeCounts)
{
	uint32 numActiveNodes;
	countActiveNodesKernel<<<1,1>>>(numBlocks,d_numActiveNodes, d_nodeCounts);
	cudaError_t err=cudaGetLastError();
	HANDLE_ERROR( err );
	HANDLE_ERROR( cudaMemcpy(&numActiveNodes,d_numActiveNodes,sizeof(uint32),cudaMemcpyDeviceToHost) );
	return numActiveNodes;
}

uint32 countActiveTriangles(uint32 numBlocks, uint32 * d_numActiveTriangles, uint32 * d_triangleCounts)
{
	uint32 numActiveTriangles;
	countActiveTrianglesKernel<<<1,1>>>(numBlocks,d_numActiveTriangles, d_triangleCounts);
	HANDLE_ERROR( cudaMemcpy(&numActiveTriangles,d_numActiveTriangles,sizeof(uint32),cudaMemcpyDeviceToHost) );
	return numActiveTriangles;
}

__global__ void countActiveNodesKernel(uint32 numBlocks, uint32 * d_numActiveNodes, uint32 * d_nodeCounts)
{
	uint32 count=0;
	for(int i=0;i<numBlocks;i++)
	{
		count += d_nodeCounts[i];
	}
	*d_numActiveNodes=count;
}

__global__ void countActiveTrianglesKernel(uint32 numBlocks, uint32 * d_numActiveTriangles, uint32 * d_triangleCounts)
{
	uint32 count=0;
	for(int i=0;i<numBlocks;i++)
	{
		count += d_triangleCounts[i];
	}
	*d_numActiveTriangles=count;
}

uint32 getThreadsPerNode(int numActiveNodes,int numActiveTriangles)
{
	return 32;
}


__device__ void computeCost(GPUNodeArray gpuNodes, GPUTriangleArray gpuTriangleList, 
				uint32 * d_nodeCounts, uint32 * d_triangleCounts, uint32 * d_activeOffset,
				Triangle * d_triangles, Point * d_points)
{
	__shared__ float mins[MAX_BLOCK_SIZE];
	__shared__ float maxs[MAX_BLOCK_SIZE];
		
	float min=FLT_MAX;
	float max=FLT_MIN;

	int checkid=3;

	if(threadIdx.x==checkid)
	{
		cuPrintf("here 1:Msg from kernel, tid = %d\n",threadIdx.x); 
	}
	uint32 dim = blockIdx.x % 3;
	uint32 nodeIdx = blockIdx.x + *d_activeOffset;

	GPUNode * node = gpuNodes.getNode(nodeIdx);
	if(threadIdx.x==checkid)
	{
		cuPrintf("here 2:Msg from kernel, tid = %d\n",threadIdx.x); 
	}
	if(node->nodeDepth>MAX_DEPTH)
	{
		node->splitChoice=SPLIT_NONE;
		node->isLeaf=true;
		return;
	}
	if(threadIdx.x==checkid)
	{
		cuPrintf("here 3:Msg from kernel, tid = %d\n",threadIdx.x); 
	}
	
	uint32 * triangleIDs = gpuTriangleList.getList(node->primBaseIdx);

	mins[threadIdx.x]=FLT_MAX;
	maxs[threadIdx.x]=FLT_MIN;

	uint32 currIdx = threadIdx.x;
	//cuPrintf("computeCost:tid=%d, primLength=%d\n",threadIdx.x,node->primLength);
	while(currIdx<node->primLength && currIdx < MAX_ITERATIONS)
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
	if(threadIdx.x==checkid)
	{
		cuPrintf("here 4:Msg from kernel, tid = %d\n",threadIdx.x); 
	}
	__syncthreads();

	if(threadIdx.x==checkid)
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
	if(threadIdx.x==checkid)
	{
		cuPrintf("here 5:Msg from kernel, tid = %d\n",threadIdx.x); 
		cuPrintf("splitValue=%f splitChoice=%d\n",node->splitValue,node->splitChoice);
	}
}

__device__ void splitNodes(GPUNodeArray gpuNodes, GPUTriangleArray  gpuTriangleList, uint32 * d_nodeCounts,
				uint32 * d_triangleCounts, uint32 * d_activeOffset,
				Triangle * d_triangles, Point * d_points)
{
	
	__shared__ uint32 offL[MAX_BLOCK_SIZE];
	__shared__ uint32 offD[MAX_BLOCK_SIZE];
	__shared__ uint32 offR[MAX_BLOCK_SIZE];
	
	__shared__ uint32 * leftList;
	__shared__ uint32 * rightList;
	__shared__ uint32 leftPrimBaseIdx;
	__shared__ uint32 rightPrimBaseIdx;
	uint32 triangleChoice;
	//cuPrintf("splitNodes:here0\n");
	uint32 nodeIdx = blockIdx.x + *d_activeOffset;
	GPUNode * node = gpuNodes.getNode(nodeIdx);
	
	int dim = node->splitChoice;
	float splitValue = node->splitValue;
	uint32 currIdx = threadIdx.x;
	uint32 * triangleIDs = gpuTriangleList.getList(node->primBaseIdx);
	
	uint32 leftBase=0, rightBase=0;
	uint32 leftCount=0, rightCount=0;
	//cuPrintf("splitNodes:here\n");
	//cuPrintf("splitNodes:dim=%d\n",node->splitChoice);
	if(threadIdx.x==0)
	{
	//	cuPrintf("splitNodes:setting node counts\n");
		d_nodeCounts[blockIdx.x]=0;
	//	cuPrintf("splitNodes:setting triangle counts\n");
		d_triangleCounts[blockIdx.x]=0;
	//	cuPrintf("splitNodes:should be ready for splitting now\n");
	}
	//cuPrintf("splitNodes:isLeaf=%d\n",(int)node->isLeaf);
	if(node->isLeaf)
	{
		return;
	}
	
	if(threadIdx.x==0)
	{
		gpuTriangleList.lock();

		leftPrimBaseIdx=gpuTriangleList.allocateList(node->primLength);
		rightPrimBaseIdx=gpuTriangleList.allocateList(node->primLength);

		gpuTriangleList.unlock();

		leftList=gpuTriangleList.getList(leftPrimBaseIdx);
		rightList=gpuTriangleList.getList(rightPrimBaseIdx);
	}
	__syncthreads();
	cuPrintf("tid=%d,leftPrimBaseIdx=%d,rightPrimBaseIdx=%d\n",threadIdx.x,leftPrimBaseIdx,rightPrimBaseIdx);
	float low = FLT_MAX;
	float high = FLT_MIN;
	
	//cuPrintf("splitNode:tid=%d, primLength=%d\n",threadIdx.x,node->primLength);
	//Need to initialize the offL, offD, offR arrays 
	int lim=node->primLength;
	
	while(currIdx<node->primLength && currIdx < MAX_ITERATIONS)
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
			offL[threadIdx.x] = 1;
			triangleChoice=0;
		}
	        else
		if( low >= splitValue && high >= splitValue) 
		{
			offR[threadIdx.x] = 1;
			triangleChoice=1;
		}
		else 
		{
			offD[threadIdx.x] = 1;
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
  
		uint32 _offL=offL[threadIdx.x]-1;
		uint32 _offR = offR[threadIdx.x]-1;
		uint32 tc0=leftBase+_offL;
		uint32 tc1=rightBase+_offR;
		uint32 tc2a=tc0+offD[threadIdx.x];
		uint32 tc2b=tc1+offD[threadIdx.x];
		cuPrintf("triangleChoice=%d,tc0=%d, tc1=%d, tc2a=%d, tc2b=%d\n",triangleChoice,tc0,tc1,tc2a,tc2b);
		return;
		/**
		if(triangleChoice==0)
		{
			leftList[tc0]=triangleID;
		}
		else if(triangleChoice==1)
		{
			rightList[tc1]=triangleID;
		}
		else if(triangleChoice==2)
		{
			leftList[tc2a]=triangleID;
			rightList[tc2b]=triangleID;
		}**/
		
		leftBase += offL[blockDim.x-1]+offD[blockDim.x-1];
		rightBase += offR[blockDim.x-1]+offD[blockDim.x-1];

		currIdx += blockDim.x;
	}
	return;
	if(threadIdx.x==0)
	{
		gpuNodes.lock();

		GPUNode* leftNode =  gpuNodes.allocateNode();
		GPUNode* rightNode = gpuNodes.allocateNode();

		gpuNodes.unlock();
	      
		node->leftIdx = leftNode->nodeIdx;
		node->rightIdx = rightNode->nodeIdx;
		
	
		leftNode->primBaseIdx=leftPrimBaseIdx;
		leftNode->primLength=leftCount;
		leftNode->nodeDepth=node->nodeDepth+1;
		
		rightNode->primBaseIdx=rightPrimBaseIdx;
		rightNode->primLength=rightCount;
		rightNode->nodeDepth=node->nodeDepth+1;
	      
		d_nodeCounts[blockIdx.x]=1;
		d_triangleCounts[blockIdx.x]=leftCount+rightCount;
	}
	cuPrintf("splitNode:leftIdx=%d, rightIdx=%d\n",node->leftIdx, node->rightIdx);
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
	HANDLE_ERROR( cudaMalloc(&d_points,size) );
	HANDLE_ERROR( cudaMemcpy(d_points,mesh->points,size,cudaMemcpyHostToDevice) );

	//Copy the triangle list
	size = sizeof(Triangle)*(mesh->numTriangles);
	HANDLE_ERROR( cudaMalloc(&d_triangles, size) );
	HANDLE_ERROR( cudaMemcpy(d_triangles,mesh->triangles,size,cudaMemcpyHostToDevice) );
}

void copyToHost(GPUTriangleArray gpuTriangleArray, GPUNode * h_gpuNodes, uint32 * h_numLeaves, GPUNode * d_gpuNodes, uint32 numNodes)
{
	// copy the nodes
	HANDLE_ERROR( cudaMemcpy(h_gpuNodes,d_gpuNodes,sizeof(GPUNode)*numNodes,cudaMemcpyDeviceToHost) );
	for(int i=0;i<numNodes && i< MAX_ITERATIONS;i++)
	{
		GPUNode * node = &h_gpuNodes[i];
		if(node->isLeaf)
		{
			node->hostTriangles = new uint32[node->primLength];
			gpuTriangleArray.copyList(node->hostTriangles, node->primBaseIdx, node->primLength);
			*h_numLeaves++;
		}
	}
}

void dumpKDTree(GPUNode * nodes, uint32 numNodes, uint32 numLeaves, BoundingBox bounds)
{
	ofstream file("GPU-Kd-tree",ios::out | ios::binary);

	char *buffer = new char[100];
	
	unsigned int version = 1;

	//1. Write the LAYOUT_VERSION.
	file.write((char*)&version,sizeof(unsigned int));

	//2. Write the Bounds
	float zero=0;
	file.write((char*)&zero,sizeof(float));
	file.write((char*)&bounds.min,sizeof(float)*3);
	file.write((char*)&zero,sizeof(float));
	file.write((char*)&bounds.max,sizeof(float)*3);
	
	
	//3. Write the number of nodes.
	uint64_t n = (uint64_t)numNodes;
	file.write((char*)&n,sizeof(uint64_t));

	//4. Write the nodes.
	for(int i = 0; i < numNodes; i++)
	{
		dumpNode(file,i,nodes);		
	}

	//5.Write the number of leaves
	
	uint64_t leafCount = (uint64_t)numLeaves;
	file.write((char*)&leafCount,sizeof(uint64_t));

	//6. Write the triangles
	for(int i = 0; i < numNodes; i++)
	{
		if(nodes[i].isLeaf)
		{	
			dumpTriangles(file,i,nodes);
		}
	}

	file.close();
}


void dumpNode(ofstream& file,uint32 nodeID, GPUNode* nodes)
{
	GPUNode* node = &nodes[nodeID];
	uint32 data0 = 0;
	float data1 = 0; 
	if(node->isLeaf)
	{
		file.write((char*)&data0, sizeof(uint32));
		file.write((char*)&nodeID, sizeof(uint32));
	}
	else
	{
		data0 |= node->leftIdx;
		data0 <<= 2;
		data0 |= node->splitChoice;
		
		data1 = node->splitValue;
		
		file.write((char*)&data0, sizeof(uint32));
		file.write((char*)&data1, sizeof(float));
	}
}

void dumpTriangles(ofstream& file, uint32 nodeID, GPUNode* nodes)
{
	GPUNode* node = &nodes[nodeID];
	//7. Write the length of the triangle list
	uint64_t numTriangles = node->primLength;
	file.write((char*)&numTriangles,sizeof(uint64_t));

	//8. Write the triangles
	uint32 triangleIndex = 0; //index of the triangle in the PLY file
	for(int i = 0; i < numTriangles && i < MAX_ITERATIONS; i++)
	{
		triangleIndex = node->hostTriangles[i];
		file.write((char*)&triangleIndex,sizeof(triangleIndex));
	}
}



