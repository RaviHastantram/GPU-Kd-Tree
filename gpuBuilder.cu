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
#define MIN_NODES 10

using namespace std;


Point * d_points=0;
Triangle * d_triangles=0;

uint32 * d_triangleCounts=0;
uint32 * d_nodeCounts=0;

uint32 * d_numActiveNodes=0;
uint32 * d_numActiveTriangles=0;
uint32 * d_numTotalNodes=0;

///////////////////////////
// 
// Tree Building
//
///////////////////////////

void initializeDeviceVariables()
{
	HANDLE_ERROR( cudaMalloc(&d_numActiveNodes, sizeof(uint32) ) );
	HANDLE_ERROR( cudaMalloc(&d_numTotalNodes, sizeof(uint32) ) );
	HANDLE_ERROR( cudaMalloc(&d_numActiveTriangles, sizeof(uint32) ) );
	HANDLE_ERROR( cudaMalloc(&d_numActiveNodes, sizeof(uint32) ) );

	HANDLE_ERROR( cudaMalloc(&d_triangleCounts, sizeof(uint32) * MAX_BLOCKS) );
	HANDLE_ERROR( cudaMalloc(&d_nodeCounts, sizeof(uint32) * MAX_BLOCKS) );
}

void initializeActiveNodeList(GPUNodeArray * gpuNodes, GPUTriangleArray  * triangleArray, Mesh * m)
{
	GPUNode h_node;
	h_node.nodeIdx=0;
	h_node.isLeaf=false;

	h_node.hostTriangles=new uint32[m->numTriangles];
	for(int i=0;i<m->numTriangles;i++)
	{
		h_node.hostTriangles[i]=i;
	}
	h_node.primBaseIdx=triangleArray->pushList(h_node.hostTriangles,m->numTriangles);
	delete [] h_node.hostTriangles;

	assert(h_node.primBaseIdx==0);

	h_node.primLength=m->numTriangles;

	h_node.nodeDepth=0;

	gpuNodes->pushNode(&h_node);
	printf("initializeActiveNodeList:nextAvailable:%d\n",gpuNodes->nextAvailable);
}

uint32 countActiveNodes(GPUNodeArray * d_nodeArray, uint32 * d_numActiveNodes, uint32 * d_nodeCounts)
{
	uint32 numActiveNodes;
	countActiveNodesKernel<<<1,1>>>(d_nodeArray,d_numActiveNodes, d_nodeCounts);
	cudaError_t err=cudaGetLastError();
	HANDLE_ERROR( err );
	HANDLE_ERROR( cudaMemcpy(&numActiveNodes,d_numActiveNodes,sizeof(uint32),cudaMemcpyDeviceToHost) );
	return numActiveNodes;
}

uint32 countActiveTriangles(GPUNodeArray * d_nodeArray, uint32 * d_numActiveTriangles, uint32 * d_triangleCounts)
{
	uint32 numActiveTriangles;
	countActiveTrianglesKernel<<<1,1>>>(d_nodeArray,d_numActiveTriangles, d_triangleCounts);
	HANDLE_ERROR( cudaMemcpy(&numActiveTriangles,d_numActiveTriangles,sizeof(uint32),cudaMemcpyDeviceToHost) );
	return numActiveTriangles;
}

__global__ void countActiveNodesKernel(GPUNodeArray * d_nodeArray, uint32 * d_numActiveNodes, uint32 * d_nodeCounts)
{
	GPUNode * node = &d_nodeArray->nodes[d_nodeArray->firstActive];
	while(!node->isActive && d_nodeArray->firstActive < d_nodeArray->nextAvailable)
	{
		d_nodeArray->firstActive++;
		node = &d_nodeArray->nodes[d_nodeArray->firstActive];
	}
	cuPrintf("firstActive=%d, nextAvailable=%d\n",d_nodeArray->firstActive,d_nodeArray->nextAvailable);
	uint32 numActive = d_nodeArray->nextAvailable-d_nodeArray->firstActive;
	*d_numActiveNodes= numActive;
}

__global__ void countActiveTrianglesKernel(GPUNodeArray * d_nodeArray, uint32 * d_numActiveTriangles, uint32 * d_triangleCounts)
{
	uint32 count=0;
	for(uint32 i=d_nodeArray->firstActive;i<d_nodeArray->nextAvailable;i++)
	{
		GPUNode * node = &d_nodeArray->nodes[i];
		count += node->primLength;
	}
	*d_numActiveTriangles=count;
}

uint32 getThreadsPerNode(int numActiveNodes,int numActiveTriangles)
{
	return 32;
}


__device__ void computeCost(GPUNodeArray * d_gpuNodes, GPUTriangleArray * d_gpuTriangleList, 
				uint32 * d_nodeCounts, uint32 * d_triangleCounts, Triangle * d_triangles, Point * d_points)
{
	__shared__ float mins[MAX_BLOCK_SIZE];
	__shared__ float maxs[MAX_BLOCK_SIZE];
		
	float min=FLT_MAX;
	float max=FLT_MIN;
	uint32 nodeIdx = blockIdx.x + d_gpuNodes->firstActive;
	if(nodeIdx > GPU_NODE_ARRAY_NUM_NODES)
	{
		cuPrintf("Out of bounds, nodeIdx=%d,blockIdx.x=%d,firstActive=%d\n",nodeIdx,blockIdx.x,d_gpuNodes->firstActive);
		return;
	}
	GPUNode * node = d_gpuNodes->getNode(nodeIdx);
	uint32 dim = node->nodeDepth % 3;

	if(threadIdx.x==0)
	{
		//cuPrintf("computeCost:nodeIdx=%d,nodeDepth=%d, primLength=%d, minSize=%d, firstActive=%d\n",
		//	nodeIdx,node->nodeDepth, node->primLength, MIN_NODES,d_gpuNodes->firstActive);
	}
	if(node->nodeDepth>=MAX_DEPTH   || node->primLength <= MIN_NODES)
	{
		cuPrintf("computeCost:making leaf, depth=%d, primLength=%d\n",node->nodeDepth,node->primLength);
		node->splitChoice=SPLIT_NONE;
		node->isLeaf=true;
		return;
	}
	
	uint32 * triangleIDs = d_gpuTriangleList->getList(node->primBaseIdx);

	mins[threadIdx.x]=FLT_MAX;
	maxs[threadIdx.x]=FLT_MIN;
	
	uint32 currIdx = threadIdx.x;
	__syncthreads();
	while(currIdx<node->primLength)
	{
		if(currIdx>GPU_TRIANGLE_ARRAY_SIZE)
		{
			cuPrintf("computeCost:currIdx out of bounds, currIdx=%d\n",currIdx);
		} 
		 else 
		{
			uint32 triangleID =  triangleIDs[currIdx];
			if(triangleID>947)
			{
				cuPrintf("computeCost:triangleID out of bounds, triangleID=%d, currIdx=%d,primBase=%d, primLength=%d\n",
					triangleID,currIdx,node->primBaseIdx,node->primLength);
			} 
			else
			{			
				Triangle * triangle = &d_triangles[triangleID];
				for(uint32 j=0;j<3;j++)
				{
					uint32 pointID = triangle->ids[j];
					if(pointID>452)
					{
						cuPrintf("computeCost:pointID out of bounds, pointID=%d\n",pointID);
					}
					else 
					{
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
				}
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

		//cuPrintf("min=%f,max=%f,splitValue=%f splitChoice=%d\n",min,max,node->splitValue,node->splitChoice);
	}
}

__device__ void splitNodes(GPUNodeArray * d_gpuNodes, GPUTriangleArray  * d_gpuTriangleList, uint32 * d_nodeCounts,
				uint32 * d_triangleCounts, Triangle * d_triangles, Point * d_points)
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
	uint32 nodeIdx = blockIdx.x + d_gpuNodes->firstActive;
	GPUNode * node = d_gpuNodes->getNode(nodeIdx);
	
	int dim = node->splitChoice;
	float splitValue = node->splitValue;
	uint32 currIdx = threadIdx.x;
	uint32 * triangleIDs = d_gpuTriangleList->getList(node->primBaseIdx);
	
	uint32 leftBase=0, rightBase=0;
	uint32 leftCount=0, rightCount=0;
	//uint32 leftOff=0,rightOff=0;
	//cuPrintf("splitNodes:here\n");
	//cuPrintf("splitNodes:dim=%d\n",node->splitChoice);
	
	if(threadIdx.x==0)
	{
		node->isActive=false;
		//cuPrintf("splitNodes:nodeDepth=%d, primLength=%d\n",
		//	node->nodeDepth, node->primLength);
	//	cuPrintf("splitNodes:setting node counts\n");
		d_nodeCounts[blockIdx.x]=0;
	//	cuPrintf("splitNodes:setting triangle counts\n");
		d_triangleCounts[blockIdx.x]=0;
	//	cuPrintf("splitNodes:should be ready for splitting now\n");
	}
	__syncthreads();
	//cuPrintf("splitNodes:isLeaf=%d\n",(int)node->isLeaf);
	if(node->isLeaf)
	{
		return;
	}
	
	if(threadIdx.x==0)
	{
		d_gpuTriangleList->lock();

		leftPrimBaseIdx=d_gpuTriangleList->allocateList(node->primLength);
		rightPrimBaseIdx=d_gpuTriangleList->allocateList(node->primLength);

		d_gpuTriangleList->unlock();

		leftList=d_gpuTriangleList->getList(leftPrimBaseIdx);
		rightList=d_gpuTriangleList->getList(rightPrimBaseIdx);
		
		//cuPrintf("splitNode:leftPrimBaseIdx=%d, rightPrimBaseIdx=%d, node->nodeIdx=%d\n",
		//	leftPrimBaseIdx, rightPrimBaseIdx, node->nodeIdx);
	}
	__syncthreads();
	//cuPrintf("tid=%d,leftPrimBaseIdx=%d,rightPrimBaseIdx=%d\n",threadIdx.x,leftPrimBaseIdx,rightPrimBaseIdx);
	float low = FLT_MAX;
	float high = FLT_MIN;
	
	int lowestIdx=0;
	// Use lowestIdx to ensure all threads stay in this while loop until every thread finishes
	while(lowestIdx<node->primLength)
	{
		offL[threadIdx.x]=0;
		offR[threadIdx.x]=0;
		offD[threadIdx.x]=0;

		__syncthreads();
				
		uint32 triangleID=triangleIDs[currIdx];
		Triangle * triangle=&d_triangles[triangleID];
				   		
		// only threads with a valid currIdx get to participate in the computation
		if(currIdx<node->primLength)
		{	

			if(triangleID>947)
			{
				cuPrintf("splitNodes:triangleID out of bounds, triangleID=%d, currIdx=%d\n",triangleID,currIdx);
			}
			for(uint32 j=0;j<3;j++) {
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
		
			//cuPrintf("splitValue:%f,low:%f,high:%f\n",splitValue,low,high);
			if( low <= splitValue && high <= splitValue )
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
		}
		
		// sync threads here
		__syncthreads();

		if(threadIdx.x==0)
		{
			for(int k=1;k<blockDim.x;k++)
			{		
				offL[k]   += offL[k-1];				
				offR[k]  += offR[k-1];
				offD[k]  += offD[k-1];
			}
			leftCount += offL[blockDim.x-1]+offD[blockDim.x-1];
			rightCount += offR[blockDim.x-1]+offD[blockDim.x-1];
			//cuPrintf("splitNodes:leftBase=%d, leftIncrement=%d\n",leftBase, offL[blockDim.x-1]+offD[blockDim.x-1]);
			//cuPrintf("splitNodes:rightBase=%d, rightIncrement=%d\n",rightBase, offR[blockDim.x-1]+offD[blockDim.x-1]);
		}
	  
		__syncthreads();
	
		// use an if statement to ensure only valid threads participate
		if(currIdx < node->primLength) 
		{
				if(triangleID>947)
				{
					cuPrintf("splitNodes:triangleID out of bounds, triangleID=%d, currIdx=%d,threadIdx.x=%d\n",
							triangleID,currIdx,threadIdx.x);
				}
				if(triangleChoice==0)
				{
					uint32 leftDestIndex = leftBase+offL[threadIdx.x]-1;	
					leftList[leftDestIndex]=triangleID;
					//cuPrintf("splitNodes:wrote %d to left node at %d\n",leftDestIndex,triangleID);
				}
				else if(triangleChoice==1)
				{
					uint32 rightDestIndex = rightBase+offR[threadIdx.x]-1;
					rightList[rightDestIndex]=triangleID;
					//cuPrintf("splitNodes:wrote %d to right node at %d\n",rightDestIndex,triangleID);
				}
				else if(triangleChoice==2)
				{	
					uint32 leftDestIndex = leftBase+offL[blockDim.x-1]+offD[threadIdx.x]-1;
					uint32 rightDestIndex = rightBase+offR[blockDim.x-1]+offD[threadIdx.x]-1;
					leftList[leftDestIndex]=triangleID;
					rightList[rightDestIndex]=triangleID;
					//cuPrintf("splitNodes:wrote %d to left and right nodes at %d and %d\n",
					//	triangleID,leftDestIndex,rightDestIndex);
				}
				
				leftBase += offL[blockDim.x-1]+offD[blockDim.x-1];
				rightBase += offR[blockDim.x-1]+offD[blockDim.x-1];
		}

		currIdx += blockDim.x;
		lowestIdx += blockDim.x;
		
		__syncthreads();
	}
	
	if(threadIdx.x==0)
	{
		d_gpuNodes->lock();

		GPUNode* leftNode =  d_gpuNodes->allocateNode();
		GPUNode* rightNode = d_gpuNodes->allocateNode();

		d_gpuNodes->unlock();
	      
		node->leftIdx = leftNode->nodeIdx;
		node->rightIdx = rightNode->nodeIdx;
		node->isActive=false;
		
		leftNode->primBaseIdx=leftPrimBaseIdx;
		leftNode->primLength=leftCount;
		leftNode->nodeDepth=node->nodeDepth+1;
		leftNode->isActive=true;
		leftNode->isLeaf=false;

		for(int i=0;i<leftCount;i++)
		{
			if(leftList[i] > 947)
			{
				cuPrintf("splitNodes: bad triangleID (%d) at index %d\n",leftList[i],i);
				break;
			}
		}

		rightNode->primBaseIdx=rightPrimBaseIdx;
		rightNode->primLength=rightCount;
		rightNode->nodeDepth=node->nodeDepth+1;
		rightNode->isActive=true;
		rightNode->isLeaf=false;
	      
		for(int i=0;i<rightCount;i++)
		{
			if(rightList[i] > 947)
			{
				cuPrintf("splitNodes: bad triangleID (%d) at index %d\n",rightList[i],i);
				break;
			}
		}

		d_nodeCounts[blockIdx.x]=1;
		d_triangleCounts[blockIdx.x]=leftCount+rightCount;
		//cuPrintf("splitNode:leftPrimBaseIdx:%d, leftCount:%d\n",leftPrimBaseIdx,leftCount);
		//cuPrintf("splitNode:rightPrimBaseIdx:%d, rightCount:%d\n",rightPrimBaseIdx,rightCount);
		//cuPrintf("splitNode:leftIdx=%d, rightIdx=%d\n",node->leftIdx, node->rightIdx);
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

struct SSE3Point {
	float reserved;
	float x;
	float y;
	float z;
};

struct SSEAABBOX {
	SSE3Point lo;
	SSE3Point hi;

};

void dumpKDTree(GPUNode * nodes, uint32 numNodes, uint32 numLeaves, BoundingBox bounds)
{
	ofstream file("GPU-Kd-tree",ios::out | ios::binary);

	char *buffer = new char[100];
	SSE3Point lo = {0,bounds.min[0],bounds.min[1],bounds.min[2]};
	SSE3Point hi = {0,bounds.max[0],bounds.max[1],bounds.max[2]};
	SSEAABBOX bbox = {lo,hi};
	
	unsigned int version = 0x0;
	 version = 0x02;

	//1. Write the LAYOUT_VERSION.
	file.write((char*)&version,sizeof(unsigned int));

	//2. Write the Bounds
	file.write((char*)&bbox,sizeof(SSEAABBOX));
	/*
	float zero=0;
	file.write((char*)&zero,sizeof(float));
	file.write((char*)&bounds.min,sizeof(float)*3);
	file.write((char*)&zero,sizeof(float));
	file.write((char*)&bounds.max,sizeof(float)*3);
	*/
	
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
		data0 |= (node->splitChoice + 1)%3;
		
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



