//#include "Mesh.h"
#include "gpuNode.h"
#include <sys/types.h>
#include <cuda.h>
#include <cassert>
#include <cstdio>
#include <iostream>
#include "gpuNode.h"
#include "gpuTriangleList.h"
#include "geom.h"
using namespace std;
//#include "file.h"

#define MAX_BLOCK_SIZE 128
#define MAX_BLOCKS 32000

typedef u_int64_t uint64_t;

__device__ Point* d_points;
__device__ Triangle* d_triangles;
__device__ Mesh* d_mesh;
__device__ uint32 d_numActiveTriangles;

//////
//
//	d_numActiveNodes - number of current active nodes
//	d_activeOffset - offset to the first active node (all nodes before this are completed) 
//
/////
//__device__ GPUNodeArray d_gpuNodes;
__device__ uint32 d_numActiveNodes=0;
__device__ uint32 d_activeOffset=0;
__device__ uint32 d_numTotalNodes=0;


////////////////////
// Tree Building
////////////////////
__host__   void initializeActiveNodeList(GPUNodeArray* d_gpuNodes,  GPUTriangleArray *d_triangleArray, Mesh * m);
__global__ void computeCost(GPUNodeArray* d_gpuNodes, GPUTriangleArray* gpuTriangleList);
__global__ void splitNodes(GPUNodeArray* d_gpuNodes, GPUTriangleArray* gpuTriangleList, int * nodeCounts, int * triangleCounts);

uint32 getThreadsPerNode(int,int);

////////////////////
//  Data Import/Export
////////////////////
void copyToGPU(Mesh *mesh);
void copyToHost(GPUTriangleArray * d_gpuTriangleArray, GPUNode * h_nodeList, uint32 * h_numLeaves, GPUNode * d_nodes, uint32 numNodes);
void dumpKDTree(GPUNode * nodes, uint32 numNodes, uint32 numLeaves, BoundingBox bounds);
void dumpNode(ofstream& file,uint32 nodeID, GPUNode* nodes);
void dumpTriangles(ofstream& file, uint32 nodeID,GPUNode* nodes);


