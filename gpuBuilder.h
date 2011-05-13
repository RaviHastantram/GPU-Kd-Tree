#ifndef __GPU_BUILDER_H__
#define __GPU_BUILDER_H__

//#include "Mesh.h"
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

/////////////////////
// Initialization
/////////////////////
void initializeDeviceVariables();
void initializeActiveNodeList(GPUNodeArray gpuNodes, GPUTriangleArray triangleArray, Mesh * m);

////////////////////
// Tree Building
////////////////////
__global__ void computeCost(GPUNodeArray gpuNodes, GPUTriangleArray  gpuTriangleList, uint32 * d_nodeCounts,
				 uint32 * d_triangleCounts, uint32 * d_activeOffset, Triangle * d_triangles, Point * d_points);
__global__ void splitNodes(GPUNodeArray gpuNodes, GPUTriangleArray gpuTriangleList, uint32 * d_nodeCounts,
				 uint32 * d_triangleCounts, uint32 * d_activeOffset, Triangle * d_triangles, Point * d_points);

uint32 getThreadsPerNode(int,int);

__global__ void countActiveNodesKernel(uint32 numBlocks, uint32 * d_numActiveNodes, uint32 * d_nodeCounts);
__global__ void countActiveTrianglesKernel(uint32 numBlocks, uint32 * d_numActiveTriangles, uint32 * d_triangleCounts);
uint32 countActiveNodes(uint32 numBlocks, uint32 * d_numActiveNodes, uint32 * d_nodeCounts);
uint32 countActiveTriangles(uint32 numBlocks, uint32 * d_numActiveTriangles, uint32 * d_triangleCounts);


////////////////////
//  Data Import/Export
////////////////////

void copyToGPU(Mesh *mesh);
void copyToHost(GPUTriangleArray gpuTriangleArray, GPUNode * h_nodeList, uint32 * h_numLeaves, GPUNode * d_nodes, uint32 numNodes);
void dumpKDTree(GPUNode * nodes, uint32 numNodes, uint32 numLeaves, BoundingBox bounds);
void dumpNode(ofstream& file,uint32 nodeID, GPUNode* nodes);
void dumpTriangles(ofstream& file, uint32 nodeID,GPUNode* nodes);

#endif


