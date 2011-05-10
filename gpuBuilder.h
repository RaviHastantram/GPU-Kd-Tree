//#include "Mesh.h"
#include "gpuNode.h"
#include <sys/types.h>
#include <cuda.h>
#include <cassert>
#include <cstdio>
#include <iostream>

using namespace std;
//#include "file.h"


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
__device__ GPUNodeArray d_gpuNodes;
__device__ uint32 d_numActiveNodes=0;
__device__ uint32 d_activeOffset=0;


////////////////////
// Tree Building
////////////////////
__global__ void computeCost();
__global__ void splitNodes();

uint32 getActiveNodes();
uint32 getActiveTriangles();
uint32 getThreadsPerNode(int,int);

////////////////////
//  Data Import/Export
////////////////////
void copyToGPU(Mesh *mesh);


