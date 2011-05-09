//#include "Mesh.h"
#include "kdtree.h"
#include <linux/types.h>
#include <cuda.h>
#include <cassert>
#include <cstdio>
#include <iostream>

using namespace std;
//#include "file.h"


typedef __u64 uint64_t;

__device__ bool treeBuildInitialized=false;
__device__ Point* d_points;
__device__ Triangle* d_triangles;
__device__ Mesh* d_mesh;
__device__ uint32 d_numActiveTriangles;
__device__ uint32 d_numActiveNodes;

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
void copyToHost(KDTree *kdtree);
void copyNode(KDTree *kdtree,NodeID nodeID, Node* nodes);

void copyToGPU(Mesh *mesh);
void dumpKDTree(KDTree *kdtree);
void dumpNode(ofstream& file,NodeID nodeID,KDTree *kdtree);
void dumpTriangles(ofstream& file, NodeID nodeID, KDTree *kdtree);

