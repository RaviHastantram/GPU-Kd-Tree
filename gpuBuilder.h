#include "Mesh.h"
#include "kdtree.h"
#include "file.h"


typedef unsigned __int64 uint64_t;

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

uint32 numActiveNodes();
uint32 numActiveTriangles();

////////////////////
//  Data Import/Export
////////////////////
__global__ void copyToHost(KDTree *kdtree);
__global__ void copyNode(KDTree *kdtree,NodeID nodeID, Node* nodes);

void copyToGPU(Mesh *mesh);
void dumpKDTree(KDTree *kdtree);
void dumpNode(ofstream& file,NodeID nodeID,KDTree *kdtree);
void dumpTriangles(ofstream& file, NodeID nodeID, KDTree *kdtree);

