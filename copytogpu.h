#include "Mesh.h"
#include "kdtree.h"
#include "file.h"

void CopytoGPU(Mesh *mesh);

__global__ void CopytoHost(KDTree *kdtree);
__global__ void CopyNode(KDTree *kdtree,NodeID nodeID, Node* nodes);	
__global__ void BuildKDTree();
__device__ void computecost();
__device__ void split();

void DumpKDTree(Nodes *nodes, int numNodes);
void DumpNode(NodeID nodeID);
