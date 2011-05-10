#ifndef __LISTS_H__
#define __LISTS_H__

#include <cuda.h>
#include "kdtree.h"
struct NodeList {
	int size;
	int capacity;
	Node ** nodes;
};

struct TriangleList {
	int size;
	int capacity;
	Triangle ** triangles;
};

__device__ NodeList * activeNodes=0;
__device__ NodeList * pendingNodes=0;
__device__ NodeList * completedNodes=0;
__device__ TriangleList * activeTriangles=0;

////////////////////////
//  NodeList functions
///////////////////////

__device__ void addNode(Node * node, NodeID nodeID, NodeList * list);
__device__ void initList(NodeList * list);
__device__ void clearList(NodeList * list);
__device__ void swapLists(NodeList ** list1, NodeList ** list2);
__device__ Node * getNodeAt(uint32 idx, NodeList * list);
__device__ void initializeTreeBuild();

////////////////////////
//  activeTriangles functions
///////////////////////

__device__ void initActiveTriangles(uint32 numBlocks);
__device__ void addTriangle(Triangle * t, int listIndex);
__device__ Triangle * getTriangle(int listIndex, int triangleIndex);
__device__ uint32 getNumTriangles(int listIndex);

#endif
