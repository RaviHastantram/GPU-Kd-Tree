#ifndef __GPU_NODE_H__
#define __GPU_NODE_H__

#include <cuda.h>
#include "lock.h"

#define SPLIT_X 0
#define SPLIT_Y 1
#define SPLIT_Z 2
#define SPLIT_NONE 3
#define MAX_DEPTH 10

// 20 MB
#define GPU_NODE_ARRAY_NUM_NODES 524288
#define GPU_NODE_SIZE 48
#define GPU_NODE_ARRAY_SIZE GPU_NODE_ARRAY_NUM_NODES*GPU_NODE_SIZE

// 48 bytes - aligned on 8 byte boundary
struct GPUNode {

	uint32 nodeIdx; // 4

	bool isLeaf; // 1 

	// If internal, ID of left and right children
	uint32 leftIdx; // 4
	uint32 rightIdx; // 4

	// If active, this is the current node's working set
	// If a leaf, this is the triangles permanent triangle list
	uint32 primBaseIdx; // 4
	uint32 primLength; // 4

	// used for copying leaf node triangle list of ids to host side
	uint32 * hostTriangles; // 8

	// If internal, this is the split value
	float splitValue; // 4

	// 0 - x plane
	// 1 - y plane
	// 2 - z plane
	// 3 - no split
	uint32 splitChoice;  //4
	
	uint32 nodeDepth; // 4

	bool padding[7];
};


class GPUNodeArray
{
	public:
		__host__ GPUNodeArray() {
			capacity=GPU_NODE_ARRAY_SIZE;
			nextAvailable=0;
			cudaMalloc(&nodes,GPU_NODE_ARRAY_SIZE);
		}
	
		__device__ GPUNode * getNode(uint32 nodeIdx) {
			return &nodes[nodeIdx];
		}
			
		/*
		__device__ void putNode(GPUNode * node, uint32 nodeIx) {
			cudaMemcpy(&node[nodeIdx],node,sizeof(node),cudaMemcpyDeviceToDevice);
		}
		*/
		__device__ GPUNode * allocateNode() {
			if(capacity==nextAvailable)
			{
				return NULL;
			}

			GPUNode * node = &nodes[nextAvailable];
			node->nodeIdx=nextAvailable;
			nextAvailable++;
			return node;
		}
	
		__device__ void lock()
		{
			l.lock();
		}
		
		__device__ void unlock()
		{
			l.unlock();
		}

		__host__ GPUNode * getNodes() { return nodes; }

		__host__ void pushNode(GPUNode * node)
		{
			cudaMemcpy(&nodes[nextAvailable],node,sizeof(GPUNode),cudaMemcpyHostToDevice);
			nextAvailable++;
		}

	private:
		Lock l;
		uint32 capacity;
		uint32 nextAvailable;
		GPUNode *nodes;
};


#endif
