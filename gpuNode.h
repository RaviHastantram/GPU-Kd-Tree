#ifndef __GPU_NODE_H__
#define __GPU_NODE_H__

#include <cuda.h>
#include "util.h"
#include "lock.h"

#define SPLIT_X 0
#define SPLIT_Y 1
#define SPLIT_Z 2
#define SPLIT_NONE 3
#define MAX_DEPTH 10

// a lot of nodes
#define GPU_NODE_ARRAY_NUM_NODES 5000000

// 48 bytes - aligned on 8 byte boundary
struct GPUNode {

	uint32 nodeIdx; // 4

	bool isLeaf; // 1 
	bool isActive; // 1

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

	bool padding[6];
};


struct GPUNodeArray
{
		GPUNodeArray() {
			HANDLE_ERROR(cudaMalloc(&nodes,GPU_NODE_ARRAY_NUM_NODES*sizeof(GPUNode)));
			capacity=GPU_NODE_ARRAY_NUM_NODES;
			nextAvailable=0;
			firstActive=0;
			l=Lock();
		}
	
		void destroy() {
			HANDLE_ERROR(cudaFree(nodes));
			l.destroy();
		}

		__device__ GPUNode * getNode(uint32 nodeIdx) {
			return &nodes[nodeIdx];
		}
			

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

		// NOTE: Only works for initialization
		__host__ void pushNode(GPUNode * h_node)
		{
			printf("copy:h_node->primLength=%d, nextAvailable=%d,firstActive=%d\n",h_node->primLength,nextAvailable,firstActive);
			GPUNode check;
			HANDLE_ERROR(cudaMemcpy(nodes,h_node,sizeof(GPUNode),cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(&check,nodes,sizeof(GPUNode),cudaMemcpyDeviceToHost));
			printf("check:check.primLength=%d\n",check.primLength);
			nextAvailable=1;
		}

		Lock l;
		uint32 capacity;
		uint32 nextAvailable;
		uint32 firstActive;
		GPUNode *nodes;
};


#endif
