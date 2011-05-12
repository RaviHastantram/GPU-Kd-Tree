#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>

#include "kdtypes.h"
#include "geom.h"
#include "gpuBuilder.h"
#include "util.h"
#include "gpuTriangleList.h"
#include "gpuNode.h"

using namespace std;

int main(int argc, char  ** argv)
{
	char * inputFile = argv[1];

	// load ply
	Mesh * m = loadMeshFromPLY(inputFile);
	
	// copy to device
	copyToGPU(m);
	
	GPUTriangleArray *d_triangleArray = new GPUTriangleArray();
	GPUNodeArray *d_nodeArray = new GPUNodeArray();

	int numActiveNodes=1;
	int numActiveTriangles=m->numTriangles;
	int threadsPerNode = 0;
	uint32 activeOffset;

	thrust::device_vector<int> d_nodeCountsVec(MAX_BLOCKS);
	thrust::device_vector<int> d_triangleCountsVec(MAX_BLOCKS);
	int * d_nodeCounts = thrust::raw_pointer_cast(&d_nodeCountsVec[0]);
	int * d_triangleCounts = thrust::raw_pointer_cast(&d_triangleCountsVec[0]);
	int nodeCount=0;
	int triangleCount=0;
	int numTotalNodes=1;
	uint32 numLeaves=0;
	uint32 currRound=0;

	// initialize the node list
	initializeActiveNodeList(d_nodeArray,d_triangleArray,m);

	while(numActiveNodes>0)
	{
		printf("Current round:%d\n",currRound);
		currRound++;

		// copy offset to first active node to device
		cudaMemcpy(&d_activeOffset,&activeOffset,sizeof(uint32),cudaMemcpyHostToDevice);
		
		// calculate number of threads to assign to each node
		threadsPerNode = getThreadsPerNode(numActiveNodes,numActiveTriangles);
		
		// compute the split plane and value of each node
		computeCost <<< numActiveNodes,threadsPerNode >>>(d_nodeArray,d_triangleArray);

		// split each node according to the plane and value chosen
		splitNodes<<<numActiveNodes,threadsPerNode>>>(d_nodeArray,d_triangleArray,d_nodeCounts,d_triangleCounts);

		// force threads to synchronize globally
		cudaThreadSynchronize();
		
		// increment pointer to first active node
		cudaMemcpy(&activeOffset,&d_activeOffset,sizeof(uint32),cudaMemcpyDeviceToHost);
		activeOffset += numActiveNodes;

		// calculate number of active nodes in next round
		nodeCount=thrust::count(d_nodeCountsVec.begin(), d_nodeCountsVec.end() + numActiveNodes, 1);
		numActiveNodes=2*nodeCount;
		numTotalNodes += numActiveNodes;		
	
		// calculate number of triangles in next round
		triangleCount=thrust::reduce(d_triangleCountsVec.begin(),
					d_triangleCountsVec.end() +  numActiveNodes,
					 (int) 0, thrust::plus<int>());
		numActiveTriangles=triangleCount;		
	}

	// allocate host storage for nodes
	GPUNode * h_gpuNodes=new GPUNode[numTotalNodes];

	// copy out triangles out
	copyToHost(d_triangleArray, h_gpuNodes, &numLeaves, d_nodeArray->getNodes(), numTotalNodes);
	
	// copy triangles to disk
	dumpKDTree(h_gpuNodes, numTotalNodes, numLeaves,  m->bounds);
		
	return 0;
}
