#include <iostream>

#include "kdtypes.h"
#include "geom.h"
#include "gpuBuilder.h"
#include "util.h"
#include "gpuTriangleList.h"
#include "gpuNode.h"
#include "cuPrintf.cuh"

#define MAX_ITERATIONS 1000

using namespace std;

extern Point * d_points;
extern Triangle * d_triangles;
extern uint32 * d_triangleCounts;
extern uint32 * d_nodeCounts;
extern uint32 * d_numActiveNodes;
extern uint32 * d_numActiveTriangles;
extern uint32 * d_numTotalNodes;

int main(int argc, char  ** argv)
{
	uint32  currIteration=0;

	char * inputFile = argv[1];
	cudaPrintfInit();
	// load ply
	Mesh * m = loadMeshFromPLY(inputFile);
	
	// copy to device
	copyToGPU(m);
	
	GPUTriangleArray triangleArray=GPUTriangleArray();
	GPUNodeArray nodeArray=GPUNodeArray();
	
	int numActiveNodes=1;
	int numActiveTriangles=m->numTriangles;
	int threadsPerNode = 0;
	int numTotalNodes=1;
	uint32 numLeaves=0;
	uint32 currRound=0;

	// initialize device variables
	printf("initializeDeviceVariables\n");
	initializeDeviceVariables();

	
	// initialize the node list
	printf("initializeActiveNodeList\n");
	initializeActiveNodeList(&nodeArray,&triangleArray,m);

	GPUNodeArray * d_nodeArray;
	GPUTriangleArray * d_triangleArray;

	HANDLE_ERROR(cudaMalloc(&d_nodeArray,sizeof(GPUNodeArray)));
	HANDLE_ERROR(cudaMalloc(&d_triangleArray,sizeof(GPUTriangleArray)));
	HANDLE_ERROR(cudaMemcpy(d_nodeArray,&nodeArray,sizeof(GPUNodeArray),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_triangleArray,&triangleArray,sizeof(GPUTriangleArray),cudaMemcpyHostToDevice));

	CHECK_ERROR();

	while(currIteration <MAX_DEPTH && numActiveNodes>0)
	{
		currIteration++;

		printf("Current round:%d\n",currRound);
		currRound++;
		
		// calculate number of threads to assign to each node
		threadsPerNode = getThreadsPerNode(numActiveNodes,numActiveTriangles);
		
		printf("calling computeCost(%d,%d)\n",numActiveNodes,threadsPerNode);
		CHECK_ERROR();
		// compute the split plane and value of each node
		computeCost <<< numActiveNodes,threadsPerNode >>>(d_nodeArray, d_triangleArray, d_nodeCounts, 
								d_triangleCounts, d_triangles, d_points);
	
		
		HANDLE_ERROR(cudaThreadSynchronize());
		
		
		//CHECK_ERROR();

		printf("calling splitNodes (%d,%d)\n",numActiveNodes,threadsPerNode);
		// split each node according to the plane and value chosen
		splitNodes<<<numActiveNodes,threadsPerNode>>>(d_nodeArray, d_triangleArray, d_nodeCounts, 
								d_triangleCounts, d_triangles, d_points);
		CHECK_ERROR();
		
		printf("cudaThreadSynchronize\n");
		// force threads to synchronize globally
		HANDLE_ERROR(cudaThreadSynchronize());

		// status check
		//CHECK_ERROR();
		//cudaPrintfDisplay(stdout,true);
		//CHECK_ERROR();

		printf("Count active nodes\n");
		// calculate number of active nodes in next round and find first active node
		numActiveNodes=countActiveNodes(d_nodeArray,d_numActiveNodes, d_nodeCounts);
		HANDLE_ERROR(cudaThreadSynchronize());

		HANDLE_ERROR(cudaMemcpy(&nodeArray,d_nodeArray,sizeof(GPUNodeArray),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(&triangleArray,d_triangleArray,sizeof(GPUTriangleArray),cudaMemcpyDeviceToHost));
		printf("nodeArray:numActiveNodes=%d, nextAvailable=%d, firstActive=%d\n",
			numActiveNodes,nodeArray.nextAvailable,nodeArray.firstActive);
		printf("triangleArray:capacity=%d,nextAvailable=%d\n",triangleArray.capacity,triangleArray.nextAvailable);

		// update total nodes
		numTotalNodes += numActiveNodes;		
	
		printf("Count active triangles\n");
		// calculate number of triangles in next round
		numActiveTriangles=countActiveTriangles(d_nodeArray,d_numActiveTriangles, d_triangleCounts);			
	}
	
	// allocate host storage for nodes
	GPUNode * h_gpuNodes=new GPUNode[numTotalNodes];

	// copy out triangles out
	copyToHost(triangleArray, h_gpuNodes, &numLeaves, nodeArray.getNodes(), numTotalNodes);
	
	// copy triangles to disk
	dumpKDTree(h_gpuNodes, numTotalNodes, numLeaves,  m->bounds);
	cudaPrintfEnd();		

	triangleArray.destroy();
	nodeArray.destroy();

	return 0;
}
