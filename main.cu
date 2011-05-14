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
extern uint32 * d_activeOffset;
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
	uint32 activeOffset;
	int numTotalNodes=1;
	uint32 numLeaves=0;
	uint32 currRound=0;

	// initialize device variables
	printf("initializeDeviceVariables\n");
	initializeDeviceVariables();

	
	// initialize the node list
	printf("initializeActiveNodeList\n");
	initializeActiveNodeList(nodeArray,triangleArray,m);

	GPUTriangleArray * d_nodeArray;
	GPUTriangleArray * d_triangleArray;

	HANDLE_ERROR(cudaMemcpy(&d_nodeArray,&nodeArray,sizeof(GPUNodeArray)));
	HANDLE_ERROR(cudaMemcpy(&d_triangleArray,&triangleArray,sizeof(GPUTriangleArray)));

	CHECK_ERROR();

	while(currIteration <3 && numActiveNodes>0)
	{
		currIteration++;

		printf("Current round:%d\n",currRound);
		currRound++;

		printf("calling cudaMemcpy\n");
		// copy offset to first active node to device
		HANDLE_ERROR( cudaMemcpy(d_activeOffset,&activeOffset,sizeof(uint32),cudaMemcpyHostToDevice) );
		
		// calculate number of threads to assign to each node
		threadsPerNode = getThreadsPerNode(numActiveNodes,numActiveTriangles);
		
		printf("calling computeCost(%d,%d)\n",numActiveNodes,threadsPerNode);
		CHECK_ERROR();
		// compute the split plane and value of each node
		computeCost <<< numActiveNodes,threadsPerNode >>>(d_nodeArray, d_triangleArray, d_nodeCounts, 
								d_triangleCounts, d_activeOffset, 
								d_triangles, d_points);
		//CHECK_ERROR();
		//cudaPrintfDisplay(stdout,true);
		//CHECK_ERROR();
		
		HANDLE_ERROR(cudaThreadSynchronize());
		
		
		//CHECK_ERROR();

		printf("calling splitNodes (%d,%d)\n",numActiveNodes,threadsPerNode);
		// split each node according to the plane and value chosen
		splitNodes<<<numActiveNodes,threadsPerNode>>>(nodeArray, triangleArray, d_nodeCounts, 
								d_triangleCounts, d_activeOffset,
								d_triangles, d_points);
		CHECK_ERROR();
		
		printf("cudaThreadSynchronize\n");
		// force threads to synchronize globally
		HANDLE_ERROR(cudaThreadSynchronize());
		
		printf("Update activeOffset\n");
		// increment pointer to first active node
		HANDLE_ERROR(cudaMemcpy(&activeOffset,d_activeOffset,sizeof(uint32),cudaMemcpyDeviceToHost));
		activeOffset += numActiveNodes;
	
		printf("Count active nodes\n");
		// calculate number of active nodes in next round		
		numActiveNodes=countActiveNodes(numActiveNodes,d_numActiveNodes, d_nodeCounts);
		printf("numActiveNodes=%d\n",numActiveNodes);

		// update total nodes
		numTotalNodes += numActiveNodes;		
	
		printf("Count active triangles\n");
		// calculate number of triangles in next round
		numActiveTriangles=countActiveTriangles(numActiveNodes,d_numActiveTriangles, d_triangleCounts);		
	}
	cudaPrintfDisplay(stdout,true);
	CHECK_ERROR();
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
