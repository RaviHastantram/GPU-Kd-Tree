#include <iostream>

#include "kdtypes.h"
#include "util.h"
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
	//KDTree * kd = new KDTree;
	//printMesh(m);
	
	copyToGPU(m);
	
	GPUTriangleArray *d_triangleArray = new GPUTriangleArray();
	GPUNodeArray *d_nodeArray = new GPUNodeArray();

	int numActiveNodes=1;
	int numActiveTriangles=m->numTriangles;
	int threadsPerNode = 0;

	while(numActiveNodes>0)
	{
		threadsPerNode = getThreadsPerNode(numActiveNodes,numActiveTriangles);
		
		computeCost <<< numActiveNodes,threadsPerNode >>>(d_nodeArray,d_triangleArray);

		splitNodes<<<numActiveNodes,threadsPerNode>>>(d_nodeArray,d_triangleArray);

		cudaThreadSynchronize();
		
		numActiveNodes = getActiveNodes();
		numActiveTriangles = getActiveTriangles();		
	}

	//copyToHost(kd);

	//kd->verifyTree();
	//kd->printTreeStats();

	//dumpKDTree(kd);
	
	return 0;
}
