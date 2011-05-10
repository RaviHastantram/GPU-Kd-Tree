#include <iostream>

#include "kdtypes.h"
#include "util.h"
#include "geom.h"
#include "kdtree.h"
#include "gpuBuilder.h"
#include "util.h"

using namespace std;

int main(int argc, char  ** argv)
{
	char * inputFile = argv[1];
	// load ply
	Mesh * m = loadMeshFromPLY(inputFile);
	KDTree * kd = new KDTree;
	//printMesh(m);
	
	copyToGPU(m);

	int numActiveNodes=1;
	int numActiveTriangles=m->numTriangles;
	int threadsPerNode = 0;

	while(numActiveNodes>0)
	{
		threadsPerNode = getThreadsPerNode(numActiveNodes,numActiveTriangles);
		
		computeCost <<< numActiveNodes,threadsPerNode >>>();

		splitNodes<<<numActiveNodes,threadsPerNode>>>();

		cudaThreadSynchronize();
		
		numActiveNodes = getActiveNodes();
		numActiveTriangles = getActiveTriangles();		
	}

	copyToHost(kd);

	//kd->verifyTree();
	kd->printTreeStats();

	dumpKDTree(kd);
	
	return 0;
}
