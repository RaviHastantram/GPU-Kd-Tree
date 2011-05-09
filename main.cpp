#include <iostream>

#include "kdtypes.h"
#include "util.h"
#include "geom.h"

using namespace std;

int main(int argc, char  ** argv)
{
	char * inputFile = argv[1];
	// load ply
	Mesh * m = loadMeshFromPLY(inputFile);
	KDtree * kd = new KDTree;
	//printMesh(m);
	
	copyToGPU(m);

	numActiveNodes=1;
	numActiveTriangles=m->numTriangles;

	while(numActiveNodes>0)
	{
		threadsPerNode = threadsPerNode(numActiveNodes,numActiveTriangles);
		
		computeCost<<<numActiveNodes,threadsPerNode>>();

		splitNodes<<<numActiveNodes,threadsPerNode>>();

		numActiveNodes = numActiveNodes();
		numActiveTriangles = numActiveTriangles();		
	}

	copyToHost(kdtree);

	kd->verifyTree();
	kd->printTreeStats();

	dumpKDTree(kd);
	
	return 0;
}
