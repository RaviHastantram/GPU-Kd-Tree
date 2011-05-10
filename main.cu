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
	uint32 activeOffset;

	thrust::device_vector<int> d_countsVec(MAX_BLOCKS);
	int * d_counts = thrust::raw_pointer_cast(&d_vec[0]);
	int count=0;

	while(numActiveNodes>0)
	{
		cudaMemcpy(&d_activeOffset,&activeOffset,sizeof(uint32),0,cudaMemcpyHostToDevice);

		threadsPerNode = getThreadsPerNode(numActiveNodes,numActiveTriangles);
		
		computeCost <<< numActiveNodes,threadsPerNode >>>(d_nodeArray,d_triangleArray);

		splitNodes<<<numActiveNodes,threadsPerNode>>>(d_nodeArray,d_triangleArray,d_counts);

		cudaThreadSynchronize();
		
		numActiveNodes = getActiveNodes();
		numActiveTriangles = getActiveTriangles();

		cudaMemcpyFromSymbol(&activeOffset,&d_activeOffset,sizeof(uint32),0,cudaMemcpyDeviceToHost);
		count=thrust::count(d_countsVec.begin(), d_countsVec.end() + numActiveNodes, 1);
		activeOffset += 2*count;		
	}

	//copyToHost(kd);

	//kd->verifyTree();
	//kd->printTreeStats();

	//dumpKDTree(kd);
	
	return 0;
}
