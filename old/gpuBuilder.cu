#include "kdtree.h"
#include "kdtypes.h"
#include "lists.h"
#include "gpuBuilder.h"
#include <iostream>
#include <fstream>
#include <cstdio>

using namespace std;

///////////////////////////
// 
// Tree Building
//
///////////////////////////
uint32 getActiveNodes()
{
	uint32 numNodes=0;
	 
	if(cudaMemcpyFromSymbol(&numNodes,
				"d_numActiveNodes",
				sizeof(uint32),
				0,
				cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Copying d_numActiveNodes failed.\n");
	}
	return numNodes;
}

uint32 getThreadsPerNode(int numActiveNodes,int numActiveTriangles)
{
	return 0;
}

uint32 getActiveTriangles()
{
	uint32 numTriangles=0;
	
	if(cudaMemcpyFromSymbol(&numTriangles,
				"d_numActiveTriangles",
				sizeof(uint32),
				0,
				cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Copying d_numActiveTriangles failed.\n");
	}
	return numTriangles;
}

__device__ void computeCost()
{
	
}

__device__ void splitNodes()
{

}

////////////////////////////////
//
// Data Import/Export
//
///////////////////////////////
void copyToGPU(Mesh *mesh)
{
	//Copy the Points list
	int size = sizeof(Point)*(mesh->numPoints);
	cudaMalloc(&d_points,size);
	cudaMemcpy(d_points,mesh->points,size,cudaMemcpyHostToDevice);

	//Copy the triangle list
	size = sizeof(Triangle)*(mesh->numTriangles);
	cudaMalloc(&d_triangles, size);
	cudaMemcpy(d_triangles,mesh->triangles,size,cudaMemcpyHostToDevice);

	//Copy the mesh
	size = sizeof(Mesh);
	cudaMalloc(&d_mesh,size);
	cudaMemcpy(d_mesh,mesh,size,cudaMemcpyHostToDevice);
}


void copyToHost(KDTree *kdtree)
{
	//Allocate the Nodes array on the host
	int size = sizeof(Node)*(kdtree->numNodes);

	// TODO - Fix this.  Cannot call malloc fromt the device.	
	//	- But, still need some way to copy the tree back to host.
		
	
	Node *nodes = (Node*)malloc(size*sizeof(Node));
		
	//Copy the nodes array
	for(int i =0 ; i < kdtree->numNodes; i++)
	{
		 copyNode(kdtree,i,nodes); //Copy node "i" from kdtree to nodes array on the host
	}
	
}


void copyNode(KDTree *kdtree,NodeID nodeID, Node* nodes)
{
	// TODO - Fix this.   Not allowed to call host functions from device.

	Node* node = kdtree->getNode(nodeID);
	int size = 0;
	if(ISLEAF(nodeID))
	{
		size = kdtree->size(node);
	}
	else
	{
		size = 2;
	}

	assert(size != 0);
	uint32* objectIDs = (uint32*)malloc(size);
	
	//Copy the children to the host
	cudaMemcpy(objectIDs,node->objectIDs,size,cudaMemcpyDeviceToHost);

	//Copy the node to the host
	size = sizeof(node);
	cudaMemcpy(&nodes[nodeID],node,size,cudaMemcpyDeviceToHost);

	//Replace the triangle list pointer with pointer in host.
	nodes[nodeID].objectIDs = objectIDs;
}	

void dumpKDTree(KDTree *kdtree)
{
	ofstream file("GPU-Kd-tree",ios::out | ios::binary);

	char *buffer = new char[100];
	
	unsigned int version = 1;
	//1. Write the LAYOUT_VERSION.
	file.write((char*)&version,sizeof(unsigned int));

	//2.TODO Write the Bounds
	
	//3. Write the number of nodes.
	uint64_t numNodes = kdtree->numNodes;
	file.write((char*)&numNodes,sizeof(uint64_t));

	//4. Write the nodes.
	for(int i = 0; i < kdtree->numNodes; i++)
	{
		dumpNode(file,i,kdtree);		
	}

	//5.Write the number of leaves
	uint64_t leafcount = (uint64_t)kdtree->leafCount;
	file.write((char*)&leafcount,sizeof(uint64_t));

	//6. Write the triangles
	for(int i = 0; i < kdtree->numNodes; i++)
	{
		if(ISLEAF(i))
		{	
			dumpTriangles(file,i,kdtree);
		}
	}

	file.close();
}

void dumpNode(ofstream& file,NodeID nodeID,KDTree *kdtree)
{
	Node* node = kdtree->getNode(nodeID);
	uint32 data0 = 0;
	uint32 data1 = 0; 
	if(ISLEAF(nodeID))
	{
		data1 = nodeID;
		file.write((char*)&data0, sizeof(uint32));
		file.write((char*)&data1, sizeof(uint32));
	}
	else
	{
		data0 = GETINDEX(nodeID) << 2;
		data0 |= GETSPLITDIM(nodeID);
		
		file.write((char*)&data0, sizeof(uint32));
		file.write((char*)&node->v.split, sizeof(float));
	}
	
}

void dumpTriangles(ofstream& file, NodeID nodeID, KDTree *kdtree)
{
	Node* node = kdtree->getNode(nodeID);
	//7. Write the length of the triangle list
	uint64_t numTriangles = node->v.size;
	file.write((char*)&numTriangles,sizeof(uint64_t));

	//8. Write the triangles
	uint32 triangleID = 0;
	uint32 triangleIndex = 0; //index of the triangle in the PLY file
	for(int i = 0; i < node->v.size; i++)
	{
		triangleID = GETINDEX(node->objectIDs[i]);
		triangleIndex = kdtree->mesh->triangles[triangleID].index;
		file.write((char*)&triangleIndex,sizeof(triangleIndex));
	}
}

