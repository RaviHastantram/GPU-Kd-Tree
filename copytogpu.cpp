#include "Mesh.h"
#include "kdtree.h"
#include "file.h"
#include "iostream.h"
typedef unsigned __int64 uint64_t;

void CopytoGPU(Mesh *mesh)

{
	Point* d_points;
	Triangle* d_triangles;
	Mesh* d_mesh;

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

__global__ void CopytoHost(KDTree *kdtree)
{
	//Allocate the Nodes array on the host
	int size = sizeof(Node)*(kdtree->numNodes);
	Node *nodes = (Node*)malloc(size);
		
	//Copy the nodes array
	for(int i =0 ; i < numNodes; i++)
	{
		CopyNode(kdtree,i,nodes); //Copy node "i" from kdtree to nodes array on the host
	}	
}

__global__ void CopyNode(KDTree *kdtree,NodeID nodeID, Node* nodes)
{
	Node* node = kdtree->getNode(nodeID);
	uint32 *triangles;
	int size = 0;
	if(ISLEAF(nodeID))
	{
		size = kdtree->size(node)
	}
	else
	{
		size = 2;
	}

	assert(size != 0);
	objectIDs = (uint32*)malloc(size);
	
	//Copy the children to the host
	cudaMemcpy(objectIDs,node->objectIDs,size,cudaMemcpyDeviceToHost);

	//Copy the node to the host
	size = sizeof(node);
	cudaMemcpy(nodes[i],node,size,cudaMemcpyDeviceToHost);

	//Replace the triangle list pointer with pointer in host.
	nodes[i]->objectIDs = triangles;
}	

void DumpKDTree(KDTree *kdtree)
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
		DumpNode(file,i,kdtree);		
	}

	//5.Write the number of leaves
	uint64_t leafcount = kdtree->leafcount;
	file.write((char*)&leafcount,sizeof(uint64_t));

	//6. Write the triangles
	for(int i = 0; i < kdtree->numNodes; i++)
	{
		if(ISLEAF(i))
		{	
			DumpTriangles(file,i,kdtree);
		}
	}

	file.close();
}

void DumpNode(ofstream& file,NodeID nodeID,KDTree *kdtree)
{
	Node* node = kdtree->getNode(nodeID);
	uint32 data0 = 0;
	uint32 data1 = 0;
	if(ISLEAF(i))
	{
		//TODO

	}
	else
	{
		data0 = GETINDEX(nodeID) << 2;
		data0 |= GETSPLITDIM(nodeID);
		
		file.write((char*)&data0, sizeof(uint32));
		file.write((char*)&node->v.split, sizeof(float));
	}
	
}

void DumpTriangles(ofstream& file, NodeID nodeID, KDTree *kdtree)
{
	Node* node = kdtree->getNode(nodeID);
	//7. Write the length of the triangle list
	uint64_t numTriangles = node->v.size;
	file.write((char*)&numTriangles,sizeof(uint64_t));

	//8. Write the triangles
	uint32 triangleID = 0;
	uint32 triangleIndex = 0; //index of the triangle in the PLY file
	for(int i = 0; i < node->v.size);
	{
		triangleID = GETINDEX(node->objectIDs[i]);
		triangleIndex = kdtree->mesh->triangles[triangleID].index;
		file.write((char*)&triangleIndex,sizeof(triangleIndex));
	}
}


__global__ void BuildKDTree()
{
	
}

__device__ void computecost()
{
	
}

__device__ void split()
{

}
