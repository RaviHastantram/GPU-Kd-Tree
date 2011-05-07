#include "Mesh.h"
#include "kdtree.h"
#include "file.h"

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

void DumpKDTree(Nodes *nodes, int numNodes)
{
	//1. Write the LAYOUT_VERSION.
	//2. Write the Bounds
	FILE* fp = fopen("GPU-Kd-tree","w+");
	
	//3. Write the number of nodes.
	fprintf(fp,"%d\n" numNodes);

	//4. Write the nodes.
	for(int i = 0; i < numNodes; i++)
	{
		
	}
	
}

void DumpNode(NodeID nodeID)
{
	
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
