#include "Mesh.h"
#include "kdtree.h"
#include "kdtypes.h"
#include <cfile>
#include <iostream>

using namespace std;

typedef unsigned __int64 uint64_t;

struct NodeList {
	int size;
	int capacity;
	Node ** nodes;
};

struct TriangleList {
	int size;
	int capacity;
	Triangle ** triangles;
};

__device__ Point* d_points;
__device__ Triangle* d_triangles;
__device__ Mesh* d_mesh;
__device__ uint32 d_numActiveTriangles;
__device__ uint32 d_numActiveNodes;
__device__ bool treeBuildInitialized=false;
__device__ NodeList * activeNodes=0;
__device__ NodeList * pendingNodes=0;
__device__ NodeList * completedNodes=0;
__device__ TriangleList * activeTriangles=0;


void CopytoGPU(Mesh *mesh)

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

int numActiveNodes()
{
	uint32 numNodes=-1;
	 
	if(cudaMemcpyFromSymbol(&numNodes,
				"d_numActiveNodes",
				sizeof(uint32),
				0,
				cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Copying d_numActiveNodes failed.\n"):
	}
	return numNodes;;
}

int numActiveTriangles()
{
	uint32 numTriangles=-1;
	
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


////////////////////////
//
//  NodeList functions
//
///////////////////////
__device__ void addNode(Node * node, NodeID nodeID, NodeList * list)
{
	if(list->size==list->capacity)
	{
		list->capacity=2*(list->size+1);
		Nodes * pNew;
		cudaMalloc( (void **)&pNew,list->capacity*sizeof(Node *));
		cudaMemcpy(pNew, list->nodes, list->size*sizeof(Node), cudaMemcpyDeviceToDevice);
		cudaFree(list->nodes);
		list->nodes=pNew;
	} 
	uint32 count=2;
	if(ISLEAF(nodeID))
	{	
		count=KDTree::size(node);	
	}
	list->nodes[list->size-1]=node;
	list->size++;
}

__device__ void initList(NodeList * list)
{
	list->size=0;
	list->capacity=1024;
	cudaMalloc((void **)&list->nodes, list->capacity*sizeof(Node *));
}

__device__ void clearList(NodeList * list) 
{
	cudaFree(list->nodes);
	initList(list);
}

__device__ void swapLists(NodeList ** list1, NodeList ** list2)
{
	NodeList * temp=*list1;
	*list1=*list2;
	*list2=*temp;
}

__device__ Node * getNodeAt(uint32 idx, NodeList * list)
{
	return list->nodes[idx];
}

__device__ void InitializeTreeBuild()
{
	d_numActiveTriangles=d_mesh->numTriangles;
	d_numActiveNodes=1;	
}

////////////////////////
//
//  activeTriangles functions
//
///////////////////////
__device__ void InitActiveTriangles(uint32 numBlocks)
{
	if(activeTriangles!=0)
	{
		cudaFree(activeTriangles);
	}
	cudaMalloc( (void **) activeTriangleList, numBlocks*sizeof(activeTriangles));
}

__device__ void AddTriangle(Triangle * t, int listIndex)
{
	TriangleList * list = &activeTriangles[listIndex];
	if(list->size==list->capacity)
	{
		list->capacity=2*list->size;
		Triangle * pNew;
		cudaMalloc( (void **) &pNew, list->capacity*sizeof(Triangle *));
		cudaMemcpy( pNew, list->triangles, sizeof(Triangle *)*list->size); 
		cudaFree(list->triangle);
		list->triangles=pNew;
	}
	list->triangles[list->size-1]=t;
	list->size++;
}

// KERNEL
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

// KERNEL
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
		data1 = i;
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
