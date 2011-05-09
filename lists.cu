#include "lists.h"

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

__device__ void initializeTreeBuild()
{
	d_numActiveTriangles=d_mesh->numTriangles;
	d_numActiveNodes=1;	
}

////////////////////////
//
//  activeTriangles functions
//
///////////////////////

__device__ void initActiveTriangles(uint32 numBlocks)
{
	if(activeTriangles!=0)
	{
		cudaFree(activeTriangles);
	}
	cudaMalloc( (void **) activeTriangleList, numBlocks*sizeof(activeTriangles));
}

__device__ void addTriangle(Triangle * t, int listIndex)
{
	TriangleList * list = &activeTriangles[listIndex];
	if(list->size==list->capacity)
	{
		list->capacity=2*(list->size+1);
		Triangle * pNew;
		cudaMalloc( (void **) &pNew, list->capacity*sizeof(Triangle *));
		cudaMemcpy( pNew, list->triangles, sizeof(Triangle *)*list->size); 
		cudaFree(list->triangle);
		list->triangles=pNew;
	}
	list->triangles[list->size-1]=t;
	list->size++;
}

__device__ Triangle * getTriangle(int listIndex, int triangleIndex)
{
	return activeTriangles[listIndex]->list[triangleIndex];	
}

__device__ uint32 getNumTriangles(int listIndex)
{
	return activeTriangles[listIndex]->size;
}
