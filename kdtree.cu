#include "kdtree.h"

#include <iostream>

using namespace std;

uint32 KDTree::MIN_LEAF_SIZE=1;
uint32 KDTree::MAX_DEPTH=25;
float KDTree::K_ISECT=10;
float KDTree::K_TRAV=1;	
float KDTree::lambda=0.8;

//--------------------------------------------------------------------
// stats functions
//--------------------------------------------------------------------

void KDTree::printTreeStats()
{
}

void KDTree::computeTreeStats()
{
}

void KDTree::computeTreeStats(NodeID nodeID, TreeStats & stats)
{
	Node * node = getNode(nodeID);
	
	float PR=boxArea(stats.box)/stats.sceneArea;
	
	if(ISLEAF(nodeID))
	{
		stats.averageLeafDepth=0;
		stats.nLeaves=1;
		stats.nNodes=0;
		stats.traversalCost=0;
		stats.intersectionCost=K_ISECT*size(node)*PR;
		stats.nEmptyLeaves = (size(node)>0?0:1);
		stats.nAverageNonEmptySize=size(node);
	} else {
		TreeStats lstats,rstats;
		
		BoundingBox pair[2];
		splitBoundingBox(GETSPLITDIM(nodeID),  getSplit(node), stats.box, pair);
													
		lstats.box=pair[0];
		lstats.sceneArea=stats.sceneArea;
		
		rstats.sceneArea=stats.sceneArea;
		rstats.box=pair[1];
		
		computeTreeStats(leftID(node),lstats);
		computeTreeStats(rightID(node),rstats);
		
		stats.traversalCost=lstats.traversalCost+rstats.traversalCost+K_TRAV*PR;
		stats.intersectionCost=lstats.intersectionCost+rstats.intersectionCost;
		
		stats.nLeaves=lstats.nLeaves+rstats.nLeaves;
		stats.nNodes=lstats.nNodes+rstats.nNodes+1;
		
		stats.nEmptyLeaves = lstats.nEmptyLeaves+rstats.nEmptyLeaves;
		
		int lcount=lstats.nLeaves;
		int rcount=rstats.nLeaves;	
		float lsum=lcount*lstats.averageLeafDepth;
		float rsum=rcount*rstats.averageLeafDepth;
		
		
		stats.averageLeafDepth=(lsum+rsum+lcount+rcount)/(lcount+rcount);
		
		lcount = lstats.nLeaves-lstats.nEmptyLeaves;
		rcount = rstats.nLeaves-rstats.nEmptyLeaves;
		lsum=lstats.nAverageNonEmptySize*lcount;
		rsum=rstats.nAverageNonEmptySize*rcount;
		
		stats.nAverageNonEmptySize=(lsum+rsum+lcount+rcount)/(lcount+rcount);
		
	}
	stats.treeCost=stats.intersectionCost+stats.traversalCost;
}

void KDTree::initTraversalStats()
{
	nIntersect=0;
	nTraverse=0;
	nOperations=0;
	nRays=0;
	nRaysActual=0;
}

void KDTree::printTraversalStats()
{
	cout << "Hitbox Rays:"<<KDTree::nRays<<endl;
	cout << "Total Rays:"<<KDTree::nRaysActual<<endl;
	cout << "Intersections:"<<KDTree::nIntersect<<endl;
	cout << "Int. per Ray:"<<((float)KDTree::nIntersect)/KDTree::nRays<<endl;
	cout << "Traversals:"<<KDTree::nTraverse<<endl;
	cout << "Operations:"<<KDTree::nTraverse+KDTree::nIntersect<<endl;
}

//--------------------------------------------------------------------
// traversal functions
//--------------------------------------------------------------------

// TODO 

//--------------------------------------------------------------------
// build functions
//--------------------------------------------------------------------

float KDTree::boxArea(const BoundingBox & box) const
{
	float diffx = box.max[0]-box.min[0];
	float diffy = box.max[1]-box.min[1];
	float diffz = box.max[2]-box.min[2];
	float area = 2*diffx*diffy + 2*diffx*diffz + 2*diffy*diffz;
	return area;
}
	
void KDTree::splitBoundingBox(const uint32& dim,  float value, 
			  const BoundingBox & box, BoundingBox * pair)
{
    for(int i=0;i<3;i++)
    {
		pair[0].min[i]=box.min[i];
		pair[0].max[i]=box.max[i];
		pair[1].min[i]=box.min[i];
		pair[1].max[i]=box.max[i];	
    }
    pair[0].max[dim] = value;
    pair[1].min[dim]  = value;	
}

void KDTree::build()
{
	// TODO
}

//---------------------------------------------------------------------
// split functions
//---------------------------------------------------------------------

// TODO

///--------------------------------------------------------------------
// SAH functions
//--------------------------------------------------------------------

// TODO

//--------------------------------------------------------------------
// general functions
//--------------------------------------------------------------------
void  KDTree::destroy(NodeID nodeID)
{
	Node * node = getNode(nodeID);
	if(ISLEAF(nodeID))
	{
		delete [] triangles(node);
	} else
	{
	    if(left(node))
			destroy(leftID(node));
	    if(right(node))
			destroy(rightID(node));
	    if(node->objectIDs)
			delete [] node->objectIDs;
	}    
}

//--------------------------------------------------------------------
// internal nodes
//--------------------------------------------------------------------
NodeID KDTree::leftID(Node * n) const
{
	return n->objectIDs[0];
}

NodeID KDTree::rightID(Node * n) const
{
	return n->objectIDs[1];
}

NodeID * KDTree::children(Node * n)
{
	return n->objectIDs;
}

Node * KDTree::left(Node * n) const
{
	return &nodes[n->objectIDs[0]];
}

Node * KDTree::right(Node * n) const
{
	return &nodes[n->objectIDs[1]];
}

void  KDTree::setLeft(Node * n, Node * l)
{
	nodes[n->objectIDs[0]]=*l;
}

void KDTree::setRight(Node * n, Node * r)
{
	nodes[n->objectIDs[1]]=*r;
}

float KDTree::getSplit(Node * n) const
{
	return n->v.split;
}

void KDTree::setSplit(Node * n, float value)
{
	n->v.split=value;
}

Node * KDTree::getNode(NodeID id)
{
	return &nodes[id];
}
				 
//------------------------------------------------------------------
// leaves
//------------------------------------------------------------------
uint32 KDTree::size(Node * n)
{
	return n->v.size;
}

uint32 * KDTree::triangles(Node * n)
{
	return n->objectIDs;
}

Triangle * KDTree::get(Node * n, uint32 triangleID) const
{
	return &mesh->triangles[n->objectIDs[triangleID]];
}

void KDTree::put(Node * n, uint32 triangleID, uint32 i)
{
	n->objectIDs[i]=triangleID;
}
