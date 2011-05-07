#ifndef __KDTREE_H__
#define __KDTREE_H__

#define SETSPLITNONE(id) id=(id & 0x3FFFFFFF) // 0100 0000 0000 0000
#define SETSPLITX(id) id=((id & 0x3FFFFFFF) | 0x40000000) // 0100 0000 0000 0000
#define SETSPLITY(id) id=((id & 0x3FFFFFFF) | 0x80000000) // 1000 0000 0000 0000
#define SETSPLITZ(id) id=((id & 0x3FFFFFFF) | 0xC0000000) // 1100 0000 0000 0000
#define SETINDEX(id,index) id=(id & 0xC0000000) | (0x3FFFFFFF & index)
#define GETINDEX(id) (id & 0x3FFFFFFF)
#define GETSPLITDIM(id) (id & 0xC0000000) >> 30)
#define ISLEAF(id) ((id & 0xC0000000) == 0x00000000)   
#define ISSPLITX(id) (id & 0xC0000000) == 0x00000001) 
#define ISSPLITY(id) (id & 0xC0000000) == 0x00000002) 
#define ISSPLITZ(id) (id & 0xC0000000) == 0x00000003) 

typedef uint32 NodeID;

struct  TreeStats
{
	TreeStats():
	traversalCost(0),
	intersectionCost(0),
	averageLeafDepth(0),
	nLeaves(0),
	nNodes(0),
	sceneArea(0),
	nEmptyLeaves(0),
	nAverageNonEmptySize(0)
	{}
	float averageLeafDepth;
	float traversalCost;
	float intersectionCost;
	float treeCost;
	uint32 nLeaves;
	uint32 nNodes;
	float sceneArea;
	uint32 nEmptyLeaves;
	uint32 nAverageNonEmptySize;
	BoundingBox box;
};

struct Node {
	uint32 * objectIDs;  //  1.  if internal: id of left child, followed by id of right child
						 //  2.  if leaf: list of triangle ids
	union {
		float split;
		uint32 size;
	} v; // this nodes value
};

class KDTree {
	
public:
	
	Node * root;
	BoundingBox  bounds;
	Mesh * mesh;
	uint32 numNodes;  // after moved back to CPU side
	Node * nodes;     
	
	static uint32 MIN_LEAF_SIZE;
	static uint32 MAX_DEPTH;
	static float K_ISECT;
	static float K_TRAV;	
	static float lambda;
	
	uint32 leafCount;
	uint32 internalCount;
	uint32 treeDepth;
	uint32 minLeafSize;
	uint32 maxLeafSize;
	float averageLeafDepth;
	uint32 emptyLeafCount;
	float cost;
	
	uint32 nIntersect;
	uint32 nTraverse;
	uint32 nOperations;
	uint32 nRays;
	uint32 nRaysActual;
	
	//--------------------------------------------------------------------
	// stats functions
	//--------------------------------------------------------------------
	
	void computeTreeStats(Node * node, TreeStats & stats);
	static void initTraversalStats();
	static void printTraversalStats();

	//--------------------------------------------------------------------
	// traversal functions
	//--------------------------------------------------------------------
	
	// TODO 
	
	//--------------------------------------------------------------------
	// build functions
	//--------------------------------------------------------------------
	
	void build();
	void verify();
	void splitBoundingBox(const uint16 & dim,  
						  float value, 
						  const BoundingBox & bbox, 
						  const BoundingBox * bboxes) const;
	float boxArea(const BoundingBox & bbox) const;
	
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
	
	void  destroy(Node * k);
		
	//--------------------------------------------------------------------
	// internal nodes
	//--------------------------------------------------------------------
	
	Node * getNode(NodeID nodeID);
	Node * left(Node * n) const;
	Node * right(Node * n) const;
	NodeID leftID(Node * n) const;
	NodeID rightID(Node * n) const;
	void  setLeft(Node * n, Node * l);
	void setRight(Node * n, Node * r);
	float getSplit(Node * n) const;
	void setSplit(Node * n, float value);
	NodeID * children(Node * n);
	
	
	//------------------------------------------------------------------
	// leaves
	//------------------------------------------------------------------
	
	uint32 size(Node * n) const;   
	uint32 * triangles(Node * n) const;
	Triangle * get(Node * k, uint32 triangleID) const;
	int put(KNode * n, Geometry * o, uint32 i);
	
};

#endif