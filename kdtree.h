#ifndef __KDTREE_H__
#define __KDTREE_H__

#define SETSPLITNONE(idx) idx=(idx & 0x3FFFFFFF) // 0100 0000 0000 0000
#define SETSPLITX(idx) idx=((idx & 0x3FFFFFFF) | 0x40000000) // 0100 0000 0000 0000
#define SETSPLITY(idx) idx=((idx & 0x3FFFFFFF) | 0x80000000) // 1000 0000 0000 0000
#define SETSPLITZ(idx) idx=((idx & 0x3FFFFFFF) | 0xC0000000) // 1100 0000 0000 0000
#define SETINDEX(idx,index) idx=(idx & 0xC0000000) | (0x3FFFFFFF & index)
#define GETINDEX(idx) (idx & 0x3FFFFFFF)
#define GETSPLIT(idx) (id & 0xC0000000) >> 30)
#define ISLEAF(idx) ((idx & 0xC0000000) == 0x00000000)   
#define ISSPLITX(idx) (idx & 0xC0000000) == 0x00000001) 
#define ISSPLITY(idx) (idx & 0xC0000000) == 0x00000002) 
#define ISSPLITZ(idx) (idx & 0xC0000000) == 0x00000003) 

typedef uint32 NodeIndex;

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
	void splitBox(const uint16 & dim,  
				  float value, 
				  const BoundingBox & bbox, 
				  const BoundingBox * bboxes);
	
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
	
	Node * left(Node * n) const;
	Node * right(Node * n) const;
	void  setLeft(Node * n, KNode * l);
	void setRight(Node * n, KNode * r);
	float getSplit(Node * n) const;
	void setSplit(Node * n, float value);
	
	//------------------------------------------------------------------
	// leaves
	//------------------------------------------------------------------
	
	int size(Node * n) const;   
	Triangles ** triangles(KNode * n) const;
	Triangle * get(KNode * k, int i) const;
	int put(KNode * n, Geometry * o, int i);
	void swap(Node * n, int i, int j) const;	
};

#endif