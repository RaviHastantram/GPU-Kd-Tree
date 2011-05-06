#ifndef __KDTREE_H__
#define __KDTREE_H__


#define NULLIDX 
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

struct Node {
	union  {
		Node ** children;
		Triangle * triangles;
	} i;
	union {
		float split;
		int size;
	}
};

class kdtree {
	
};

#endif