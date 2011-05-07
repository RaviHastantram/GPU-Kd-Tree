#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

struct Point {
	float x;
	float y;
	float z;
};

struct Triangle {
	uint32 index; // index of this triangle
	uint32 id1;	// index to first point
	uint32 id2;	// index to second point
	uint32 id3;	// index to third point
};

struct BoundingBox {
	float min[3];
	float max[3];
};

struct Mesh {
	uint32 numTriangles; // number of triangles
	uint32 numPoints;
	Triangle * triangles;	// list of triangles
	BoundingBox * boxes; // list of bounding boxes
	Point * points;	// list of points
};

Mesh * createMesh(uint32 size);
void destroyMesh();

#endif
