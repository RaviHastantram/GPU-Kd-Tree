#ifndef __GEOM_H__
#define __GEOM_H__

struct Point {
	float values[3];
};

struct Triangle {
	uint32 index; // index of this triangle
	uint32 ids[3];
};

struct BoundingBox {
	float min[3];
	float max[3];
};

struct Mesh {
	uint32 numTriangles; // number of triangles
	uint32 numPoints;
	Triangle * triangles;	// list of triangles
	Point * points;	// list of points
};

Mesh * createMesh(uint32 size);
void destroyMesh();

#endif
