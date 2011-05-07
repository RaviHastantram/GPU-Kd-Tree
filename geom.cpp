#include "triangle.h"

Mesh * createMesh(uint32 numTriangles, uint32 numPoints)
{
	Mesh * m = new Mesh;
	m->triangles = new Triangle[numTriangles];
	m->boxes = new BoundingBox[numTriangles];
	m->points = new Point[numPoints];
}

void destroyMesh(Mesh * m)
{
	delete [] m->triangles;
	delete [] m->boxes;
	delete [] m->points;
	delete m;
}