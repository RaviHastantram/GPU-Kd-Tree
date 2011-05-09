#include "kdtypes.h"
#include "geom.h"

using namespace std;

Mesh * createMesh(uint32 numTriangles, uint32 numPoints)
{
	Mesh * m = new Mesh;
	m->triangles = new Triangle[numTriangles];
	m->points = new Point[numPoints];
	return m;
}

void destroyMesh(Mesh * m)
{
	delete [] m->triangles;
	delete [] m->points;
	delete m;
}
