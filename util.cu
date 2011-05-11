#include "kdtypes.h"
#include "util.h"
#include "geom.h"
#include <cfloat>

using namespace std;

static Mesh * pCurrMesh=0;
static uint32 currTriangleIndex=0;
static uint32 currPointIndex=0;
static uint32 currDimIndex=0; 

static int vertex_cb(p_ply_argument argument) 
{
    long eol;
	
    ply_get_argument_user_data(argument, NULL, &eol);
	
    //printf("%g", ply_get_argument_value(argument));
	
    Point * pCurrPoint = &pCurrMesh->points[currPointIndex];

    pCurrPoint->values[currDimIndex] = ply_get_argument_value(argument);
    float value = pCurrPoint->values[currDimIndex];

    if(value<pCurrMesh->bounds.min[currDimIndex])
    {
	pCurrMesh->bounds.min[currDimIndex]=value;
    }
    if(value>pCurrMesh->bounds.max[currDimIndex])
    {
	pCurrMesh->bounds.max[currDimIndex]=value;
    }

    if (eol) 
    {
	        currDimIndex=0;
		currPointIndex++;
//		printf("\n");
    }
    else
    {
		currDimIndex++;
//		printf(" ");
    }
	
    return 1;
}

static int face_cb(p_ply_argument argument) {
    long length, value_index;
	
    ply_get_argument_property(argument, NULL, &length, &value_index);
	
   /** switch (value_index) 
	{
        case 0:
        case 1: 
            printf("%g ", ply_get_argument_value(argument));
            break;
        case 2:
            printf("%g\n", ply_get_argument_value(argument));
            break;
        default: 
            break;
    }**/

    Triangle * pCurrTriangle = &pCurrMesh->triangles[currTriangleIndex];

    pCurrTriangle->ids[value_index]=ply_get_argument_value(argument);
  

    if(value_index==2)
    {
       currTriangleIndex++;
    }
	
    return 1;
}

void printMesh(Mesh * m)
{
	printf("Points:\n");
	for(int i=0;i<m->numPoints;i++)
	{
		printf("%d:(%f,%f,%f)\n",
			i,
			m->points[i].values[0],
			m->points[i].values[1],
			m->points[i].values[2]);
	}
	printf("Triangles:\n");
	for(int j=0;j<m->numTriangles;j++)
	{
		printf("%d:(%d,%d,%d)\n",
			j,
			m->triangles[j].ids[0],
			m->triangles[j].ids[1],
			m->triangles[j].ids[2]);
	}
}

Mesh * loadMeshFromPLY(const char * filename)
{
    long nvertices, ntriangles;
	
    p_ply ply = ply_open(filename, NULL);
	
    if (!ply) 
    {
		return NULL;
    }
    if (!ply_read_header(ply)) 
    {
		return NULL;
    }
	
    pCurrMesh = new Mesh;

    nvertices = ply_set_read_cb(ply, "vertex", "x", vertex_cb, NULL, 0);

    pCurrMesh->numPoints = nvertices;
    pCurrMesh->points = new Point[nvertices];
    currPointIndex=0;
	
    ply_set_read_cb(ply, "vertex", "y", vertex_cb, NULL, 0);
	
    ply_set_read_cb(ply, "vertex", "z", vertex_cb, NULL, 1);
	
    ntriangles = ply_set_read_cb(ply, "face", "vertex_indices", face_cb, NULL, 0);
	
    pCurrMesh->numTriangles = ntriangles;
    pCurrMesh->triangles = new Triangle[ntriangles];
    currTriangleIndex=0;

    printf("Loaded %ld points and %ld vertices.\n", nvertices, ntriangles);
	
    if (!ply_read(ply)) 
	{
		return NULL;
	}
	
    ply_close(ply);
	
    return pCurrMesh;
}
