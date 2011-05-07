#ifndef __UTIL_H__
#define __UTIL_H__

#include <cstdio>

using namespace std;

#include "rply.h"
#include "geom.h"

static int vertex_cb(p_ply_argument argument);
static int face_cb(p_ply_argument argument);
Mesh * loadMeshFromPLY(const char * filename);

#endif
