#include <iostream>

#include "kdtypes.h"
#include "util.h"
#include "geom.h"

using namespace std;

int main(int argc, char  ** argv)
{
	char * inputFile = argv[1];
	// load ply
	loadMeshFromPLY(inputFile);
	
	// ship to gpu
	// import from gpu
	// compute some stats (check goodness of tree)
	// render
	
	return 0;
}
