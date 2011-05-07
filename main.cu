
#include <iostream>
#include <cuda.h>

using namespace std;

__global__ void computecost()
{
	
}

__global__ void simpleKernel(int* d_a,int size)
{
	//Initialized in the device
	int arrayIndex = threadIdx.x + blockIdx.x*gridDim.x; 
	d_a[arrayIndex] = arrayIndex;
}

int main(int argc, char  ** argv)
{
	// load ply
	// ship to gpu
	// import from gpu
	// compute some stats (check goodness of tree)
	// render
	int *h_a, *d_a;
	int size = 100*sizeof(int);
	h_a = (int*)malloc(size);
	
	//Allocate memory on Device
	cudaMalloc(&d_a,size);
	//Launch the kernel.
	simpleKernel <<< 10,10 >>> (d_a,size);
	//Copy the data back to host
	cudaMemcpy(h_a,d_a,size,cudaMemcpyDeviceToHost);

	for(int i = 0; i < 100; i++)
	{
		cout << h_a[i] << " " ;
	}
	cout << endl;
	cudaFree(d_a);
	free(h_a);
	return 0;
}


