#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>

__device__ thrust::device_vector<int> d_vec(10);

__global__ void testKernel()
{
	int threadidx = threadIdx.x*blockIdx.x + threadIdx.x;
	d_vec[threadidx] = threadidx;
	//if(threadidx > d_vec.capacity() -1)
	//	d_vec.reserve(2*(threadidx+1));
	//d_vec[threadidx] = threadidx;
}

int main(void)
{
	// generate 32M random numbers on the host
    //thrust::host_vector<int> h_vec(32 << 20);
    //thrust::generate(h_vec.begin(), h_vec.end(), rand);

    // transfer data to the device
    //thrust::device_vector<int> d_vec(10);
	
    testKernel <<<10,10>>> ();

    // sort data on the device (846M keys per second on GeForce GTX 480)
    //thrust::sort(d_vec.begin(), d_vec.end());

    printf("%d", d_vec.size()) ;

}


