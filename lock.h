#ifndef __CUDA_LOCK_H__
#define __CUDA_LOCK_H__

#include <cuda.h>

class Lock {

	public:
		__device__ Lock (void ) {
			mutex[0]=0;
		}

		__device__ void lock( void ) {
			while( atomicCAS( (int *) mutex, 0, 1) != 0);
		}
		
		__device__ void unlock( void ) {
			atomicExch( mutex, 1);
		}

	private:
		int  mutex[1];
};	

#endif
