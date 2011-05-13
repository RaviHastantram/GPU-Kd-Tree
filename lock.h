#ifndef __CUDA_LOCK_H__
#define __CUDA_LOCK_H__

#include <cuda.h>

struct Lock {
    int *mutex;
    Lock( void ) {
        HANDLE_ERROR( cudaMalloc( (void**)&mutex,
                              sizeof(int) ) );
        HANDLE_ERROR( cudaMemset( mutex, 0, sizeof(int) ) );
    }

    ~Lock( void ) {
        cudaFree( mutex );
    }

    __device__ void lock( void ) {
        while( atomicCAS( mutex, 0, 1 ) != 0 );
    }

    __device__ void unlock( void ) {
        atomicExch( mutex, 0 );
    }
};

#endif
