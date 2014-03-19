// matrixMul_kernel.cu
//
//	Matrix multiplication: C = A * B.
// 
//

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "CH\chTimer.h"

#define block_size 32

/*
 * Fast impl of CUDA matrix multiplication using shared memory
 *
 * My version removes all the unnecessary vram allocations,
 * only allocates the shared vram once, makes use of a lot of
 * code order and calculation order optimizations,
 * and some other tricks :D
 */
extern "C" __global__ void
matrixMulKernel( float* C, float* A, float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
	// c value buffer
	float c = 0.0f;

	// calculated indexes
	unsigned int row = blockIdx.y * block_size + threadIdx.y,
				col = blockIdx.x * block_size + threadIdx.x,
				ai, bi;

	// allocate shared memory
	__shared__ float a[block_size][block_size];
	__shared__ float b[block_size][block_size];

	// start the process
	for(unsigned int k = 0, ke = (block_size + wA - 1) / block_size; k < ke; ++k)
	{
		// precalculate
		ai = k * block_size + threadIdx.x;
		bi = k * block_size + threadIdx.y;

		// copy chunk of A to a
		if(ai < wA && row < hA)
		{
			a[threadIdx.y][threadIdx.x] = A[row * wA + ai];
		}
		else
		{
			// zero out for overallocation
			a[threadIdx.y][threadIdx.x] = 0.0f;
		}

		// copy chunk of B to b
		if(bi < wA && col < wB)
		{
			b[threadIdx.y][threadIdx.x] = B[bi * wB + col];
		}
		else
		{
			// zero out for overallocation
			b[threadIdx.y][threadIdx.x] = 0.0f;
		}

		// sync the shared mem
		__syncthreads();

		// calculate value of c
		for(unsigned int n = 0; n < block_size; ++n)
		{
			c += a[threadIdx.y][n] * b[n][threadIdx.x];
		}

		// resync to prevent corruption of other threads
		__syncthreads();
	}

	// Copy value to c if it is there
	if( row < hA && col < wB)
	{
		C[((blockIdx.y * blockDim.y + threadIdx.y) * wB) + (blockIdx.x * blockDim.x) + threadIdx.x] = c;
	}
}



//
// Our CUDA matrix multiplication interface function
//
int CUDA_matrixMul(float* C, float* A, float* B, unsigned int HA, unsigned int WA, unsigned int WB)		
{
    float *d_A = 0, *d_B = 0, *d_C = 0;

	cudaError_t cudaStatus;

	// Make sure CUDA device 0 is available
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }
 
    /* Allocate device memory for the matrices (d_A, d_B, and d_C) */
    if (cudaMalloc((void **)&d_A, HA * WA * sizeof(d_A[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }
    if (cudaMalloc((void **)&d_B, WA * WB * sizeof(d_B[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
    }
    if (cudaMalloc((void **)&d_C, HA * WB * sizeof(d_C[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }
	// Copy host memory (A, and B) to device
    cudaStatus = cudaMemcpy(d_A, A, HA*WA*sizeof(A[0]), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaMemcpy (d_A, A) returned error code %d\n", cudaStatus);
        exit(EXIT_FAILURE);
    }
    cudaStatus = cudaMemcpy(d_B, B, WA*WB*sizeof(B[0]), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaMemcpy (d_B, B) returned error code %d\n", cudaStatus);
        exit(EXIT_FAILURE);
    }

	// Setup execution parameters and call kernel
    dim3 block(block_size, block_size);
    dim3 grid ((WB+block_size-1)/block_size, (HA+ block_size-1)/block_size);
	
	// Call kernel
	matrixMulKernel<<< grid, block >>>(d_C, d_A, d_B, HA, WA, WB);  

	// this isn't necessary anymore, but good to have for compatability sake
    cudaDeviceSynchronize();

	// Copy result (C)  from device to host
    cudaStatus = cudaMemcpy(C, d_C, HA*WB*sizeof(C[0]), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaMemcpy (C, d_C) returned error code\n", cudaStatus);
        exit(EXIT_FAILURE);
    }

    /* Device memory clean up */
    if (cudaFree(d_A) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }
    if (cudaFree(d_B) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }
    if (cudaFree(d_C) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }

	return EXIT_SUCCESS;
}
