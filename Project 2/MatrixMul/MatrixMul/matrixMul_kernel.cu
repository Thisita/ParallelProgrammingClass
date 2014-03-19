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
#include <cublas_v2.h>

#define block_size 32

/*
#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif
*/

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B (No shared memory used)
//! hA is A's height, wA is A's width and wB is B's width 
//! In other words, it computes hAxwA matrix multiplied by wAxWB matrix.
//! Assume matrices are stored in row-major linear array, and matrix indexing is 0-based.
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void
matrixMul_v1( float* C, float* A, float* B, int hA, int wA, int wB)
{
    // index of the C matrix element computed by this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
    // Cvalue is used to store the element of the C matrix computed by this thread
	float Cvalue;
    if (row < hA && col < wB)
    {
		Cvalue = 0.0f;

		// Loop over all the matrices of A and B required to compute C
		for (int k = 0; k < wA; k++) 
			Cvalue += A[row * wA + k] * B[k * wB + col];

		// Write the block sub-matrix to device memory; each thread writes one element
		C[row * wB + col] = Cvalue;
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B (shared memory used)
//! hA is A's height, wA is A's width and wB is B's width 
//! In other words, it computes hAxwA matrix multiplied by wAxWB matrix.
//! Assume matrices are stored in row-major linear array, and matrix indexing is 0-based.
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void
matrixMul_v2( float* C, float* A, float* B, int hA, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * block_size  * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = block_size;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = block_size * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = block_size * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) 
    {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[block_size][block_size];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[block_size][block_size];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < block_size; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * block_size * by + block_size * bx;
    C[c + wB * ty + tx] = Csub;
}


/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
/*
template <int block_size> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * block_size * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = block_size;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = block_size * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = block_size * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[block_size][block_size];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[block_size][block_size];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < block_size; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * block_size * by + block_size * bx;
    C[c + wB * ty + tx] = Csub;
}


int matrixMultiply(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t error;

    error = cudaMalloc((void **) &d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16)
    {
        matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    else
    {
        matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    printf("done\n");

    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Execute the kernel
    int nIter = 300;

    for (int j = 0; j < nIter; j++)
    {
        if (block_size == 16)
        {
            matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else
        {
            matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }

    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    // Copy result from device to host
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula 
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps 
    double eps = 1.e-6 ; // machine zero
    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
    {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err/abs_val/dot_length ;
        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x*valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nNote: For peak performance, please refer to the matrixMulCUBLAS example.\n");

    cudaDeviceReset();

    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}

*/

//
// Our CUDA matrix multiplication interface function
//
int GPU_SGEMM (float* C, float* A, float* B, int HA, int WA, int WB, int version)		
{
    float *d_A = 0;
    float *d_B = 0;
    float *d_C = 0;
    // float alpha = 1.0f;
    // float beta = 0.0f;

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
	if (version == 1)
		matrixMul_v1<<< grid, block >>>(d_C, d_A, d_B, HA, WA, WB); 
	else 
		matrixMul_v2<<< grid, block >>>(d_C, d_A, d_B, HA, WA, WB);   
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

	return 0;
}
