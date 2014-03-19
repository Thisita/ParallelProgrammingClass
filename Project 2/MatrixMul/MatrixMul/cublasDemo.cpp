#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "CH\chTimer.h"
#include <cublas_v2.h>


// Single precision matrix multiplication using CUBLAS
//	C = A * B
//
// CUBLAS function called
//
//  cublasSgemm
//
//  CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2 (
//		cublasHandle_t handle, 
//      cublasOperation_t transa,
//      cublasOperation_t transb, 
//      int m,		// number of rows in matrices A and C
//      int n,		// number of columns in matrices B and C
//      int k,		// number of columns in A and number of rows in B
//      const float *alpha, /* host or device pointer */  
//      const float *A, 
//      int lda,
//      const float *B,
//      int ldb, 
//      const float *beta, /* host or device pointer */  
//      float *C,
//      int ldc);
//
int cublasSGEMM (float* C, float* A, float* B, int HA, int WA, int WB)
{
	cublasStatus_t status;
    float *d_A = 0;
    float *d_B = 0;
    float *d_C = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    float error_norm;
    float ref_norm;
    float diff;
    cublasHandle_t handle;
	cudaError_t cudaStatus;

	// Make sure CUDA device 0 is available
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    /* Initialize CUBLAS */
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
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
        printf("!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices (A, B, and C) */
    status = cublasSetVector(HA * WA, sizeof(A[0]), A, 1, d_A, 1);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }
    status = cublasSetVector(WA * WB, sizeof(B[0]), B, 1, d_B, 1);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }
    status = cublasSetVector(HA * WB, sizeof(C[0]), C, 1, d_C, 1);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }

    /* Performs operation using cublas */
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HA, WB, WA, &alpha, d_A, HA, d_B, WA, &beta, d_C, HA);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

    /* Read the result back (to C)*/
    status = cublasGetVector(HA * WB, sizeof(C[0]), d_C, 1, C, 1);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }

    /* Memory clean up */
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

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }

	return 0;
}