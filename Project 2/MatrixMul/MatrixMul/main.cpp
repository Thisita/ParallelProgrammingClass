#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "CH\chTimer.h"

void matrixMul( float* C, float* A, float* B, int HA, int WA, int WB);	// CPU matrix multiplication (row-major)
void matrixMul2( float* C, float* A, float* B, int HA, int WA, int WB);	// CPU matrix multiplication (row-major)
void matrixMul3( float* C, float* A, float* B, int HA, int WA, int WB);	// CPU matrix multiplication (column-major)

int cublasSGEMM (float* C, float* A, float* B, int HA, int WA, int WB);	// CUDA CUBLAS matrix multiplication

int GPU_SGEMM (float* C, float* A, float* B, int HA, int WA, int WB, int version);	// Our CUDA matrix multiplication interface function

int check (float *C_ref, float *C, int n);
void printmatrix (float * C, int w, int h, int row_major);


void CUBLASTest (float *C, float *Ref_C, float *A, float *B, int HA, int WA, int WB, int CPUTest)
{
	chTimerTimestamp start, stop;

	printf ("======================== CUBLAS and/or CPU Test ============================\n\n");

	// CPU matrix multiplication timing
	if (CPUTest)
	{
		printf("CPU timing .....\n");
		chTimerGetTime (&start);
		matrixMul3 (Ref_C, A, B, HA, WA, WB);
		chTimerGetTime (&stop);
		printf ("\tCPU time = %fl seconds\n\n", chTimerElapsedTime (&start, & stop));
	}

	for (int i = 0; i < HA * WB; ++i) C[i] = 0;

	// CUBLAS (CUDA) matrix multiplication timing
	printf("CUBLAS timing .....\n");
	chTimerGetTime (&start);
	int cublasStatus = cublasSGEMM (C, A, B, HA, WA, WB);
	chTimerGetTime (&stop);
	printf ("\tCUBLAS calls status = %d\n", cublasStatus);
	printf ("\tCUBLAS (CUDA) time = %fl seconds\n\n", chTimerElapsedTime (&start, & stop));
	
	if (CPUTest)
	{
		printf ("Compare CUBLAS result with CPU result .....\n");
		check (Ref_C, C, HA * WB);

		if (HA < 10 && WB <10)
		{
			printf("\nCPU MatrixMul3 Ref_C (column major)\n");
			printmatrix(Ref_C, HA, WB, 0);
			printf("CUBLAS C\n");
			printmatrix(C, HA, WB, 0);
		}
		printf("\n");
	}

	printf ("====================== End CUBLAS and/or CPU Test ==========================\n\n\n");
}

void OurKernelTest(float *C, float *Ref_C, float *A, float *B, int HA, int WA, int WB, int CPUTest, int version)
{
	chTimerTimestamp start, stop;

	printf ("====================== Our Kernel and/or CPU Test ==========================\n\n");

	if (CPUTest)
	{
		// CPU matrix multiplication timing
		printf("CPU timing .....\n");
		chTimerGetTime (&start);
		matrixMul (Ref_C, A, B, HA, WA, WB);
		chTimerGetTime (&stop);
		printf ("\tCPU time = %fl seconds\n\n", chTimerElapsedTime (&start, & stop));
	}

	for (int i = 0; i < HA * WB; ++i)
		C[i] = 0;

	int status;
	// Our kernel matrix multiplication timing
	printf("Our kernel timing .....\n");
	chTimerGetTime (&start);
	status = GPU_SGEMM (C, A, B, HA, WA, WB, version);
	chTimerGetTime (&stop);
	printf ("\tOur kernel call status = %d\n", status);
	printf ("\tOur kernel time = %fl seconds\n\n", chTimerElapsedTime (&start, & stop));

	if (CPUTest)
	{
		printf ("Compare our kernel result with CPU result .....\n");
		check (Ref_C, C, HA * WB);

		if (HA < 10 && WB <10)
		{
			printf("\nCPU MatrixMul Ref_C (row-major)\n");
			printmatrix(Ref_C, HA, WB, 1);
			printf("Our kernel C\n");
			printmatrix(C, HA, WB, 1);
		}
		printf("\n");
	}

	printf ("====================== End Our Kernel and/or CPU Test =======================\n\n\n");
}

#if 0
#define heightA 4800
#define widthA 4800
#define heightB widthA
#define widthB 4800
#define heightC heightA
#define widthC widthB
#endif

int main(int argc, char *argv[])
{
	if(argc != 3) {
		printf("Usage: %s version size", argv[0]);
		return EXIT_FAILURE;
	}

	int heightA, widthA, heightB, widthB, heightC, widthC;

	int version = atoi(argv[1]);
	if(version < 1) {
		printf("Invalid version number\n");
		return EXIT_FAILURE;
	}

	int size = atoi(argv[2]);
	if(size < 2) {
		printf("Invalid matrix size\n");
		return EXIT_FAILURE;
	}

	heightA = widthA = widthB = size;
	heightB = widthA;
	heightC = heightA;
	widthC = widthB;

	float *A, *B, *C, *Ref_C;

	A = (float *)malloc (heightA * widthA * sizeof (float));
	B = (float *)malloc (heightB * widthB * sizeof (float));
	C = (float *)malloc (heightC * widthC * sizeof (float));
	Ref_C = (float *)malloc (heightC * widthC * sizeof (float));

	for (int i = 0; i < heightA * widthA; ++i)
		A[i] = (float)(rand()%10);
	for (int i = 0; i < heightB * widthB; ++i)
		B[i] = (float)(rand()%10);
	for (int i = 0; i < heightC * widthC; ++i)
		C[i] = Ref_C[i] = 0;

	printf("%d x %d matrix times %d x %d matrix:\n\n", heightA, widthA, heightB, widthB);

	//CUBLASTest (C, Ref_C, A, B, heightA, widthA, widthB, 0);
	OurKernelTest(C, Ref_C, A, B, heightA, widthA, widthB, 0, 1);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! where A is a HAxWA matrix and B is a WAxWB matrix.
//! Assume matrices are stored in row-major linear array, and matrix indexing is 0-based.
////////////////////////////////////////////////////////////////////////////////

void matrixMul( float* C, float* A, float* B, int HA, int WA, int WB)
{
    // index of the C matrix element 
    int row,col;
	 
    for (row = 0; row < HA; ++row)
	{
		for (col = 0; col < WB; ++col)
		{
			float Cvalue = 0;
			int indexA = row * WA;
			int indexB = col;

			// Assuming row-major like in C (note CUBLAS and Fortran uses column-major)
			for (int k = 0; k < WA; k++)
			{
				Cvalue += A[indexA++] * B[indexB];
				indexB += WB;
			}

			C[row * WB + col] = Cvalue;
		}
	}
}
void matrixMul2( float* C, float* A, float* B, int HA, int WA, int WB)
{
    // index of the C matrix element 
    int row;
	int col;
	
	// Assuming row-major like in C (note CUBLAS and Fortran uses column-major)
    for (row = 0; row < HA; ++row)
	{
		for (col = 0; col < WB; ++col)
		{
			float Cvalue = 0;

			for (int k = 0; k < WA; k++)
			{
				Cvalue += A[row * WA + k] * B[k * WB + col];
			}

			C[row * WB + col] = Cvalue;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! where A is a HAxWA matrix and B is a WAxWB matrix.
//! Assume matrices are stored in column-major linear array, and matrix indexing is 0-based.
////////////////////////////////////////////////////////////////////////////////

void matrixMul3( float* C, float* A, float* B, int HA, int WA, int WB)
{
    // index of the C matrix element 
    int row, col;
	
	// Assuming colimn-major like in CUBLAS and Fortran, but matrix indexing is 0-based
    for (row = 0; row < HA; ++row)
	{
		for (col = 0; col < WB; ++col)
		{
			float Cvalue = 0.0f;

			for (int k = 0; k < WA; k++)
			{
				Cvalue += A[row + HA * k] * B[col * WA + k];
			}

			C[col * HA + row] = Cvalue;  // Note column-major !!
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////
//
// Utilitiy functions
//
///////////////////////////////////////////////////////////////////////////////////////////

double fmax (double a, double b)
{
	return (a >= b? a: b);
}

/* Check result (C) against reference (ref_C */
int check (float *ref_C, float *C, int n)
{ 
    double absolute_error = 0;
	double relative_error = 0;
	double max_relative_error = 0;
    double ref_norm = 0;

    for (int i = 0; i < n; ++i)
    {
		double diff = 0.0;
		double relative_diff = 0.0;

        diff = ref_C[i] - C[i];
		if (diff != 0.0)
			relative_diff = (double)fabs (diff / fmax (ref_C[i], C[i])); 

        max_relative_error = fmax (max_relative_error,  relative_diff);
		relative_error += relative_diff; 
		absolute_error += (diff * diff);
        ref_norm += ref_C[i] * ref_C[i];
    }

    absolute_error = sqrt(absolute_error);
    ref_norm = sqrt(ref_norm);

	printf ("\tTotal absolute error (Euclidean) = %fl\n", absolute_error);
	printf ("\tTotal relative error (Sum)= %fl\n", relative_error);
	printf ("\tMaximum relative error = %fl\n\n", max_relative_error);
    
	if (max_relative_error < 1e-6)
    {
        printf("Test passed (max relative eror %fl is less than 1e-6)\n", max_relative_error);
        return EXIT_SUCCESS;     // return 0;
    }
	else
	{
        printf("!!!! Test failed (Max relative eror %fl is larger than 1e-6)\n", max_relative_error);
        return EXIT_FAILURE;     // return 1;
    }

    if (fabs(ref_norm) < 1e-7)
    {
        fprintf(stderr, "!!!! reference norm is 0\n");
        return EXIT_FAILURE;
    }
}


void printmatrix (float * C, int h, int w, int row_major = 1)
{
	if (row_major)	// row-major 0-based index (like in C)
	{
		for (int row = 0; row < h; row++)
		{
			for (int col = 0; col < w; col++)
				printf ("%8.2f ", C[row * w + col]);

			printf("\n");
		}
	}
	else	// column-major 0-based index (like in Fortran and CUBLAS) 
	{
		for (int row = 0; row < h; row++)
		{
			for (int col = 0; col < w; col++)
				printf ("%8.2f ", C[col * h + row]);

			printf("\n");
		}
	}
}