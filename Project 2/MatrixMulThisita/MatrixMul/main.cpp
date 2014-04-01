#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "CH\chTimer.h"

void matrixMul(float* C, float* A, float* B, unsigned int HA, unsigned int WA, unsigned int WB);

int CUDA_matrixMul(float* C, float* A, float* B, unsigned int HA, unsigned int WA, unsigned int WB);
int CUDA_matrixVecMul(float* C, float* A, float* B, unsigned int HA, unsigned int WA);

void OurKernelTest(float *C, float *Ref_C, float *A, float *B, unsigned int HA, unsigned int WA, unsigned int WB)
{
	chTimerTimestamp start, stop;

	printf ("====================== Our Kernel and/or CPU Test ==========================\n\n");

	for(unsigned int i = 0; i < HA * WB; ++i) C[i] = 0;

	// CPU matrix multiplication timing
	printf("CPU timing .....\n");
	chTimerGetTime (&start);
	matrixMul (Ref_C, A, B, HA, WA, WB);
	chTimerGetTime (&stop);
	printf ("\tCPU time = %fl seconds\n\n", chTimerElapsedTime(&start, & stop));

	for(unsigned int i = 0; i < HA * WB; ++i) C[i] = 0;

	int status;
	// CUDA matrix multiplication timing
	printf("Our kernel timing .....\n");
	chTimerGetTime (&start);
	status = CUDA_matrixMul (C, A, B, HA, WA, WB);
	chTimerGetTime (&stop);
	printf ("\tOur kernel call status = %d\n", status);
	printf ("\tOur kernel time = %fl seconds\n\n", chTimerElapsedTime(&start, & stop));

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
	// Usage info
	if(argc != 2) {
		printf("Usage: %s size", argv[0]);
		return EXIT_FAILURE;
	}

	unsigned int heightA, widthA, heightB, widthB, heightC, widthC;

	// Parse size
	unsigned int size = static_cast<unsigned int>(atoi(argv[1]));
	if(size < 2) {
		printf("Invalid matrix size\n");
		return EXIT_FAILURE;
	}

	// Calculate the sizes out
	heightA = widthA = widthB = size;
	heightB = widthA;
	heightC = heightA;
	widthC = widthB;

	// start some memory
	float *A = NULL, *B = NULL, *C = NULL, *Ref_C = NULL;

	A = (float *)malloc(heightA * widthA * sizeof(float));
	B = (float *)malloc(heightB * widthB * sizeof(float));
	C = (float *)malloc(heightC * widthC * sizeof(float));
	Ref_C = (float *)malloc(heightC * widthC * sizeof(float));

	// Int the mem

	for(unsigned int i = 0; i < heightA * widthA; ++i)
		A[i] = (float)(rand() % 10);
	for(unsigned int i = 0; i < heightB * widthB; ++i)
		B[i] = (float)(rand() % 10);
	for(unsigned int i = 0; i < heightC * widthC; ++i)
		C[i] = Ref_C[i] = 0;

	// Say what we are doing
	printf("%d x %d matrix times %d x %d matrix:\n\n", heightA, widthA, heightB, widthB);

	// Start the tests
	//CUBLASTest (C, Ref_C, A, B, heightA, widthA, widthB, 0);
	OurKernelTest(C, Ref_C, A, B, heightA, widthA, widthB);

    return EXIT_SUCCESS;
}

/*
 * Fast NAIVE impl of CPU matrix multiplication
 */
void matrixMul(float* C, float* A, float* B, unsigned int HA, unsigned int WA, unsigned int WB)
{
	// Pre do all these so we aren't reallocating each time
	float c = 0.0f, axb;
	unsigned int ai, bi, ci;

	// Allocate in the for loop decl for compiler optimizations
    for(unsigned int row = 0; row < HA; ++row)
	{
		// Allocate in the for loop decl for compiler optimizations
		for(unsigned int col = 0; col < WB; ++col)
		{
			// reset c
			c = 0.0f;
			// pre calc here, allows for better CPU branch predictions aka faster pipelining
			ci = col * HA + row;

			// allocate in the for loop decl for compiler optimizations
			for(unsigned int k = 0; k < WA; ++k)
			{
				// pre calc here, allows for better CPU branch predictions aka faster pipelining
				ai = k * HA + row;
				bi = col * WA + k;
				axb = A[ai] * B[bi];

				// append the value
				c += axb;
			}

			// set the c value
			C[ci] = c;
		}
	}
}
