/*
From the book
CUDA for Engineers
*/

#include <stdio.h>
#define N 64
#define TPB 32

__device__ float scale(int i, int n)
{
	return ((float)i)/(n - 1);
}

__device__ float distance(float x1, float x2)
{
	return sqrt((x2 - x1)*(x2 - x1));
}

__global__  distanceKernel(float *d_out, int *indexOut, float ref, int len)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	const float x = scale(i, len); // call to scale() function
	d_out[i] = distance(x, ref);
	indexOut[i] = i;
	//printf("i = %2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i]);
}

int main()
{
	int i; // Counter
	const float ref = 0.5f;
	// arrays for host
	float hostArray*;
	hostArray = (float*)malloc(N*sizeof(float));
	int indexArray*;
	indexArray = (float*)malloc(N*sizeof(int));
	// Declare a pointer for an array of floats
	float *d_out = 0;
	// decalre a pointer for an array of ints to store indices for GPU threads
	int *index_out = 0;
	// Allocate device memory to store the output array
	cudaMalloc(&d_out, N*sizeof(float));
	cudaMalloc(&indexOut, N*sizeof(int));
	// Launch kernel to compute and store distance values
	distanceKernel<<<N/TPB, TPB>>>(d_out, indexOut, ref, N);

	// Copy d_out array from kernel to the host (hostArray)
	cudaMemcpy(hostArray, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
	// copy indexOut array
	cudaMemcpy(indexArray, indexOut, N*sizeof(int), cudaMemcpyDeviceToHost);
	for (i = 0; i < N; i++) {
		printf("i = %d, distance is %f", indexArray[i], hostArray[i]);
	}
	cudaFree(d_out); // Free the memory
	cudaFree(indexOut);
	free(indexArray);
	free(hostArray);
	return 0;
}
