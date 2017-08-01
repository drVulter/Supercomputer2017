/*
Test program to understand the basics of CUDA programming.
Kernel program doubles an array of input values.
*/

#include <stdio.h>
#include <stdlib.h>

/* Should be 2 blocks (N / TPB) = 64 / 32 */
#define N 64 // Number of hellos
#define TPB 32 // Number of threads per block

// KERNEL, gives back an array containing thread indices
__global__ void doubleKernel(int *d_out)
{
	const int i = (blockIdx.x * blockDim.x) + threadIdx.x; // index variable
	// Update "out" array
	d_out[i] = 2 * d_out[i];
}
int main(void)
{
	int i; // counter

	// Array for input on host
	int *hostArray = (int*)malloc(N * sizeof(int));
	// Array for output on host
	int *hostOut = (int*)malloc(N * sizeof(int));
	for (i = 0; i < N; i++)
	{
		hostArray[i] = i;
	}

	// Declare pointer to device array
	int *deviceArray;
	//int deviceInArray;
	// Allocate device memory.
	cudaMalloc(&deviceArray, N*sizeof(int));

	cudaMemcpy(deviceArray, hostArray, N*sizeof(int), cudaMemcpyHostToDevice);
	// Launch kernel!!!
	// <<<N/TPB blocks, TPB threads per block>>>
	doubleKernel<<<N/TPB, TPB>>>(deviceArray);

	// Copy results from device to host.
	cudaMemcpy(hostOut, deviceArray, N*sizeof(int), cudaMemcpyDeviceToHost);

	// Print results
	for (i = 0; i < N; i++)
	{
		printf("2 * %d = %d\n", i, hostOut[i]);
	}
	// Free memory for device array
	cudaFree(deviceArray);

	// Free memory for host array
	free(hostArray);

	return 0;
}
