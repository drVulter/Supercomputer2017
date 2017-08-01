/*
Program to determine which elements in a given list are prime, and then copy a
list of said primes back onto the CPU.
Demonstrates the use of a function that lives on the GPU
*/
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

//#include<cuda_runtime.h>

// Global variables
#define N 65536 // Size of array 2^(2^(2^(2)))
#define TPB 8192 // Threads per block, should be 8 blocks
__device__
bool isPrime(int num)
{
	int i; // counter
	int stop = (int)ceil(sqrt((float)num));
	for (i = 0; i < stop; i++)
	{
		if (num % i == 0)
			return false;
	}
	return true;
}
// KERNEL
__global__
void primeKernel(*d_in)
{
	int i; // thread
	d_out[i] = isPrime(d_in[i]);
}

// MAIN
int main(void)
{
	int i; // Counter
	// Create an array to be copied onto the GPU
	int *hostIn = (int*)malloc(N*sizeof(int));

	// Create an array for output from GPU
	bool *hostOut = (bool*)malloc(N*sizeof(bool));

	// Create arrays for the device
	int *d_in; // Input
	cudaMalloc(&d_in, N*sizeof(int));
	bool *d_out; // Output, bigger than necessary
	cudaMalloc(&d_out, N*sizeof(bool));

	// Initialize hostIn {1,2,...,N}
	for (i = 0; i < N; i++)
	{
		hostIn[i] = i+1;
	}
	// Copy the hostArray to the GPU's memory
	cudaMemcpy(d_in, hostIn, N*sizeof(int), cudaMemcpyHostToDevice);

	// Call the Kernel function
	primeKernel<<<N/TPB, TPB>>>(d_in);

	// Copy the device out array back onto the hostArray, writing over the previous array
	cudaMemcpy(hostOut, d_out, N*sizeof(bool), cudaMemcpyDeviceToHost);

	// Print results
	for (i = 0; i < N; i++)
	{
		if (hostOut[i])
			printf("%d is prime\n", i+1);
	}

	// Clean up
	free(hostIn);
	free(hostOut);
	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
