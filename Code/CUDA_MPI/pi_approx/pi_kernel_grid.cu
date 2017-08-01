#include <stdio.h>
#include <stdlib.h>
//#include<cuda.h>
//#include<cudaruntime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
//#include "kernel.h"

//#define N 4294967296 // Number of calculations.
//#define TPB 256 // GPU threads per block
#define TDIM 32 // number of threads per block along x and y axes
//#define TPB 512 // Maximum threads allowed per block
// Kernel function to initialize the random states
__global__ void init(unsigned int seed, curandState_t* states, long long dim) {
	//const int i = (blockIdx.x * blockDim.x) + threadIdx.x; // Index variable
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = r*dim + c;
	// Initialize an array of states
	curand_init(seed, i, 0, &states[i]);
}

// Kernel function - Actually does the point generation and
// updates circle count
__global__ void pi_kernel(curandState_t* states, int* dev_circle, int dim) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = r*dim + c;
	float x = curand_uniform(&states[i]);
	float y = curand_uniform(&states[i]);
	if (((x*x) + (y*y)) <= 1.0)
		dev_circle[i] = 1; // in the circle
	else
		dev_circle[i] = 0; // not in the circle
}

/*
	Called from pi_main
	takes the rank of the process to create distinct seeds
	also takes the local_N value
*/
extern "C" long get_circles(int rank, int dim) {
	long i; // Counter
	long circle_count = 0; // number of points in the circle
	//double local_pi; // pi approximation for each device
	//double pi_sum; // used by MASTER
	//double pi_approx; // final result
	//long long dim = (long long)sqrt(float(n)); // How many threads total needed in each direction
	//long long dim = 32768; // Assuming 4 nodes running FIX THIS
	const dim3 blockSize(TDIM, TDIM);
	const int bx = (dim + blockSize.x - 1) / blockSize.x;
	const int by = (dim + blockSize.y - 1) / blockSize.y;
	const dim3 gridSize = dim3(bx, by);
	// Create an array of random states for the GPU
	curandState_t* states;
	cudaMalloc((void**) &states, dim * dim * sizeof(curandState_t));
	// Use GPU to initialize the states , use rank in seed generation
	init<<<gridSize, blockSize>>>(time(NULL) + rank, states, dim);
	//printf("%d %d\n", rank, rank + time(NULL));
	// Allocate an array to keep track of which points are
	// in the circle
	int *in_circle = (int*)malloc(dim * dim * sizeof(int));

	// Allocate an in_circle array for the device
	int *dev_circle;
	cudaMalloc(&dev_circle, dim * dim * sizeof(int));

	// Launch the kernel
	pi_kernel<<<gridSize, blockSize>>>(states, dev_circle, dim);

	// Copy results back to the host
	cudaMemcpy(in_circle, dev_circle, dim * dim * sizeof(int), cudaMemcpyDeviceToHost);

	// increase the circle_count for every point that was in the circle
	for (i = 0; i < dim * dim; i++) {
		if (in_circle[i] == 1)
			circle_count++;
	}
	printf("Circle count from GPU process %d is %ld\n", rank, circle_count);
	// clean up memory
	cudaFree(states);
	cudaFree(dev_circle);
	free(in_circle);

	// Return the circle count

	return(circle_count);



}
