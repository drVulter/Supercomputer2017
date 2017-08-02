#include <stdio.h>
#include <stdlib.h>
//#include<cuda.h>
//#include<cudaruntime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
//#include "kernel.h"

//#define N 4294967296 // Number of calculations.
#define TPB 64 // GPU threads per block
//#define TDIM 32 // number of threads per block along x and y axes
//#define TPB 512 // Maximum threads allowed per block
//#define NUMBLOCKS 16384
#define NUMBLOCKS 512
// Kernel function to initialize the random states
/*
__global__ void init(unsigned int seed, curandState_t* states) {
	//const int i = (blockIdx.x * blockDim.x) + threadIdx.x; // Index variable
	const int i = (blockIdx.x * blockDim.x) + threadIdx.x; // Index variable
	// Initialize an array of states
	curand_init(seed, i, 0, &states[i]);
}
*/
__device__ long counter;
// Kernel function - Actually does the point generation and
// updates circle count
__global__ void pi_kernel(int seed, long* dev_counter) {
	const int i = (blockIdx.x * blockDim.x) + threadIdx.x; // Index variable
	curandState_t state_one;
	curandState_t state_two;
	curand_init(seed, i, 0, &state_one);
	curand_init(seed, i + 1, 0, &state_two);
	float x = curand_uniform(state);
	float y = curand_uniform(state);
	//float test = curand(&states[i]) / 100000000000.0;
	/*if (x * 1000.0 > 1) {
		dev_circle[i] = 1;
	} else {
		dev_circle[i] = 0;
	}*/
	//dev_circle[i] = 3.0;

	if (((x*x) + (y*y)) <= 1.0) {
		//dev_circle[i] = 1; // in the circle
		int current_val = atomicAdd(&counter, 1);
		dev_counter += counter;
	} else {
		//dev_circle[i] = 0; // not in the circle
	}
	//dev_circle[i] = 12;



}

/*
	Called from pi_main
	takes the rank of the process to create distinct seeds
	also takes the local_N value
*/
extern "C" long get_circles(int rank, long local_N) {
	int i; // Counter
	long circle_count = 0; // number of points in the circle
	//double local_pi; // pi approximation for each device
	//double pi_sum; // used by MASTER
	//double pi_approx; // final result
	//long long dim = (long long)sqrt(float(n)); // How many threads total needed in each direction
	//long long dim = 32768; // Assuming 4 nodes running FIX THIS
	long num_iter = local_N / (NUMBLOCKS * TPB);
	printf("num_iter process %d is %ld\n", rank, num_iter);
	// Create an array of random states for the GPU
	//curandState_t* states;
	//cudaMalloc((void**) &states, local_N * sizeof(curandState_t));
	// Use GPU to initialize the states , use rank in seed generation
	//curandState_t* states;
	//cudaMalloc((void**) &states, local_N * sizeof(curandState_t));
	/*int *in_circle = (int*)malloc(local_N * sizeof(int));
	int *dev_circle;
	cudaMalloc(&dev_circle, local_N * sizeof(int));
	pi_kernel<<<local_N / TPB, TPB>>>(states, dev_circle);
	cudaMemcpy(in_circle, dev_circle, local_N * sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", in_circle[0]);
	cudaFree(states);
	cudaFree(dev_circle);
	free(in_circle);
	*/
	//cudaFree(states);

	//long host_counter = 0;
	for (i = 0; i < num_iter; i++) {
		//printf("Pre state\n");
		//curandState_t* states;
		//cudaMalloc((void**) &states, NUMBLOCKS*TPB * sizeof(curandState_t));
		//init<<<NUMBLOCKS, TPB>>>(time(NULL) + rank + i, states);
		//int *in_circle = (int*)malloc(NUMBLOCKS*TPB * sizeof(int));
		//int *dev_circle;
		//cudaMalloc(&dev_circle, NUMBLOCKS*TPB * sizeof(int));
		long host_counter = 0;
		long * dev_counter;
		cudaMalloc((void**)&dev_counter, sizeof(long));
		cudaMemcpy(dev_counter, host_counter, sizeof(long), cudaMemcpyHostToDevice);
		pi_kernel<<<NUMBLOCKS, TPB>>>(time(NULL) + rank, dev_counter);
		cudaMemcpy(host_counter, dev_counter, sizeof(long), cudaMemcpyDeviceToHost);
		printf("%ld\n", host_counter);
		circle_count += host_counter;

		//long * device_count;

		//cudaMemcpy(iter_count, counter, sizeof(long), cudaMemcpyDeviceToHost);
		//printf("%ld\n", iter_count);
		//circle_count += iter_count;
		//printf("Pre memcpy\n");
		//cudaMemcpy(in_circle, dev_circle, NUMBLOCKS*TPB * sizeof(int), cudaMemcpyDeviceToHost);
		/*
		for (j = 0; j < NUMBLOCKS*TPB; j++) {
			//printf("test\n");
			//printf("process %d %d ", rank, in_circle[i]);
			if (in_circle[i] == 1)
				circle_count++;
		}
		*/
		//cudaFree(states);
		//cudaFree(dev_circle);
		//free(in_circle);
	}

	//printf("%d %d\n", rank, rank + time(NULL));
	// Allocate an array to keep track of which points are
	// in the circle
	//float *in_circle = (float*)malloc(local_N * sizeof(float));

	// Allocate an in_circle array for the device
	//float *dev_circle;
	//cudaMalloc(&dev_circle, local_N * sizeof(float));

	// Launch the kernel
	/*for (i = 0; i < num_iter; i++) {
		pi_kernel<<<NUMBLOCKS, TPB>>>(states, dev_circle);
	}*/


	// Copy results back to the host
	//cudaMemcpy(in_circle, dev_circle, local_N * sizeof(int), cudaMemcpyDeviceToHost);
	/*for (i = 0; i < local_N; i++) {
		printf("%f ", in_circle[i]);
	}*/
	//printf("\n");
	// increase the circle_count for every point that was in the circle
	/*for (i = 0; i < local_N; i++) {
		if (in_circle[i] == 1)
			circle_count++;
	}*/
	printf("Circle count from GPU process %d is %ld\n", rank, circle_count);
	// clean up memory
	//cudaFree(states);
	//cudaFree(dev_circle);
	//ree(in_circle);

	// Return the circle count
	return(circle_count);

}
