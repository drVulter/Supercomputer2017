/*
	Quinn Stratton
	Lewis University
	S.U.R.E. 2017
	CUDA component of pi_approx program to run on
	Jetson TK1 cluster.
	General algorithm based on Monte Carlo pi approximation
	algorithm found in Lawrence Livermore National Laboratory
	MPI programming tutorial.
*/

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>


#define TPB 64 // GPU threads per block
#define NUMBLOCKS 512
// Kernel function to initialize the random states
__global__ void init(unsigned int seed, curandState_t* states) {
	//const int i = (blockIdx.x * blockDim.x) + threadIdx.x; // Index variable
	const int i = (blockIdx.x * blockDim.x) + threadIdx.x; // Index variable
	// Initialize an array of states
	curand_init(seed, i, 0, &states[i]);
}
//__device__ long counter;
// Kernel function - Actually does the point generation and
// updates circle count
__global__ void pi_kernel(curandState_t* states, int* dev_circle) {
	const int i = (blockIdx.x * blockDim.x) + threadIdx.x; // Index variable
	float x = curand_uniform(&states[i]);
	float y = curand_uniform(&states[i]);
	if (((x*x) + (y*y)) <= 1.0) {
		dev_circle[i] = 1; // in the circle
		//int current_val = atomicAdd(&counter, 1);
	} else {
		dev_circle[i] = 0; // not in the circle
	}



}

/*
	Called from pi_main
	takes the rank of the process to create distinct seeds
	also takes the local_N value
*/
extern "C" long long get_circles(int rank, long long local_N) {
	int i, j; // Counter
	long long circle_count = 0; // number of points in the circle
	long num_iter = local_N / (NUMBLOCKS * TPB);
	for (i = 0; i < num_iter; i++) {
		curandState_t* states;
		cudaMalloc((void**) &states, NUMBLOCKS*TPB * sizeof(curandState_t));
		init<<<NUMBLOCKS, TPB>>>(time(NULL) + rank + i, states);
		// Allocate memory on the CPU
		int *in_circle = (int*)malloc(NUMBLOCKS*TPB * sizeof(int));
		int *dev_circle;
		// Allocate memory on the device (GPU)
		cudaMalloc(&dev_circle, NUMBLOCKS*TPB * sizeof(int));
		pi_kernel<<<NUMBLOCKS, TPB>>>(states, dev_circle);
		cudaMemcpy(in_circle, dev_circle, NUMBLOCKS*TPB * sizeof(int),
			cudaMemcpyDeviceToHost);
		for (j = 0; j < NUMBLOCKS*TPB; j++) {
			if (in_circle[i] == 1)
				circle_count++;
		}
		cudaFree(states);
		cudaFree(dev_circle);
		free(in_circle);
	}

	// Return the circle count
	return(circle_count);

}
