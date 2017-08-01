/*
	* Extension of the pi_approx program utilizing CUDA
	* Each process uses the GPU to create a PI approximation then
		each process passes the count to the MASTER
	* Currently uses an array of boolean-like values to keep track
	 	of count, is there a better way?????

*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 128 // Number of calculations.
#define TPB 64 // GPU threads per block
#define MASTER 0 // master process
// Kernel function to initialize the random states
__global__ void init(unsigned int seed, curandState_t* states) {
	const int i = (blockIdx.x * blockDim.x) + threadIdx.x; // Index variable
	// Initialize an array of states
	curand_init(seed, i, 0, &states[i]);
}

// Kernel function - Actually does the point generation and
// updates circle count
__global__ void pi_kernel(curand_Statet* states, int* dev_circle) {
	const int i = (blockIdx.x * blockDim.x) + threadIdx.x; // Index variable
	float x = curand_uniform(states[i]);
	float y = curand_uniform(states[i]);
	if (((x*x) + (y*y)) <= 1.0)
		dev_circle[i] = 1; // in the circle
	else
		dev_circle[i] = 0; // not in the circle


}

// MAIN
int main(int argc, char** argv) {
	// Set up MPI --------------------------------
	MPI_Init(NULL, NULL);
	int rank; // process rank
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int world_size;
	MPIComm_size(MPI_COMM_WORLD, &world_size);
	int rc; // return code
	// --------------------------------------------
	int i; // Counter
	int circle_count = 0; // number of points in the circle
	double local_pi; // pi approximation for each device
	double pi_sum; // used by MASTER
	double pi_approx; // final result

	// Create an array of random states for the GPU
	curandState_t* states;
	cudaMalloc((void**) &states, N * sizeof(curandState_t));
	// Use GPU to initialize the states , use rank in seed generation
	init<<<N/TPB, TPB>>>(time(NULL) + rank, states);

	// Allocate an array to keep track of which points are
	// in the circle
	int *in_circle = (int*)malloc(N * sizeof(int));

	// Allocate an in_circle array for the device
	int *dev_circle;
	cudaMalloc(&dev_circle, N * sizeof(int));

	// Launch the kernel
	pi_kernel<<<N/TPB, TPB>>>(states, dev_circle);

	// Copy results back to the host
	cudaMemcpy(in_circle, dev_circle, N * sizeof(int), cudeMemcpyDeviceToHost);

	// increase the circle_count for every point that was in the circle
	for (i = 0l i < N; i++) {
		if (in_circle[i] == 1)
			circle_count++;
	}
	local_pi = 4.0 * (double)circle_count / (double)N;
	printf("Local value of PI for process %d is %f\n", rank, local_pi);
	// sum the local values and store them in the pi_sum variable.
	rc = MPI_Reduce(&local_pi, &pi_sum, 1, MPI_DOUBLE,
		MPI_SUM, MASTER, MPI_COMM_WORLD);
	if (world_rank == MASTER) {
		pi_approx = pi_sum / (double)world_size;
		printf("Approximate value of PI is %f\n", pi_approx);
	}

	// Clean up memory
	cudaFree(states);
	cudaFree(dev_circle);
	free(in_circle);
}
