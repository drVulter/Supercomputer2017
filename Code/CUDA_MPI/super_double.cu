/*
	Massively parallel program to double an array of values
	Each "compute" node gets a chunk of the array
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define N 65536 // Total number of elements to be doubled
#define TPB 128 // threads per block - GPU
#define MASTER 0 // process 0 is the "master"
int main(void) {
	int i, j; // counters
	// Set up MPI ----------------------------------
	MPI_Init(NULL, NULL);
	int rank; // process rank
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int world_size; // number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	// ---------------------------------------------

	if (rank == MASTER) { // MASTER process
		// Set up array for host input
		int *host_array = (int*)malloc(N * sizeof(int));
		// Fill array with values to be doubled
		for (i = 0; i < N; i++) {
			host_array[i] = i;
		}
		// Array for output on host (will contain doubled values)
		int *host_out = (int*)malloc(N * sizeof(int));

		// Send chunks of array to compute processes.
		for (i = 0; i < world_size - 1; i++) {
			int *compute_array = (int*)malloc((N / (world_size - 1)) * sizeof(int));
		}
	}
}
