/*
	Quinn Stratton
	Main function for CUDA/MPI pi approximation program.
	Handles MPI environment and final calculations.
	Calls GPU function get_circles(), found in pi_kernel.cu
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#define MASTER 0 // master process
#define N 536870912 // 2^29
int main(int argc, char *argv[]) {
	int i; // Counter
	// Timing variables
	double start, end;
	// Set up MPI --------------------------------
	MPI_Init(NULL, NULL);
	int rank; // process rank
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Start the stopwatch ------------------------
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	// --------------------------------------------
	int rc; // return code
	// --------------------------------------------
	long circle_count; // number of points in the circle
	double local_pi; // pi approximation for each device
	double pi_sum; // used by MASTER
	double pi_approx; // final result
	long long local_N; // Number of iterations for THIS process
	local_N = N / world_size;
	circle_count = get_circles(rank, local_N);
	// Calculate local value of PI
	local_pi = 4.0 * (double)circle_count / (double)local_N;

	// sum the local values and store them in the pi_sum variable.
	rc = MPI_Reduce(&local_pi, &pi_sum, 1, MPI_DOUBLE,
		MPI_SUM, MASTER, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	end = MPI_Wtime();

	// Done with MPI
	MPI_Finalize();
	//printf("%d\n", world_size);
	if (rank == MASTER) {
		pi_approx = pi_sum / (double)world_size;
		FILE *f;
		if (world_size == 4) {
			f = fopen("/home/ubuntu/data/MPICUDAPIResultsFour.txt", "a"); // Create a file to be written to
		} else if (world_size == 2) {
			f = fopen("/home/ubuntu/data/MPICUDAPIResultsTwo.txt", "a"); // Create a file to be written to
		} else { // Assume 1 MPI Process
			f = fopen("/home/ubuntu/data/MPICUDAPIResultsOne.txt", "a"); // Create a file to be written to
		}

		if (f == NULL) {
			perror("This error occurred:");
			exit(1);
		}
		double time_used = end - start;
		fprintf(f, "%f %f\n", pi_approx, time_used);
		fclose(f); // Close the file
	}

	return 0;

}
