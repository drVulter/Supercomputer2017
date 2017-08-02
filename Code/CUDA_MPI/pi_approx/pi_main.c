
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
//#include "kernel.h"
#define MASTER 0 // master process
#define N 4294967296 // 2^32
//#define N 2097152
//#define N 1048576 *2*2*2*2*2*2*2
//#define N 268435456 * 4
int main(int argc, char *argv[]) {
	int i; // Counter
	//int trials;

	// Timing variables
	double start, end;



	printf("Before MPI\n");
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
	//int local_circle_count; // circle count for THIS process
	long circle_count; // number of points in the circle
	double local_pi; // pi approximation for each device
	double pi_sum; // used by MASTER
	double pi_approx; // final result
	long long local_N; // Number of iterations for THIS process
	//printf("Process %d before GPU call\n", rank);
	local_N = N / world_size;
	printf("local N process %d is: %lld\n", rank, local_N);
	// Calculate the side length of the GPU grid
	//int local_grid_dim = 256;
	//long long local_grid_dim = (long long)sqrt((long double)local_N);
	// Call to external GPU function, pass in rank to give a good seed value
	//circle_count = 12.0;
	//printf("local circle count for process %d pre kernel call is %ld\n", rank, circle_count);
	circle_count = get_circles(rank, local_N);
	printf("local circle count for process %d is %ld\n", rank, circle_count);
	// Calculate local value of PI
	local_pi = 4.0 * (double)circle_count / (double)local_N;

	printf("Local value of PI for process %d is %f\n", rank, local_pi);

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
		printf("%f %f\n", pi_approx, time_used);
		fprintf(f, "%f %f\n", pi_approx, time_used);
		//printf("Elapsed time: %d\n", diff);
		//printf("Approximate value of PI is %f\n", pi_approx);
	}

	//fclose(f); // Close the file
	return 0;

}
