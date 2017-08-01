/*
	Quinn Stratton
	Program to approximate the value of PI
	using Monte Carlo method
*/
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <stdlib.h>
#define N 4294967296 // Number of total iterations 2^32
#define MASTER 0

// Subroutine for determining whether a randomly generated point is in the circle
int circle(float x, float y) {

	x = (float)rand() / (float)RAND_MAX; // generate random number 0 - 1
	y = (float)rand() / (float)RAND_MAX;
	if (((x*x) + (y*y)) <= 1.0)
		return 1;
	else
		return 0;

}

// Actual PI approximating work
// Makes use of circle() subroutine to determine whether a randomly generated point is
// 	in the circle.

int main(int argc, char** argv)
{
	int num_trials; // Number of trials
	int j; // Counter

	// Timing variables
	double start, end;

	long long local_count = 0; // circle count for this process
	int rc; // return code
	float x, y;
	double local_pi;
	double pi_sum;
	double pi_approx;

	long long local_N; // number of iterations for THIS specific process

	long long i; // counter


	// initialize environment
	MPI_Init(NULL, NULL);

	// Find out rank and size
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Start the stopwatch
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();

	srand(time(NULL) + world_rank); // set seed value

	local_N = N / world_size;

	// perform computations
	for (i = 0; i < local_N; i++) {
		x = (float)rand() / (float)RAND_MAX; // generate random number 0 - 1
		y = (float)rand() / (float)RAND_MAX;
		if (((x*x) + (y*y)) <= 1.0)
			local_count++;
	}

	local_pi = 4.0 * (long double)local_count / (long double)local_N;
	rc = MPI_Reduce(&local_pi, &pi_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

	// End the stopwatch
	MPI_Barrier(MPI_COMM_WORLD);
	end = MPI_Wtime();

	// Done with MPI
	MPI_Finalize();

	// MASTER process calculates value of pi using the global_count
	if (world_rank == MASTER) {
		pi_approx = pi_sum / (float)world_size; // Calculate pi value

		double time_used = end - start;
		printf("Apporoximate value of PI is %f\n", pi_approx);

		FILE *f;
		if (world_size == 1) { // Single Process
			f = fopen("/home/pi/data/MPIPIRawDataSingle.txt", "a"); // Create a file to be written to APPEND
		} else if (world_size == 4) {
			f = fopen("/home/pi/data/MPIPIRawDataFour.txt", "a"); // Create a file to be written to APPEND
		} else if (world_size == 8) {
			f = fopen("/home/pi/data/MPIPIRawDataEight.txt", "a"); // Create a file to be written to APPEND
		} else { // Assume 16 processes
			f = fopen("/home/pi/data/MPIPIRawDataSixteen.txt", "a"); // Create a file to be written to APPEND
		}
		if (f == NULL) {
			printf("Error with file!\n");
			exit(1);
		}
		fprintf(f, "%f %f\n", pi_approx, time_used); // Write outputs to file "f"
		fclose(f); // Close the file.
	}


}
