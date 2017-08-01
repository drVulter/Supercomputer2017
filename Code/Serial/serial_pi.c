/*
	A program to approximate PI using the Monte Carlo method
	from Lawrence Livermore tutorial
*/

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define N 524288
//#define N 4294967296 // number of points 2^32
int main(void) {
	long i; // Counter
	int circle_count = 0; // number of circles
	//long N = 100000000;
	// points
	float x;
	float y;

	double pi_approx; // approximate value of PI

	srand(time(NULL)); // Set seed value



	int t1 = clock(); // initial time

	for (i = 0; i < N; i++) {
		x = (float)rand() / (float)RAND_MAX; // generate random number 0 - 1
		y = (float)rand() / (float)RAND_MAX;
		//printf("%f, %f\n", x, y);
		if (((x*x) + (y*y)) <= 1.0)
			circle_count++;
	}
	pi_approx = 4.0 * (double)circle_count / (double)N;

	int t2 = clock();

	printf("PI is approximately %f\n", pi_approx);

	double diff = (float)t2 - (float)t1;
	printf("Elapsed time: %f\n", diff);

	return 0;


}
