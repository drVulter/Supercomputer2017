/*
Performs numerical integration via Simpson's rule.
Demonstrates a serial version of the algorithm to be contrasted with a parallel
implementation in CUDA
*/

#include<stdio.h>
#include <time.h>
#define _USE_MATH_DEFINES // Access to PI
#include<math.h>
double func(double x) {
	return cos(x);
}
int main() {
	clock_t t1, t2;

 	double a, b; // Endpoints
	int N; // Number of subintervals
	int i; // counter
	double sum; // Holds sum
	double width; // width of subintervals
	double result; // final result

	t1 = clock(); // initial time

	a = 0.0;
	b = 2*M_PI;
	N = 1000000;
	width = (b - a) / N;
	sum = func(a) + func(b);

	for (i = 1; i < N; i++) {
		if (i % 2 == 1) {
			sum += 4 * func(a + (i * width));
		} else {
			sum += 2 * func(a + (i * width));
		}
	}
	result = (sum * width) / 3.0;

	printf("The value is %f\n", result);

	t2 = clock();
	float difference = ((float)t2 - (float)t1);
	printf("Running time is %f\n", difference);

}
