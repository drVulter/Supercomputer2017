/*
Performs numerical integration via Simpson's rule.
Demonstrates a serial version of the algorithm to be contrasted with a parallel
implementation in CUDA
*/
// Funtion to be integrated

#include<iostream>

double func(double x) {
	return x*x;
}
int main() {
 	double a, b; // Endpoints
	int N; // Number of subintervals
	int i; // counter
	double sum; // Holds sum
	double width; // width of subintervals
	double result; // final result
	a = 0.0;
	b = 1.0;
	N = 100;
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

}
