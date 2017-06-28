/*
Parallel implementation of Simpson's rule in CUDA
(annoyed grunt)
*/

#include<stdio.h>

__device__
float myFunc(float x)
{
	return x*x;
}
// kernel Function
__global__
void kernelFunc(float a, float width, int N, float *valOut) {
	int i = threadIdx.x + 1; // Begin at zero so 
	float x = a + ((float)i * width);
	if (i < N) {
		if (i % 2 == 1) {
			valOut[i] = 4 * myFunc(x);
		} else {
			valOut[i] = 2 * myFunc(x);
		}
	}
}
int main() {

 	float a, b; // Endpoints
	int N = 1000; // Number of subintervals
	int i; // counter
	float sum; // Holds sum
	float width; // width of subintervals
	float result = 0; // final result

	// array to carry values out of kernel
	float *valOut = 0;
	// Allocate device memory for output array
	cudaMalloc(&valOut, N*sizeof(float));

	// Array for host
	float hostArray* = (float*)calloc(N, sizeof(float));

	a = 0.0;
	b = 100.0;
	width = (b - a) / N;

	kernelFunc<<<1, 256>>>(a, width, N, valOut);

	cudaMemcpy(hostArray, valOut, N*sizeof(float), cudaMemcpyDeviceToHost);

	for (i = 0; i < N; i++) {
		sum += hostArray[i];
	}

	result = (sum * width) / 3.0;

	printf("The value is %f\n", result);

	// Clean up
	free(hostArray);
	cudaFree(valOut);

	return 0;

}
