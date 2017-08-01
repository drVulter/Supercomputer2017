/*
	Quinn Stratton
	An inefficient serial algorithm to generate a
	list of the first N prime numbers.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Tests whether a number num is prime by checking for the existence of
prime divisors. First the algorithms checks 2 and 3 and then proceeds to check
numers of the form 6k + 1 <= sqrt(num)
*/
int is_prime(int num) {
	int i = 5; // Counter
	if (num <= 1)
		return 0;
	else if (num <= 3)
		return 1;
	else if ((num % 2 == 0) || (num % 3 == 0))
		return 0;
	else {
		while ((i*i) <= num) {
			if ((num % i == 0) || (num % (i + 2) == 0))
				return 0;
			i += 6;
		}
		return 1;
	}
}

int main(void) {
	int N; // Number of primes desired
	int i, j = 0; // Counters


	// Get the size from user
	printf("How many numbers to check? ");
	scanf("%d", &N);

	// Create an array to store the primes.
	int prime_list[N]; 

	int t1 = clock(); // initial time

	// Generate the list.
	while (j < N) {
		if (is_prime(i) != 0) {
			prime_list[j] = i;
			j++;
		}
		i++;
	}

	int t2 = clock(); // final time

	double diff = (double)t2 - (double)t1;

	// Print the results
	for (i = 0; i < N; i++) {
		printf("%d is prime.\n", prime_list[i]);
	}
	printf("Elapsed time: %f\n", diff);

}
