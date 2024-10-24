#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int factorial = 1, partial_sum = 0, total_sum = 0;
    double start_time, end_time, elapsed_time, max_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    start_time = MPI_Wtime();

    // Calculate the factorial of the process's rank + 1
    for (int i = 1; i <= rank + 1; i++) {
        factorial *= i;
    }

    // Perform a prefix sum using MPI_Scan
    MPI_Scan(&factorial, &partial_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Gather the total sum at the root process
    if (rank == size - 1) {
        total_sum = partial_sum;
    }

    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;

    // Calculate the maximum time taken by any process
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Root process prints the total sum and the timing information
    if (rank == size - 1) {
        printf("Total sum 1! + 2! + ... + %d! = %d\n", size, total_sum);
    }

    if (rank == 0) {
        printf("Maximum time taken by any process: %f seconds\n", max_time);
    }

    printf("Process %d took %f seconds\n", rank, elapsed_time);

    MPI_Finalize();
    return 0;
}
