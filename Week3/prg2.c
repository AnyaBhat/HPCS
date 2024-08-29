#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, M, m;
    int sum = 0, avg = 0, total_avg = 0;
    int *array = NULL;
    int *subarr = NULL;
    int *arr = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Enter the number of elements (m) : ");
        scanf("%d", &m);
        M = m * size;
        array = (int*)malloc(M * sizeof(int));
        if (array == NULL) {
            perror("Unable to allocate memory");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Enter the elements of the array:");
        for (int i = 0; i < M; i++) {
            scanf("%d", &array[i]);
        }
    }

    subarr = (int*)malloc((M / size) * sizeof(int));
    if (subarr == NULL) {
        perror("Unable to allocate memory");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Scatter the array to all processes
    MPI_Scatter(array, M / size, MPI_INT, subarr, M / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute the sum of the subarray
    for (int i = 0; i < M / size; i++) {
        sum += subarr[i];
    }

    // Compute the average for this process
    avg = (sum * size) / M;

    // Gather all averages at root
    if (rank == 0) {
        arr = (int*)malloc(size * sizeof(int));
        if (arr == NULL) {
            perror("Unable to allocate memory");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(&avg, 1, MPI_INT, arr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate total average on rank 0
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            total_avg += arr[i];
        }
        printf("Total of the averages sent to rank %d : %d\n", rank, total_avg);

        // Free allocated memory
        free(array);
        free(arr);
    }

    free(subarr);
    MPI_Finalize();
    return 0;
}
