#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_SIZE 9 // 3x3 matrix
#define ROWS 3
#define COLS 3

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[MATRIX_SIZE];
    int search_value, local_count = 0, global_count = 0;
    int local_matrix[MATRIX_SIZE / 3];
    int counts[3]; // Array to hold the counts of occurrences for each process

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0) {
            printf("This program requires exactly 3 processes.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Enter the 3x3 matrix (9 elements):\n");
        for (int i = 0; i < MATRIX_SIZE; i++) {
            scanf("%d", &matrix[i]);
        }

        printf("Enter the element to be searched:\n");
        scanf("%d", &search_value);
    }

    // Broadcast the search value to all processes
    MPI_Bcast(&search_value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the matrix data to all processes
    MPI_Scatter(matrix, MATRIX_SIZE / size, MPI_INT, local_matrix, MATRIX_SIZE / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Count the occurrences of the search value in the local matrix
    for (int i = 0; i < MATRIX_SIZE / size; i++) {
        if (local_matrix[i] == search_value) {
            local_count++;
        }
    }

    // Use MPI_Scan to compute the prefix sum of local counts
    MPI_Scan(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Use MPI_Reduce to gather the total count from all processes
    MPI_Reduce(&local_count, counts, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Number of occurrences of %d in the matrix: %d\n", search_value, counts[0]);
    }

    MPI_Finalize();
    return 0;
}

/*
Enter the 3x3 matrix (9 elements):
1 2 3 4 5 3 7 8 3
Enter the element to be searched:
3
Number of occurrences of 3 in the matrix: 3
*/
