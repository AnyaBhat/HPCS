#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int matrix[3][3];
    int element, count = 0;
    int local_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0) {
            printf("This program requires exactly 3 processes.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    if (rank == 0) {
        // Read the 3x3 matrix and the element to search
        printf("Enter the 3x3 matrix:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
        printf("Enter the element to search: ");
        scanf("%d", &element);
    }

    // Broadcast the element to all processes
    MPI_Bcast(&element, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the matrix rows to different processes
    int local_row[3];
    MPI_Scatter(matrix, 3, MPI_INT, local_row, 3, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process counts the occurrences of the element in its row
    for (int i = 0; i < 3; i++) {
        if (local_row[i] == element) {
            local_count++;
        }
    }

    // Reduce all local counts to get the total count
    MPI_Reduce(&local_count, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The element %d occurs %d times in the matrix.\n", element, count);
    }

    MPI_Finalize();
    return 0;
}
