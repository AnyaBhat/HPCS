#include <mpi.h>
#include <stdio.h>

#define N 4 // Size of the matrix

void print_matrix(int matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != N) {
        if (rank == 0) {
            fprintf(stderr, "This program requires exactly %d processes.\n", N);
        }
        MPI_Finalize();
        return 1;
    }
    
    int matrix[N][N];
    int local_row[N];
    int result_matrix[N][N];
    
    if (rank == 0) {
        printf("Enter %d numbers (4x4 matrix):\n", N * N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
    }
    
    // Broadcast the matrix to all processes
    MPI_Bcast(matrix, N * N, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Scatter rows of the matrix to all processes
    MPI_Scatter(matrix, N, MPI_INT, local_row, N, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Compute prefix sum for each row
    int local_prefix_sum[N];
    MPI_Scan(local_row, local_prefix_sum, N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // Gather the results back to the root process
    MPI_Gather(local_prefix_sum, N, MPI_INT, result_matrix, N, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Print the resulting matrix on rank 0
    if (rank == 0) {
        printf("Resulting Matrix:\n");
        print_matrix(result_matrix);
    }
    
    MPI_Finalize();
    return 0;
}
