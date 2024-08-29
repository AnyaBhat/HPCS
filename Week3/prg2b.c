//Might be wrong
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

float calculate_average(float *data, int count) {
    float sum = 0.0;
    for (int i = 0; i < count; i++) {
        sum += data[i];
    }
    return sum / count;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int M, N;
    float *data = NULL;
    float *sub_data = NULL;
    float local_avg, global_avg;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process
        N= size;
        printf("Enter the value for M: ");
        scanf("%d", &M);

        // Allocate memory for the data
        data = (float *)malloc(N * M * sizeof(float));
        
        // Read data
        printf("Enter %d x %d elements:\n", N, M);
        for (int i = 0; i < N * M; i++) {
            scanf("%f", &data[i]);
        }
    }

    // Allocate memory for the sub_data (chunk of data each process will receive)
    sub_data = (float *)malloc(M * sizeof(float));

    // Scatter the data to all processes
    MPI_Scatter(data, M, MPI_FLOAT, sub_data, M, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Compute the average of the received chunk
    local_avg=calculate_average(sub_data, M);

    // Gather all the local averages at the root process
    float *all_avgs = NULL;
    if (rank == 0) {
        all_avgs = (float *)malloc(size * sizeof(float));
    }

    MPI_Gather(&local_avg, 1, MPI_FLOAT, all_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Calculate the final average of all gathered averages
        float sum_of_avgs = 0.0;
        for (int i = 0; i < size; i++) {
            sum_of_avgs += all_avgs[i];
        }
        //global_avg = sum_of_avgs / size;
        printf("The final average of averages is: %f\n", sum_of_avgs);

        // Free allocated memory
        free(data);
        free(all_avgs);
    }

    // Free memory allocated for each process
    free(sub_data);

    MPI_Finalize();
    return 0;
}
