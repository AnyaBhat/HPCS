#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size;
    int N; // Size of the array
    int *array = NULL;
    int *sub_array = NULL;
    int search_number;
    int found = 0; // Flag to indicate if the number was found
    int global_found = 0; // Flag to indicate if the number was found by any process
    
    MPI_Init(&argc, &argv);
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  
    if (rank == 0) {
        // Root process initializes the array
        
    	printf("Enter the Number of elements in the array :");
   	scanf("%d",&N);
        printf("Enter the array elements:");
        array = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) {
            scanf("%d",&array[i]); // Initialize array with values 0, 1, 2, ..., N-1
        }
        
        printf("Enter the Number to be searched in the array :");
   	scanf("%d",&search_number);
   	
	for (int i = 1; i < size; i++) {
            MPI_Send(&N, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        // Send the search number to all processes
        for (int i = 1; i < size; i++) {
            MPI_Send(&search_number, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        // Send portions of the array to each process
        int local_size = N / size;
        for (int i = 1; i < size; i++) {
            MPI_Send(array + i * local_size, local_size, MPI_INT, i, 1, MPI_COMM_WORLD);
        }

        // Root process handles its own portion
        sub_array = array; // Root process works with the full array
        local_size = N / size;
        for (int i = 0; i < local_size; i++) {
            if (sub_array[i] == search_number) {
                found = 1;
                break;
            }
        }

        // Collect results from other processes
        global_found = found; // Start with the result from the root process
        for (int i = 1; i < size; i++) {
            int recv_found;
            MPI_Recv(&recv_found, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            global_found = global_found || recv_found;
        }

        // Print the result on the root process
        if (global_found) {
            printf("Number %d found in the array.\n", search_number);
        } else {
            printf("Number %d not found in the array.\n", search_number);
        }

        // Free the allocated memory
        free(array);

    } else {
        // Worker processes

        // Receive the search number
         MPI_Recv(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        MPI_Recv(&search_number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        // Determine the size of the local portion of the array
        int local_size = N / size;
        sub_array = (int*)malloc(local_size * sizeof(int));

        // Receive the portion of the array
        MPI_Recv(sub_array, local_size, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

        // Search for the number in the local portion of the array
        for (int i = 0; i < local_size; i++) {
            if (sub_array[i] == search_number) {
                found = 1;
                break;
            }
        }

        // Send the result back to the root process
        MPI_Send(&found, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);

        // Free the allocated memory
        free(sub_array);
    }

    MPI_Finalize();
    return 0;
}
