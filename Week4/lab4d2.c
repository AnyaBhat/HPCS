#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) {
    int rank, size;
    char input_word[100];
    char local_char;
    char output_word[400];
    int prefix_sum_length;
    double start_time, end_time, elapsed_time, total_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the word: ");
        scanf("%s", input_word);

        if (strlen(input_word) != size) {
            printf("The word length must match the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

 
    start_time = MPI_Wtime();

  
    MPI_Bcast(input_word, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

    
    local_char = input_word[rank];

    
    MPI_Scan(&rank, &prefix_sum_length, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    prefix_sum_length += 1; 

   
    for (int i = 0; i < prefix_sum_length; i++) {
        output_word[prefix_sum_length - 1 - i] = local_char;
    }

   
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;

    
    MPI_Reduce(&elapsed_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    
    if (rank == 0) {
        
        printf("Output word: ");
        for (int i = 0; i < size; i++) {
            for (int j = 0; j <= i; j++) {
                printf("%c", input_word[i]);
            }
        }
        printf("\n");

       
        printf("Total execution time of the whole program: %f seconds\n", total_time);
    }

   
    printf("Process %d took %f seconds\n", rank, elapsed_time);

    MPI_Finalize();
    return 0;
}
