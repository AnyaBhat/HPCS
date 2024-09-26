//Using Single Block and Single Thread 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

__global__ void selectionSortKernel(char *d_array, int n) {
    // Each thread executes a single pass of selection sort
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < n; j++) {
            if (d_array[j] < d_array[minIdx]) {
                minIdx = j;
            }
        }

        // Swap the found minimum element with the first element
        if (minIdx != i) {
            char temp = d_array[i];
            d_array[i] = d_array[minIdx];
            d_array[minIdx] = temp;
        }
    }
}

void selectionSort(char *h_array, int n) {
    char *d_array;

    // Allocate device memory
    cudaMalloc((void**)&d_array, n * sizeof(char));
    
    // Copy the string to device
    cudaMemcpy(d_array, h_array, n * sizeof(char), cudaMemcpyHostToDevice);

    // Launch kernel with a single block and a single thread
    selectionSortKernel<<<1, 1>>>(d_array, n);

    // Copy the sorted array back to host
    cudaMemcpy(h_array, d_array, n * sizeof(char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_array);
}

int main() {
    char h_array[256];
    printf("Enter a string to sort: ");
    fgets(h_array, sizeof(h_array), stdin);

    // Remove newline character if present
    size_t len = strlen(h_array);
    if (len > 0 && h_array[len - 1] == '\n') {
        h_array[len - 1] = '\0';
    }

    // Calculate the actual length of the string
    int n = strlen(h_array);

    printf("Original string: %s\n", h_array);

    selectionSort(h_array, n);

    printf("Sorted string: %s\n", h_array);
    
    return 0;
}


/*
$  nvcc -o prg1.out string_sort.cu 
$ ./prg1.out
Enter a string to sort: cudaprogram
Original string: cudaprogram
Sorted string: aacdgmoprru

*/
