#include <stdio.h>
#include <cuda.h>

// CUDA kernel to perform operations on the matrix
__global__ void processMatrix(int *A, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / N; // Row index
    int j = idx % N; // Column index

    if (idx < N * N) {
        if (i == j) {
            // Principal diagonal element -> replace with 0
            A[i * N + j] = 0;
        }
        else if (i < j) {
            // Above principal diagonal -> replace with factorial
            int val = A[i * N + j];
            int fact = 1;
            for (int k = 1; k <= val; k++) {
                fact *= k;
            }
            A[i * N + j] = fact;
        }
        else if (i > j) {
            // Below principal diagonal -> replace with sum of digits
            int val = A[i * N + j];
            int sumDigits = 0;
            while (val > 0) {
                sumDigits += val % 10;
                val /= 10;
            }
            A[i * N + j] = sumDigits;
        }
    }
}

// Function to print the matrix
void printMatrix(int *A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", A[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    int N;
    
    // Input size of matrix N
    printf("Enter the size of the matrix (N): ");
    scanf("%d", &N);

    // Allocate memory for matrix on host
    int *h_A = (int *)malloc(N * N * sizeof(int));

    // Input matrix elements
    printf("Enter the elements of the matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            scanf("%d", &h_A[i * N + j]);
        }
    }

    // Allocate memory for matrix on device
    int *d_A;
    cudaMalloc((void **)&d_A, N * N * sizeof(int));

    // Copy matrix from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define the number of threads and blocks
    int blockSize = 256; 
    int gridSize = (N * N + blockSize - 1) / blockSize;

    // Launch the kernel
    processMatrix<<<gridSize, blockSize>>>(d_A, N);

    // Copy the result back to host
    cudaMemcpy(h_A, d_A, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the updated matrix
    printf("Processed matrix:\n");
    printMatrix(h_A, N);

    // Free device memory
    cudaFree(d_A);

    // Free host memory
    free(h_A);

    return 0;
}


/*
Enter the size of the matrix (N): 4
Enter the elements of the matrix:
1 2 3 4
5 6 7 8
9 10 11 6
13 14 15 16

Processed matrix:
0   2   6    24
5   0   5040 40320
9   1   0    720
4   5   6    0

*/
