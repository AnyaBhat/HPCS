#include <stdio.h>
#include <cuda_runtime.h>

__device__ int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

__device__ int sum_of_digits(int n) {
    int sum = 0;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

__global__ void modify_matrix(int *A, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        for (int col = 0; col < n; col++) {
            if (row == col) {
                A[row * n + col] = 0; // Set diagonal elements to zero
            } else if (row < col) {
                A[row * n + col] = factorial(row + 1); // Above diagonal
            } else {
                A[row * n + col] = sum_of_digits(A[row * n + col]); // Below diagonal
            }
        }
    }
}

void print_matrix(int *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", A[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    int N; // Size of the matrix
    printf("Enter the size of the matrix N: ");
    scanf("%d", &N);
    
    size_t sizeA = N * N * sizeof(int);
    
    // Allocate host memory
    int *h_A = (int*)malloc(sizeA);
    
    // Initialize host matrix with some values (for demonstration)
    
    for (int i = 0; i < N * N; i++) {
        h_A[i] = i + 1; // Values from 1 to N*N
    }
    
    printf("Matrix A:\n");
    print_matrix(h_A,N);
    
    
    // Device pointer
    int *d_A;
    cudaMalloc(&d_A, sizeA);

    // Copy matrix from host to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    
    // Kernel launch parameters
    dim3 dimBlock(256);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);
    
    // Launch the kernel
    modify_matrix<<<dimGrid, dimBlock>>>(d_A, N);
    
    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(h_A, d_A, sizeA, cudaMemcpyDeviceToHost);
    
    // Print the modified matrix
    printf("\nModified Matrix:\n");
    print_matrix(h_A, N);

    // Clean up
    cudaFree(d_A);
    free(h_A);
    
    return 0;
}

/*
nter the size of the matrix N: 5
Matrix A:
1 2 3 4 5 
6 7 8 9 10 
11 12 13 14 15 
16 17 18 19 20 
21 22 23 24 25 

Modified Matrix:
0 1 1 1 1 
6 0 2 2 2 
2 3 0 6 6 
7 8 9 0 24 
3 4 5 6 0 
*/
