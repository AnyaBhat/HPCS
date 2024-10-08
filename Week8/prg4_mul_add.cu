#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 5 // Size of the matrix (N x N)

__global__ void matrixAdd(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

__global__ void matrixMul(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0;
    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = (float)(rand() % 10);
    }
}

void printMatrix(float *matrix) {
    for (int i = 0; i < N * N; ++i) {
        printf("%f ", matrix[i]);
        if ((i + 1) % N == 0) printf("\n");
    }
}

int main() {
    // Allocate host memory
    size_t size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_add = (float *)malloc(size);
    float *h_C_mul = (float *)malloc(size);

    // Initialize matrices
    initializeMatrix(h_A, N * N);
    initializeMatrix(h_B, N * N);

    // Allocate device memory
    float *d_A, *d_B, *d_C_add, *d_C_mul;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C_add, size);
    cudaMalloc((void**)&d_C_mul, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(4, 4);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch matrix addition kernel
    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C_add);
    cudaDeviceSynchronize();

    // Launch matrix multiplication kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C_mul);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_C_add, d_C_add, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_mul, d_C_mul, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Matrix A:\n");
    printMatrix(h_A);
    printf("Matrix B:\n");
    printMatrix(h_B);
    printf("Matrix Addition Result:\n");
    printMatrix(h_C_add);
    printf("Matrix Multiplication Result:\n");
    printMatrix(h_C_mul);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_add);
    free(h_C_mul);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_add);
    cudaFree(d_C_mul);

    return 0;
}

/*
Matrix A:
3.000000 6.000000 7.000000 5.000000 3.000000 
5.000000 6.000000 2.000000 9.000000 1.000000 
2.000000 7.000000 0.000000 9.000000 3.000000 
6.000000 0.000000 6.000000 2.000000 6.000000 
1.000000 8.000000 7.000000 9.000000 2.000000 
Matrix B:
0.000000 2.000000 3.000000 7.000000 5.000000 
9.000000 2.000000 2.000000 8.000000 9.000000 
7.000000 3.000000 6.000000 1.000000 2.000000 
9.000000 3.000000 1.000000 9.000000 4.000000 
7.000000 8.000000 4.000000 5.000000 0.000000 
Matrix Addition Result:
3.000000 8.000000 10.000000 12.000000 8.000000 
14.000000 8.000000 4.000000 17.000000 10.000000 
9.000000 10.000000 6.000000 10.000000 5.000000 
15.000000 3.000000 7.000000 11.000000 10.000000 
8.000000 16.000000 11.000000 14.000000 2.000000 
Matrix Multiplication Result:
169.000000 78.000000 80.000000 136.000000 103.000000 
156.000000 63.000000 52.000000 171.000000 119.000000 
165.000000 69.000000 41.000000 166.000000 109.000000 
102.000000 84.000000 80.000000 96.000000 50.000000 
216.000000 82.000000 78.000000 169.000000 127.000000 


*/
