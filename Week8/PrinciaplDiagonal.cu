#include <stdio.h>
#include <cuda.h>

__device__ int factorial(int num) {
    int fact = 1;
    for (int i = 1; i <= num; i++) {
        fact *= i;
    }
    return fact;
}

__device__ int sumOfDigits(int num) {
    int sum = 0;
    while (num != 0) {
        sum += num % 10;
        num /= 10;
    }
    return sum;
}

__global__ void modifyMatrix(int *A, int N) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row == col) {
        A[row * N + col] = 0; 
    } else if (col > row) {
        A[row * N + col] = factorial(A[row * N + col]);  
    } else {
        A[row * N + col] = sumOfDigits(A[row * N + col]); 
    }
}

int main() {
    int N;

    printf("Enter the size of the matrix (N x N): ");
    scanf("%d", &N);

    int *h_A = (int *)malloc(N * N * sizeof(int));

    printf("Enter the elements of the matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("A[%d][%d]: ", i, j);
            scanf("%d", &h_A[i * N + j]);
        }
    }

    int *d_A;
    cudaMalloc((void **)&d_A, N * N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);

    modifyMatrix<<<N, N>>>(d_A, N);

    cudaMemcpy(h_A, d_A, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Modified matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_A[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    free(h_A);

    return 0;
}
