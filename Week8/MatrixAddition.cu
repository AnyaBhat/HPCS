#include <stdio.h>
#include <cuda.h>

#define M 3 
#define N 3

__global__ void rowAdd(int *A, int *B, int *C, int m, int n) {
    int row = threadIdx.x;
    if(row < m) {
        for (int j = 0; j < n; j++) {
            C[row * n + j] = A[row * n + j] + B[row * n + j];
        }
    }
}

__global__ void colAdd(int *A, int *B, int *C, int m, int n) {
    int col = threadIdx.x;
    if (col < n) {
        for (int i = 0; i < m; i++) {
            C[i * n + col] = A[i * n + col] + B[i * n + col];
        }
    }
}

__global__ void elemAdd(int *A, int *B, int *C, int m, int n) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if(row < m && col < n) {
        C[row * n + col] = A[row * n + col] + B[row * n + col];
    }
}

void printMatrix(int *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int A[M*N], B[M*N], C[M*N];
    int *d_A, *d_B, *d_C;

    printf("Enter matrix A (%dx%d):\n", M, N);
    for (int i = 0; i < M * N; i++) {
        scanf("%d", &A[i]);
    }

    printf("Enter matrix B (%dx%d):\n", M, N);
    for (int i = 0; i < M * N; i++) {
        scanf("%d", &B[i]);
    }

    cudaMalloc((void **)&d_A, M * N * sizeof(int));
    cudaMalloc((void **)&d_B, M * N * sizeof(int));
    cudaMalloc((void **)&d_C, M * N * sizeof(int));

    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * N * sizeof(int), cudaMemcpyHostToDevice);

    printf("Row-wise Addition Result:\n");
    rowAdd<<<1, M>>>(d_A, d_B, d_C, M, N);
    cudaMemcpy(C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    printMatrix(C, M, N);

    printf("Column-wise Addition Result:\n");
    colAdd<<<1, N>>>(d_A, d_B, d_C, M, N);
    cudaMemcpy(C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    printMatrix(C, M, N);

    printf("Element-wise Addition Result:\n");
    elemAdd<<<M, N>>>(d_A, d_B, d_C, M, N);
    cudaMemcpy(C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    printMatrix(C, M, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
