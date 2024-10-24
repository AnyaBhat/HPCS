#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

__global__ void matrixMulKernel(float *A, float *B, float *C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        float value = 0;
        for (int i = 0; i < N; ++i) {
            value += A[row * N + i] * B[i * P + col];
        }
        C[row * P + col] = value;
    }
}
int main() {
    int M, N, P;

    printf("Enter the number of rows in matrix A (M): ");
    scanf("%d", &M);
    printf("Enter the number of columns in matrix A / rows in matrix B (N): ");
    scanf("%d", &N);
    printf("Enter the number of columns in matrix B (P): ");
    scanf("%d", &P);

    float *h_A = (float*) malloc(M * N * sizeof(float));
    float *h_B = (float*) malloc(N * P * sizeof(float));
    float *h_C = (float*) malloc(M * P * sizeof(float));

    printf("Enter elements of matrix A (%d x %d):\n", M, N);
    for (int i = 0; i < M * N; ++i) {
        scanf("%f", &h_A[i]);
    }

    printf("Enter elements of matrix B (%d x %d):\n", N, P);
    for (int i = 0; i < N * P; ++i) {
        scanf("%f", &h_B[i]);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * P * sizeof(float));
    cudaMalloc((void**)&d_C, M * P * sizeof(float));

    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * P * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((P + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    clock_t totalStart = clock();
    cudaEventRecord(start, 0);

    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, P);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);

    cudaMemcpy(h_C, d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    clock_t totalEnd = clock();
    double totalTime = ((double)(totalEnd - totalStart)) / CLOCKS_PER_SEC;

    printf("Resultant matrix C (%d x %d):\n", M, P);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            printf("%f ", h_C[i * P + j]);
        }
        printf("\n");
    }

    printf("Kernel execution time: %f ms\n", kernelTime);
    printf("Total execution time: %f seconds\n", totalTime);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
