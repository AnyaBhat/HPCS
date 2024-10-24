%%writefile AddMul.cu

#include <stdio.h>
#include <cuda_runtime.h>

// #define TILE_WIDTH 16  

__global__ void matrixAdd(const float *A, const float *B, float *C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

__global__ void matrixMul(const float *A, const float *B, float *C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        float value = 0.0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = value;
    }
}

int main() {
    int M, N, P;

    printf("Enter number of rows and columns for matrix A: ");
    scanf("%d %d", &M, &N);

    printf("Enter number of rows and columns for matrix B: ");
    scanf("%d %d", &N, &P);

    float *h_A = (float *)malloc(M * N * sizeof(float));
    float *h_B = (float *)malloc(N * P * sizeof(float));
    float *h_C_add = (float *)malloc(M * N * sizeof(float));
    float *h_C_mul = (float *)malloc(M * P * sizeof(float));

    printf("Enter elements for matrix A:\n");
    for (int i = 0; i < M * N; ++i) {
        scanf("%f", &h_A[i]);
    }

    printf("Enter elements for matrix B:\n");
    for (int i = 0; i < N * P; ++i) {
        scanf("%f", &h_B[i]);
    }

    float *d_A, *d_B, *d_C_add, *d_C_mul;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * P * sizeof(float));
    cudaMalloc(&d_C_add, M * N * sizeof(float));
    cudaMalloc(&d_C_mul, M * P * sizeof(float));

    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * P * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    matrixAdd<<<dimBlock, dimGrid>>>(d_A, d_B, d_C_add, M, N);
    cudaDeviceSynchronize();

    matrixMul<<<dimBlock, dimGrid>>>(d_A, d_B, d_C_mul, M, N, P);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_add, d_C_add, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_mul, d_C_mul, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nMatrix C (Addition):\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", h_C_add[i * N + j]);
        }
        printf("\n");
    }

    printf("\nMatrix C (Multiplication):\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            printf("%f ", h_C_mul[i * P + j]);
        }
        printf("\n");
    }

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
