#include <stdio.h>

#define N 4  // Dimensions of the matrices

__global__ void addElement(float *A, float *B, float *C, int rows, int cols) {
    int row = blockIdx.x;  // Row index
    int col = threadIdx.x; // Column index
    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

int main() {
    int rows = 4, cols = 4;
    float h_A[rows][cols], h_B[rows][cols], h_C[rows][cols];
    float *d_A, *d_B, *d_C;

    // Initialize matrices A and B
    printf("\nMatrix A:\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_A[i][j] = (float)(i + j);
            printf("%.2f ",h_A[i][j]);
        }
        printf("\n");
    }
    printf("\nMatrix B:\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_B[i][j] = (float)(i * j);
            printf("%.2f ",h_B[i][j]);
        }
        printf("\n");
    }

    cudaMalloc((void**)&d_A, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_B, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(float));

    cudaMemcpy(d_A, h_A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(cols);
    dim3 numBlocks(rows);
    addElement<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Resultant Matrix C (Element-wise Addition):\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", h_C[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

