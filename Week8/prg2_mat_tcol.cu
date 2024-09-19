#include <stdio.h>

#define N 4  // Number of rows

__global__ void addColumn(float *A, float *B, float *C, int cols) {
    int col = blockIdx.x;  // Each thread handles one column
    if (col < cols) {
        for (int row = 0; row < N; row++) {
            C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
        }
    }
}

int main() {
    int cols = 4;
    float h_A[N][cols], h_B[N][cols], h_C[N][cols];
    float *d_A, *d_B, *d_C;

    // Initialize matrices A and B
    printf("\nMatrix A:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_A[i][j] = (float)(i + j);
            printf("%.2f ",h_A[i][j]);
        }
        printf("\n");
    }
    printf("\nMatrix B:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_B[i][j] = (float)(i * j);
            printf("%.2f ",h_B[i][j]);
        }
        printf("\n");
    }

    cudaMalloc((void**)&d_A, N * cols * sizeof(float));
    cudaMalloc((void**)&d_B, N * cols * sizeof(float));
    cudaMalloc((void**)&d_C, N * cols * sizeof(float));

    cudaMemcpy(d_A, h_A, N * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * cols * sizeof(float), cudaMemcpyHostToDevice);

    addColumn<<<cols, 1>>>(d_A, d_B, d_C, cols);

    cudaMemcpy(h_C, d_C, N * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Resultant Matrix C (Column-wise Addition):\n");
    for (int i = 0; i < N; i++) {
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
/*
Matrix A:
0.00 1.00 2.00 3.00 
1.00 2.00 3.00 4.00 
2.00 3.00 4.00 5.00 
3.00 4.00 5.00 6.00 

Matrix B:
0.00 0.00 0.00 0.00 
0.00 1.00 2.00 3.00 
0.00 2.00 4.00 6.00 
0.00 3.00 6.00 9.00 
Resultant Matrix C (Column-wise Addition):
0.000000 1.000000 2.000000 3.000000 
1.000000 3.000000 5.000000 7.000000 
2.000000 5.000000 8.000000 11.000000 
3.000000 7.000000 11.000000 15.000000 

*/
