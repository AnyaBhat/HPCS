#include <stdio.h>

#define N 4  // Number of columns

__global__ void addRow(float *A, float *B, float *C, int rows) {
    int row = blockIdx.x;  // Each thread handles one row
    if (row < rows) {
        for (int col = 0; col < N; col++) {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}

int main() {
    int rows = 4;
    float h_A[rows][N], h_B[rows][N], h_C[rows][N];
    float *d_A, *d_B, *d_C;


    
    printf("\nMatrix A:\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i][j] = (float)(i + j);
            printf("%.2f ",h_A[i][j]);
        }
        printf("\n");
    }
    printf("\nMatrix B:\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < N; ++j) {
            h_B[i][j] = (float)(i * j);
            printf("%.2f ",h_B[i][j]);
        }
        printf("\n");
    }
    
    cudaMalloc((void**)&d_A, rows * N * sizeof(float));
    cudaMalloc((void**)&d_B, rows * N * sizeof(float));
    cudaMalloc((void**)&d_C, rows * N * sizeof(float));

    cudaMemcpy(d_A, h_A, rows * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, rows * N * sizeof(float), cudaMemcpyHostToDevice);

    addRow<<<rows, 1>>>(d_A, d_B, d_C, rows);

    cudaMemcpy(h_C, d_C, rows * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Resultant Matrix C (Row-wise Addition):\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
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
Resultant Matrix C (Row-wise Addition):
0.000000 1.000000 2.000000 3.000000 
1.000000 3.000000 5.000000 7.000000 
2.000000 5.000000 8.000000 11.000000 
3.000000 7.000000 11.000000 15.000000

*/
