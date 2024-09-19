#include <stdio.h>
#include <stdlib.h>

#define N 4  // Number of rows and columns

__global__ void addColumn(float *A, float *B, float *C, int cols) {
    int col = blockIdx.x;  // Each thread handles one column
    if (col < cols) {
        for (int row = 0; row < N; row++) {
            C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
        }
    }
}

__global__ void addElement(float *A, float *B, float *C, int rows, int cols) {
    int row = blockIdx.x;  // Row index
    int col = threadIdx.x; // Column index
    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

__global__ void addRow(float *A, float *B, float *C, int rows) {
    int row = blockIdx.x;  // Each thread handles one row
    if (row < rows) {
        for (int col = 0; col < N; col++) {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}

void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int cols = N, rows = N;
    float *h_A = (float *)malloc(rows * cols * sizeof(float));
    float *h_B = (float *)malloc(rows * cols * sizeof(float));
    float *h_Col = (float *)malloc(rows * cols * sizeof(float));
    float *h_Row = (float *)malloc(rows * cols * sizeof(float));
    float *h_Ele = (float *)malloc(rows * cols * sizeof(float));
    float *d_A, *d_B, *d_C;

    // Initialize matrices A and B
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_A[i * cols + j] = (float)(i + j);
        }
    }
    printf("\nMatrix A:\n");
    printMatrix(h_A, rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_B[i * cols + j] = (float)(i * j);
        }
    }
    printf("\nMatrix B:\n");
    printMatrix(h_B, rows, cols);

    cudaMalloc((void**)&d_A, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_B, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(float));

    cudaMemcpy(d_A, h_A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Column-wise addition
    addColumn<<<cols, 1>>>(d_A, d_B, d_C, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Col, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("\nResultant Matrix C (Column-wise Addition):\n");
    printMatrix(h_Col, rows, cols);
    
    // Row-wise addition
    addRow<<<rows, 1>>>(d_A, d_B, d_C, rows);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Row, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("\nResultant Matrix C (Row-wise Addition):\n");
    printMatrix(h_Row, rows, cols);
    
    // Element-wise addition
    dim3 threadsPerBlock(cols);
    dim3 numBlocks(rows);
    addElement<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Ele, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("\nResultant Matrix C (Element-wise Addition):\n");
    printMatrix(h_Ele, rows, cols);
    
    // Free memory
    free(h_A);
    free(h_B);
    free(h_Col);
    free(h_Row);
    free(h_Ele);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

/*
./prg2.out

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
0.00 1.00 2.00 3.00 
1.00 3.00 5.00 7.00 
2.00 5.00 8.00 11.00 
3.00 7.00 11.00 15.00 

Resultant Matrix C (Row-wise Addition):
0.00 1.00 2.00 3.00 
1.00 3.00 5.00 7.00 
2.00 5.00 8.00 11.00 
3.00 7.00 11.00 15.00 

Resultant Matrix C (Element-wise Addition):
0.00 1.00 2.00 3.00 
1.00 3.00 5.00 7.00 
2.00 5.00 8.00 11.00 
3.00 7.00 11.00 15.00 


*/
