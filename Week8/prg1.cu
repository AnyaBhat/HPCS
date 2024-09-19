#include <stdio.h>
#include <cuda_runtime.h>

__global__ void mat_mul(float *A, float *B, float *C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < P) {
        float Cvalue = 0;
        for (int k = 0; k < N; ++k) {
            Cvalue += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = Cvalue;
    }
}

int main() {
    int M, N, P; // Dimensions of the matrices
    printf("Enter the size of the matrix M,N and P(A[M][N] B[N][P]):");
    scanf("%d",&M);scanf("%d",&N);scanf("%d",&P);
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * P * sizeof(float);
    size_t sizeC = M * P * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    
    // Initialize host matrices
    
    printf("\nMatrix A:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = (float)(i + j);
            printf("%.2f ",h_A[i * N + j]);
        }
        printf("\n");
    }
    printf("\nMatrix B:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            h_B[i * P + j] = (float)(i * j);
            printf("%.2f ",h_B[i * P + j]);
        }
        printf("\n");
    }

    // Device pointers
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    // Kernel launch parameters
    dim3 dimBlock(16, 16);
    dim3 dimGrid((P + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    
    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    // Launch the kernel
    mat_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, P);
    cudaEventRecord(stop);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("\nResultant Matrix:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            printf("%.2f ",h_C[i*P+j]);
        }
        printf("\n");
    }

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Output execution time
    printf("\nKernel execution time: %f ms\n", milliseconds);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}



/*

$ nvcc -o mat_mul.out mat_mul.cu
$ ./mat_mul.out
Enter the size of the matrix M,N and P(A[M][N] B[N][P]):5 4 5

Matrix A:
0.00 1.00 2.00 3.00 
1.00 2.00 3.00 4.00 
2.00 3.00 4.00 5.00 
3.00 4.00 5.00 6.00 
4.00 5.00 6.00 7.00 

Matrix B:
0.00 0.00 0.00 0.00 0.00 
0.00 1.00 2.00 3.00 4.00 
0.00 2.00 4.00 6.00 8.00 
0.00 3.00 6.00 9.00 12.00 

Resultant Matrix:
0.00 14.00 28.00 42.00 56.00 
0.00 20.00 40.00 60.00 80.00 
0.00 26.00 52.00 78.00 104.00 
0.00 32.00 64.00 96.00 128.00 
0.00 38.00 76.00 114.00 152.00 

Kernel execution time: 0.011200 ms
*/
