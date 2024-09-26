#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void oddPhase(int *d_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx % 2 == 1 && idx < n - 1) {
        if (d_array[idx] > d_array[idx + 1]) {
            int temp = d_array[idx];
            d_array[idx] = d_array[idx + 1];
            d_array[idx + 1] = temp;
        }
    }
}

__global__ void evenPhase(int *d_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx % 2 == 0 && idx < n - 1) {
        if (d_array[idx] > d_array[idx + 1]) {
            int temp = d_array[idx];
            d_array[idx] = d_array[idx + 1];
            d_array[idx + 1] = temp;
        }
    }
}


int main() {
    int n;
    
    // Get the length of the array from user
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    int *h_array = (int*)malloc(n * sizeof(int));
    int *d_array;

    // Get the array elements from user
    printf("Enter %d integers:\n", n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_array[i]);
    }

    printf("Original array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    cudaMalloc((void**)&d_array, n * sizeof(int));
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (n + 1) / 2; // Number of blocks needed for odd and even phases

    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            evenPhase<<<blocks, 2>>>(d_array, n);
        } else {
            oddPhase<<<blocks, 2>>>(d_array, n);
        }

        // Check for errors in kernel launch
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
            break;
        }
    }

    // Copy the sorted array back to the host
    cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");
   
    // Free device memory
    cudaFree(d_array);

    // Free host memory
    free(h_array);
    
    return 0;
}

