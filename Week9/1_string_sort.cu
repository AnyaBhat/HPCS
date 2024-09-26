#include <stdio.h>
#include <cuda.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void selectionSort(char *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (idx >= n) return;

    for (int i = 0; i < n - 1; i++) {
       
        if (idx == 0) {
            int minIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIdx]) {
                    minIdx = j;
                }
            }
            // Swap if needed
            if (minIdx != i) {
                char temp = arr[i];
                arr[i] = arr[minIdx];
                arr[minIdx] = temp;
            }
        }
        //__syncthreads();
    }
}

int main() {
    char *str = (char *)malloc(100 * sizeof(char)); // Allocate memory for the input string
    if (str == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    printf("Enter a string (up to 99 characters): ");
    fgets(str, 100, stdin);
    
    // Remove newline character if present
    str[strcspn(str, "\n")] = 0;
	
    int n = strlen(str);
    printf("Length = %d\n", n);
   
    printf("Original string: %s\n", str);
   
    char *d_a;

   
    cudaMalloc((void **)&d_a, n * sizeof(char));

 
    cudaMemcpy(d_a, str, n * sizeof(char), cudaMemcpyHostToDevice);

   
    int blockSize = 1;
    int numBlocks = 1;
    selectionSort<<<numBlocks, blockSize>>>(d_a, n);
 
    cudaMemcpy(str, d_a, n * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
   
    printf("Sorted string: %s\n", str);

    return 0;
}
/*

./prg1.out 
Enter a string (up to 99 characters): cudaprogram
Length = 11
Original string: cudaprogram
Sorted string: aacdgmoprru

*/
