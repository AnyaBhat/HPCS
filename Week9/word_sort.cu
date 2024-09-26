#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ char toLower(char c) {
    return (c >= 'A' && c <= 'Z') ? (c + 32) : c; // Convert uppercase to lowercase
}

__device__ void selectionSort(char* word, int length) {
    for (int i = 0; i < length - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < length; j++) {
            if (toLower(word[j]) < toLower(word[min_idx])) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            // Swap characters
            char temp = word[i];
            word[i] = word[min_idx];
            word[min_idx] = temp;
        }
    }
}

__global__ void sortWordsKernel(char* words, int* lengths, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        selectionSort(&words[idx * 100], lengths[idx]); // Assuming max word length is 100
    }
}

void sortWords(const char* input) {
    // Split input into words
    char* inputCopy = strdup(input); // Duplicate input to avoid modifying the original
    const char* delimiter = " ";
    char* token = strtok(inputCopy, delimiter);

    // Count words and prepare lengths
    int numWords = 0;
    int lengths[100]; // Assume a max of 100 words for simplicity
    while (token != NULL) {
        lengths[numWords++] = strlen(token);
        token = strtok(NULL, delimiter);
    }
    free(inputCopy); // Free duplicated string

    char* d_words;
    int* d_lengths;

    // Prepare words for CUDA
    char wordsBuffer[100][100] = {0}; // Assume max 100 words, each of max length 100
    inputCopy = strdup(input);
    token = strtok(inputCopy, delimiter);
    for (int i = 0; i < numWords; i++) {
        strncpy(wordsBuffer[i], token, lengths[i]);
        wordsBuffer[i][lengths[i]] = '\0'; // Null-terminate each word
        token = strtok(NULL, delimiter);
    }
    free(inputCopy);

    // Allocate device memory
    cudaMalloc(&d_words, sizeof(wordsBuffer));
    cudaMalloc(&d_lengths, numWords * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_words, wordsBuffer, sizeof(wordsBuffer), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths, numWords * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (numWords + threadsPerBlock - 1) / threadsPerBlock;
    sortWordsKernel<<<blocks, threadsPerBlock>>>(d_words, d_lengths, numWords);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(wordsBuffer, d_words, sizeof(wordsBuffer), cudaMemcpyDeviceToHost);

    // Print sorted words
    printf("Sorted Words:\n");
    for (int i = 0; i < numWords; i++) {
        printf("%s\n", wordsBuffer[i]); // Print each word directly
    }

    // Cleanup
    cudaFree(d_words);
    cudaFree(d_lengths);
}

int main() {
    // Get input from user
    printf("Enter a string: ");
    char* input = (char*)malloc(256 * sizeof(char)); // Allocate memory for input
    fgets(input, 256, stdin); // Read input string
    input[strcspn(input, "\n")] = '\0'; // Remove newline character if present

    sortWords(input);

    // Free allocated memory
    free(input);
    return 0;
}

