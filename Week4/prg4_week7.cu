#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to reverse each word in parallel
__global__ void reverse_words(char *input, char *output, int *word_start, int *word_len, int word_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < word_count) {
        int start = word_start[idx];
        int len = word_len[idx];
        // Reverse the word by copying from end to beginning
        for (int i = 0; i < len; ++i) {
            output[start + i] = input[start + len - 1 - i];
        }
    }
}

// Helper function to find words and their lengths
void find_words(const char *input, int *word_start, int *word_len, int *word_count, int len) {
    int count = 0;
    int in_word = 0;
    for (int i = 0; i < len; ++i) {
        if (input[i] != ' ' && !in_word) {
            // New word starts
            word_start[count] = i;
            in_word = 1;
        } else if (input[i] == ' ' && in_word) {
            // Word ends
            word_len[count] = i - word_start[count];
            count++;
            in_word = 0;
        }
    }
    // Capture the last word if it doesn't end with a space
    if (in_word) {
        word_len[count] = len - word_start[count];
        count++;
    }
    *word_count = count;
}

// Function to launch the CUDA kernel and process the string
void process_string(const char *input, char *output, int len) {
    int word_start[100], word_len[100];
    int word_count;

    // Find where each word starts and its length
    find_words(input, word_start, word_len, &word_count, len);

    // Allocate memory on the GPU
    char *d_input, *d_output;
    int *d_word_start, *d_word_len;

    cudaMalloc((void**)&d_input, len * sizeof(char));
    cudaMalloc((void**)&d_output, len * sizeof(char));
    cudaMalloc((void**)&d_word_start, word_count * sizeof(int));
    cudaMalloc((void**)&d_word_len, word_count * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_input, input, len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word_start, word_start, word_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word_len, word_len, word_count * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid size for the CUDA kernel
    int blockSize = 256;
    int numBlocks = (word_count + blockSize - 1) / blockSize;

    // Launch the kernel to reverse each word
    reverse_words<<<numBlocks, blockSize>>>(d_input, d_output, d_word_start, d_word_len, word_count);

    // Copy the result back to host
    cudaMemcpy(output, d_output, len * sizeof(char), cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_word_start);
    cudaFree(d_word_len);
}

int main() {
    // Input string from the user
    char input[256];
    printf("Enter a string: ");
    fgets(input, 256, stdin);

    // Remove the newline character if present
    int len = 0;
    while (input[len] != '\0' && input[len] != '\n') {
        len++;
    }

    // Allocate memory for the output string
    char output[len + 1];
    output[len] = '\0';  // Null terminate the output string

    // Process the input string using CUDA
    process_string(input, output, len);

    // Print the resultant string
    printf("Resultant string: %s\n", output);

    return 0;
}


/*
Enter a string: Hello CUDA World
Resultant string: olleH ADUC dlroW


*/
