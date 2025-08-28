#include <stdio.h>
#include <cuda_runtime.h> // Needed for CUDA functions and types

// 1. KERNEL DEFINITION
// This function will execute on the GPU
__global__ void addArrays(int n, float *a, float *b, float *result) {
    // Calculate a unique index for each thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 10){
    printf("index: %d = %d * %d + %d \n", index, blockIdx.x, blockDim.x, threadIdx.x);

   // printf("bockIdx = %d %d %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
   // printf("blockDim = %d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
   // printf("threadIdx = %d %d %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
    }
    // Check if this thread's index is within the array bounds
    if (index < n) {
        // This single line of code is executed in parallel by ALL threads
        result[index] = a[index] + b[index];
    }
}

int main() {
    // 2. SETUP PROBLEM SIZE AND HOST (CPU) MEMORY
    int numElements = 1000000;
    size_t size = numElements * sizeof(float);

    // Allocate and initialize host arrays
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_result = (float *)malloc(size); // To store results from GPU

    for (int i = 0; i < numElements; i++) {
        h_a[i] = 1.0f; // Initialize array a with 1.0
        h_b[i] = 2.0f; // Initialize array b with 2.0
    }

    // 3. ALLOCATE DEVICE (GPU) MEMORY
    float *d_a = NULL, *d_b = NULL, *d_result = NULL;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, size);

    // 4. COPY DATA FROM HOST TO DEVICE
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 5. CONFIGURE AND LAUNCH THE KERNEL
    // Define the execution configuration
    int threadsPerBlock = 256; // A common choice
    printf("threadsPerBlock = %d \n", threadsPerBlock);

    // Calculate the number of blocks needed to cover the entire array
    int blocksPerGrid = (numElements) / threadsPerBlock;
    printf("blocksPerGrid = %d \n", blocksPerGrid);

    // Launch the kernel on the GPU
    // Syntax: <<<Number of Blocks, Threads per Block>>>
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(numElements, d_a, d_b, d_result);

    // 6. COPY RESULT BACK FROM DEVICE TO HOST
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

    // 7. VERIFY THE RESULTS
    // Check the first and last few elements for correctness
    for (int i = 0; i < 5; i++) {
        printf("Element %d: %.1f + %.1f = %.1f (expected 3.0)\n",
               i, h_a[i], h_b[i], h_result[i]);
    }
    printf("...\n");
    for (int i = numElements-5; i < numElements; i++) {
        printf("Element %d: %.1f + %.1f = %.1f (expected 3.0)\n",
               i, h_a[i], h_b[i], h_result[i]);
    }

    // 8. FREE ALL ALLOCATED MEMORY
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(h_a);
    free(h_b);
    free(h_result);

    printf("Done!\n");
    return 0;
}