#include <cuda_runtime.h>
#include <iostream>

// Kernel definition for matrix addition
__global__ void matrixAdd(int N, float* A, float* B, float* C) {
    // Calculate row and column indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if within bounds
    if (i < N && j < N) {
        // Linear index for 1D array representation
        int idx = j * N + i;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1024; // Matrix size (1024x1024)
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize host matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16); // 256 threads per block
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrixAdd<<<numBlocks, threadsPerBlock>>>(N, d_A, d_B, d_C);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result (optional)
    bool success = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            success = false;
            break;
        }
    }
    std::cout << "Result: " << (success ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}