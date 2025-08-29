#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> // for rand()

__global__ void matrixMulKernel(int M, int N, int K, const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMultiply(int M, int N, int K, const float* h_A, const float* h_B, float* h_C) {
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;

    // 1. Allocate device memory
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // 2. Copy data to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // 3. Configure and launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    matrixMulKernel<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);

    // 4. Copy result back
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // 5. Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int M = 512, N = 512, K = 512; // Matrix dimensions
    
    // Allocate host memory
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Run matrix multiplication
    matrixMultiply(M, N, K, h_A, h_B, h_C);

    // Print a sample result
    std::cout << "C[0][0] = " << h_C[0] << std::endl;
    std::cout << "C[100][100] = " << h_C[100 * N + 100] << std::endl;

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}