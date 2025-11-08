#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void detect_spikes(float* A, int* B, int N, float alpha) {
    __shared__ float shared_data[BLOCK_SIZE + 2];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    
    if (idx < N) {
        shared_data[tx + 1] = A[idx];
    }
    
    if (tx == 0 && idx > 0) {
        shared_data[0] = A[idx - 1];
    }
    
    if (tx == blockDim.x - 1 && idx < N - 1) {
        shared_data[tx + 2] = A[idx + 1];
    }
    
    __syncthreads();
    
    if (idx > 0 && idx < N - 1) {
        int found = 0;
        float current = shared_data[tx + 1];
        float left = shared_data[tx];
        float right = shared_data[tx + 2];
        
        if (current > left && current > right && current > alpha) {
            found = 1;
        }
        B[idx] = found;
    }
}

int main() {
    int N = 10;
    float alpha = 2.0f;

    float sequence[] = { -3.2f, 4.1f, -5.0f, 2.5f, 1.9f, 6.7f, -2.3f, 0.3f, 1.1f, 10.5f };
    float* h_A = (float*)malloc(N * sizeof(float));
    int* h_B = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_A[i] = sequence[i];
        h_B[i] = 0;
    }

    float* d_A;
    int* d_B;

    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    detect_spikes<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N, alpha);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%4d", h_B[i]);
    }
    printf("\n");

    int count = 0;
    for (int i = 0; i < N; i++) {
        if (h_B[i] == 1) {
            count++;
        }
    }

    printf("Broj otkucaja srca (spikes): %d\n", count);

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}
