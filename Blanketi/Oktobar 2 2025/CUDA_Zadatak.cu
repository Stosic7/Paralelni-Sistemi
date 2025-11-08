#include <stdio.h>
#include <cuda_runtime.h>

#define THRESHOLD 0.5f
#define BLOCK_SIZE 256

__global__ void checkPlants(float* A, int M, int N, int* result) {
    __shared__ float shared_row[BLOCK_SIZE * 8];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < M) {
        for (int col = 0; col < N; col++) {
            shared_row[threadIdx.x * N + col] = A[idx * N + col];
        }
    }
    
    __syncthreads();
    
    if (idx >= M) return;
    
    int count = 0;
    
    for (int col = N - 1; col > 0; col--) {
        float curr = shared_row[threadIdx.x * N + col];
        float prev = shared_row[threadIdx.x * N + (col - 1)];
        
        if (curr < THRESHOLD && curr < prev) {
            count++;
        } else {
            break;
        }
    }
    
    result[idx] = count;
}

int main() {
    int M = 4;
    int N = 8;
    float hostA[32] = {
        0.8, 0.7, 0.6, 0.45, 0.40, 0.38, 0.37, 0.35,
        0.6, 0.55, 0.52, 0.50, 0.49, 0.48, 0.47, 0.46,
        0.9, 0.9,  0.9,  0.9,  0.9,  0.9,  0.9,  0.9,
        0.7, 0.6,  0.55, 0.54, 0.53, 0.52, 0.51, 0.50
    };

    float* devA;
    int* devRes;
    int* hostRes = (int*)malloc(sizeof(int) * M);

    cudaMalloc(&devA, sizeof(float) * M * N);
    cudaMalloc(&devRes, sizeof(int) * M);

    cudaMemcpy(devA, hostA, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int numBlocks = (M + blockSize - 1) / blockSize;

    checkPlants<<<numBlocks, blockSize>>>(devA, M, N, devRes);
    cudaDeviceSynchronize();

    cudaMemcpy(hostRes, devRes, sizeof(int) * M, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        if (hostRes[i] == 5) {
            printf("Biljka %d nije polivena 5 sati.\n", i + 1);
        } else if (hostRes[i] == 0) {
            printf("Biljka %d upravo zalivena.\n", i + 1);
        } else {
            printf("Biljka %d polivena korektno [pre %d sati poslednji put zalivena]\n", i + 1, hostRes[i]);
        }
    }

    cudaFree(devA);
    cudaFree(devRes);
    free(hostRes);

    return 0;
}
