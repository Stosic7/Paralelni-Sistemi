#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define IDX(i, j, N) ((i)*(N)+(j))
#define TILE 16

__global__ void local_variance(const unsigned char* input_image, float* variance_matrix, int rows, int cols) {
    __shared__ unsigned char tile[TILE + 2][TILE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;

    int tileSize = TILE + 2;
    for (int i = ty; i < tileSize; i += TILE) {
        for (int j = tx; j < tileSize; j += TILE) {
            int gr = blockIdx.y * TILE + i - 1;
            int gc = blockIdx.x * TILE + j - 1;
            tile[i][j] = (gr >= 0 && gr < rows && gc >= 0 && gc < cols) 
                         ? input_image[IDX(gr, gc, cols)] : 0;
        }
    }
    __syncthreads();

    if (row >= rows || col >= cols) return;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    #pragma unroll
    for (int dr = -1; dr <= 1; dr++) {
        #pragma unroll
        for (int dc = -1; dc <= 1; dc++) {
            unsigned char val = tile[ty + 1 + dr][tx + 1 + dc];
            sum += val;
            sum_sq += val * val;
        }
    }

    float mean = sum * 0.111111f; // 1/9
    variance_matrix[IDX(row, col, cols)] = sum_sq * 0.111111f - mean * mean;
}

int main() {
    int M = 4;
    int N = 5;

    unsigned char sample[20] = {
        10, 20, 30, 40, 50,
        10, 30, 80, 120, 240,
        22, 33, 44, 55, 60,
        110, 120, 130, 140, 150
    };

    unsigned char* h_A = (unsigned char*)malloc(M * N * sizeof(unsigned char));
    float* h_B = (float*)malloc(N * M * sizeof(float));

    for (int i = 0; i < M * N; i++)
        h_A[i] = sample[i];

    unsigned char* d_A;
    float* d_B;

    cudaMalloc((void**)&d_A, M * N * sizeof(unsigned char));
    cudaMalloc((void**)&d_B, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A, M * N * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    local_variance<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Matrica varijanse (B):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%.2f ", h_B[IDX(i, j, N)]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    return 0;
}
