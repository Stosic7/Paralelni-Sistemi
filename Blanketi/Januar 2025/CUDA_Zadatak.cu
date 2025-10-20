#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define IDX(i, j, N) ((i)*(N)+(j)) // indeksiranje 2D matrice u 1D nizu

__global__ void local_variance(const unsigned char* A, float* B, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M || j >= N) return;

    float sum = 0.0f;
    int window[3][3];

    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            int ni = i + di;
            int nj = j + dj;
            int val = 0;
            if (ni >= 0 && ni < M && nj >= 0 && nj < N) {
                val = (int)A[IDX(ni, nj, N)];
            }
            window[di+1][dj+1] = val;
            sum += val;
        }
    }

    float mean = sum / 9.0f;

    float var = 0.0f;
    for (int di = 0; di < 3; di++) {
        for (int dj = 0; dj < 3; dj++) {
            float diff = window[di][dj] - mean;
            var += diff * diff;
        }
    }

    var /= 9.0f;

    B[IDX(i, j, N)] = var;
}

int main() {
    // CPU
    int M = 4;
    int N = 5;

    unsigned char sample[20] = {
        10, 20, 30, 40, 50,
        10, 30, 80, 120, 240,
        22, 33, 44, 55, 60,
        110, 120, 130, 140, 150
    };

    unsigned char* h_A = (unsigned char*)malloc(M * N * sizeof(unsigned char));
    float* h_B = (float*)malloc(N*M*sizeof(float));

    for (int i = 0; i < M * N; i++)
        h_A[i] = sample[i];
    

    // GPU
    unsigned char* d_A;
    float* d_B;

    cudaMalloc((void**)&d_A, M*N*sizeof(unsigned char));
    cudaMalloc((void**)&d_B, M*N*sizeof(float));
    cudaMemcpy(d_A, h_A, M*N*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // definicija blokova i grid-a
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1)/threadsPerBlock.x, (M + threadsPerBlock.y - 1)/threadsPerBlock.y);

    // kernel
    local_variance<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);

     printf("Matrica varijanse (B):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%.2f ", h_B[IDX(i, j, N)]);
        printf("\n");
    }

    cudaFree(d_A); 
    cudaFree(d_B);
    free(h_A); free(h_B);
    return 0;
    
}
