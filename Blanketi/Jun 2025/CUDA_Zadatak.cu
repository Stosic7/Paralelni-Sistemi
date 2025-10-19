#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define IDX(i, j, N) ((i)*(N)+(j)) // indeksiranje 2D matrice u 1D nizu

__global__ void detect_stars(unsigned char* A, int* B, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // index reda
    int j = blockIdx.x * blockDim.x + threadIdx.x; // index kolone
    
    if (i >= M || j >= N) return;

    int value = A[IDX(i, j, N)];

    if (value > 128) {
        int sum = 0;
        int count = 0;
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                if (di == 0 && dj == 0) continue;
                int ni = i + di;
                int nj = j + dj;
                if (ni >= 0 && ni < M && nj >= 0 && nj < N) {
                    sum += A[IDX(ni, nj, N)];
                    count++;
                }
            }
        }

        float avg = (count > 0) ? ((float)sum/count) : 0.0f;
        B[IDX(i, j, N)] = avg > 64 ? 1 : 0;
    } else {
        B[IDX(i, j, N)] = 0;
    }
}

int main() {
    // cpu
    int M = 6;
    int N = 6;

    unsigned char* h_A = (unsigned char*)malloc(M*N*sizeof(unsigned char));
    int* h_B = (int*)malloc(M * N * sizeof(int));

    // Primer slike
    unsigned char sample[36] = {
        20, 30, 140, 200, 50, 10,
        22, 135, 210, 210, 60, 20,
        10, 220, 255, 190, 120, 70,
        10, 128, 129, 130, 128, 10,
        20, 30, 90, 110, 220, 100,
        5, 5, 130, 140, 180, 60
    };

    for (int i = 0; i < N * M; i++)
        h_A[i] = sample[i];
    
    // gpu
    unsigned char* d_A;
    int* d_B;

    cudaMalloc((void**)&d_A, M*N*sizeof(unsigned char));
    cudaMalloc((void**)&d_B, M*N*sizeof(int));

    cudaMemcpy(d_A, h_A, M*N*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // definicija blokova i grid-a
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // kernel poziv
    detect_stars<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, M*N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Matrica rezultata (B) – zvezde:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_B[IDX(i, j, N)]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}
