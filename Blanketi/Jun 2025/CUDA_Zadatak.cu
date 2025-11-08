#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define IDX(i,j,N) ((i)*(N)+(j))
#define TILE 16

__global__ void detect_stars_shared(unsigned char* image, int* stars, int rows, int cols) {

    __shared__ unsigned char tile[TILE+2][TILE+2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;

    for (int dy = ty; dy < TILE+2 && row+dy-ty < rows; dy += blockDim.y) {
        for (int dx = tx; dx < TILE+2 && col+dx-tx < cols; dx += blockDim.x) {
            int global_row = blockIdx.y * TILE + dy - 1;
            int global_col = blockIdx.x * TILE + dx - 1;
            if (global_row >= 0 && global_row < rows && global_col >=0 && global_col < cols) {
                tile[dy][dx] = image[IDX(global_row, global_col, cols)];
            } else {
                tile[dy][dx] = 0; 
            }
        }
    }
    __syncthreads();

    if (row >= rows || col >= cols) return;

    int pixel_value = tile[ty+1][tx+1];
    int neighbor_sum = 0;
    int neighbor_count = 0;

    if (pixel_value > 128) {
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (dr == 0 && dc == 0) continue;
                neighbor_sum += tile[ty+1+dr][tx+1+dc];
                neighbor_count++;
            }
        }

        float neighbor_avg = (neighbor_count > 0) ? ((float)neighbor_sum / neighbor_count) : 0.0f;
        stars[IDX(row, col, cols)] = neighbor_avg > 64 ? 1 : 0;
    } else {
        stars[IDX(row, col, cols)] = 0;
    }
}

int main() {
    int M = 6;
    int N = 6;

    unsigned char* h_A = (unsigned char*)malloc(M*N*sizeof(unsigned char));
    int* h_B = (int*)malloc(M*N*sizeof(int));

    unsigned char sample[36] = {
        20, 30, 140, 200, 50, 10,
        22, 135, 210, 210, 60, 20,
        10, 220, 255, 190, 120, 70,
        10, 128, 129, 130, 128, 10,
        20, 30, 90, 110, 220, 100,
        5, 5, 130, 140, 180, 60
    };

    for (int i = 0; i < N*M; i++)
        h_A[i] = sample[i];

    unsigned char* d_A;
    int* d_B;
    cudaMalloc((void**)&d_A, M*N*sizeof(unsigned char));
    cudaMalloc((void**)&d_B, M*N*sizeof(int));
    cudaMemcpy(d_A, h_A, M*N*sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((N + TILE - 1)/TILE, (M + TILE - 1)/TILE);

    detect_stars_shared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, M*N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Matrica rezultata (B) – zvezde:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_B[IDX(i,j,N)]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}
