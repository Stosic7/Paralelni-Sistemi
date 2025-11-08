#include <stdio.h>
#include <cuda_runtime.h>

#define n 20
#define m 4
#define BLOCK_SIZE 256

__global__ void find_subsequence(char* text, char* pattern, int* niz) {
    __shared__ char shared_pattern[m];
    __shared__ char shared_text[BLOCK_SIZE + m - 1];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (threadIdx.x < m) {
        shared_pattern[threadIdx.x] = pattern[idx];
    }
    
    if (idx < n) {
        shared_text[threadIdx.x] = text[idx];
    }
    if (threadIdx.x < m - 1 && idx + blockDim.x < n) {
        shared_text[threadIdx.x + blockDim.x] = text[idx + blockDim.x];
    }
    
    __syncthreads();
    
    if (idx < n - m + 1) {
        int match = 1;
        for (int i = 0; i < m; i++) {
            if (shared_text[threadIdx.x + i] != shared_pattern[i]) {
                match = 0;
                break;
            }
        }
        niz[idx] = match;
    }
}

int main() {
    char* text = (char*)malloc(n * sizeof(char));
    char* pattern = (char*)malloc(m * sizeof(char));
    int* niz = (int*)malloc((n - m + 1) * sizeof(int));

    strcpy(text, "AUGCAUGCAUGCAUGCAUGC");
    strcpy(pattern, "AUGC");
    
    char* text_d;
    char* pattern_d;
    int* niz_d;

    cudaMalloc((void**)&text_d, n * sizeof(char));
    cudaMalloc((void**)&pattern_d, m * sizeof(char));
    cudaMalloc((void**)&niz_d, (n - m + 1) * sizeof(int));

    cudaMemcpy(text_d, text, n * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(pattern_d, pattern, m * sizeof(char), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = ((n - m + 1) + threadsPerBlock - 1) / threadsPerBlock;

    find_subsequence<<<blocksPerGrid, threadsPerBlock>>>(text_d, pattern_d, niz_d);
    cudaDeviceSynchronize();

    cudaMemcpy(niz, niz_d, (n - m + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n - m + 1; i++) {
        printf("%4d", niz[i]);
    }
    printf("\n");

    cudaFree(text_d);
    cudaFree(pattern_d);
    cudaFree(niz_d);
    free(text);
    free(pattern);
    free(niz);

    return 0;
}
