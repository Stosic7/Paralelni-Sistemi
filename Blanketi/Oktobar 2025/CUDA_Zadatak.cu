#include <stdio.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void find_subsequence(char* A, char* pattern, int* B, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n - m + 1) {
        int match = 1;
        for (int i = 0; i < m; i++) {
            if (A[idx + i] != pattern[i]) {
                match = 0;
                break;
            }
        }

        B[idx] = match;
    }
}

int main() {
    // CPU part
    int n = 20; // duzina RNK sekvence
    int m = 4; // duzina trazene podsekvence

    char* h_A = (char*)malloc(n * sizeof(char));
    char* h_pattern = (char*)malloc(m * sizeof(char));
    int* h_B = (int*)malloc((n - m + 1) * sizeof(int));

    char sequence[] = "AUGCAUGCUAGCUAGCUAGC";
    memcpy(h_A, sequence, n * sizeof(char));

    char search_pattern[] = "AUGC";
    memcpy(h_pattern, search_pattern, m * sizeof(char));

    for (int i = 0; i < n - m + 1; i++) {
        h_B[i] = 0;
    }

    printf("RNK sekvenca: %s\n", h_A);
    printf("Trazena podsekevnca: %s\n", h_pattern);
    printf("Duzina sekvence: %d\n", n);
    printf("Duzina podsekevnce: %d\n", m);
    printf("\n");

    // GPU part
    char* d_A;
    char* d_pattern;
    int* d_B;

    cudaMalloc((void**)&d_A, n * sizeof(char));
    cudaMalloc((void**)&d_pattern, m * sizeof(char));
    cudaMalloc((void**)&d_B, (n - m + 1) * sizeof(int));

    cudaMemcpy(d_A, h_A, n * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, h_pattern, m * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, (n - m + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;

    int blocksPerGrid = (n - m + 1 + threadsPerBlock - 1) / threadsPerBlock;

    printf("Konfiguracija kernela:\n");
    printf("  Blokova: %d\n", blocksPerGrid);
    printf("  Thread-ova po bloku: %d\n", threadsPerBlock);
    printf("  Ukupno thread-ova: %d\n", blocksPerGrid * threadsPerBlock);
    printf("  Potrebno thread-ova: %d\n", n - m + 1);
    printf("\n");

    // pokretanje
    find_subsequence<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_pattern, d_B, n, m);
    cudaDeviceSynchronize();

    //rezultati
    cudaMemcpy(h_B, d_B, (n - m + 1)*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Rezultati pretrage:\n");
    printf("Pozicija | Pronadjeno\n");
    printf("---------|------------\n");
    for (int i = 0; i < n - m + 1; i++) {
        printf("   %2d    |     %d\n", i, h_B[i]);
        if (h_B[i] == 1) {
            printf("          | --> Podsekevnca pronadjena na poziciji %d: ", i);
            for (int j = 0; j < m; j++) {
                printf("%c", h_A[i + j]);
            }
            printf("\n");
        }
    }

    // oslobadjanje memorije
    cudaFree(d_A);
    cudaFree(d_pattern);
    cudaFree(d_B);

    free(h_A);
    free(h_pattern);
    free(h_B);

    return 0;
}
