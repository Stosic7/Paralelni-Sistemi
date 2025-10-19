#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define n 4  
#define m 6  

int main(int argc, char** argv) {
    static int A[n][m];
    int b[m];
    int rank, p, i, j;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int num_local_cols = (m + p - 1) / p;

    int local_columns[n * num_local_cols]; 
    int local_b[num_local_cols];             
    int local_result[n];                     

    if (rank == 0) {
        // Inicijalizuj matricu A
        for (i = 0; i < n; i++)
            for (j = 0; j < m; j++)
                A[i][j] = i * m + j + 1;

        printf("Matrica A (%d×%d):\n", n, m);
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++)
                printf("%4d", A[i][j]);
            printf("\n");
        }

        // Inicijalizuj vektor b
        for (i = 0; i < m; i++)
            b[i] = i + 1;

        printf("\nVektor b (%d elemenata): ", m);
        for (i = 0; i < m; i++)
            printf("%4d", b[i]);
        printf("\n-------------------\n");
    }

    MPI_Bcast(&A[0][0], n * m, MPI_INT, 0, MPI_COMM_WORLD);

    int *sendbuf_A = NULL;
    
    if (rank == 0) {
        sendbuf_A = (int*)malloc(n * m * sizeof(int));
        int idx = 0;

        for (int proc = 0; proc < p; proc++) {
            for (int col = proc; col < m; col += p) {
                for (int row = 0; row < n; row++) {
                    sendbuf_A[idx++] = A[row][col];
                }
            }
        }

        while (idx < p * num_local_cols * n) {
            sendbuf_A[idx++] = 0;
        }
    }

    MPI_Scatter(sendbuf_A, num_local_cols * n, MPI_INT,
                local_columns, num_local_cols * n, MPI_INT,
                0, MPI_COMM_WORLD);

    int *sendbuf_b = NULL;
    
    if (rank == 0) {
        sendbuf_b = (int*)malloc(m * sizeof(int));
        int idx = 0;

        for (int proc = 0; proc < p; proc++) {
            for (int elem = proc; elem < m; elem += p) {
                sendbuf_b[idx++] = b[elem];
            }
        }

        while (idx < p * num_local_cols) {
            sendbuf_b[idx++] = 0;
        }
    }

    MPI_Scatter(sendbuf_b, num_local_cols, MPI_INT,
                local_b, num_local_cols, MPI_INT,
                0, MPI_COMM_WORLD);


    for (i = 0; i < n; i++)
        local_result[i] = 0;

    for (int col_idx = 0; col_idx < num_local_cols; col_idx++) {
        int global_col = rank + col_idx * p;

        if (global_col < m) {
            for (int row = 0; row < n; row++) {
                local_result[row] += local_columns[col_idx * n + row] * local_b[col_idx];
            }
        }
    }

    printf("Proces %d: parcijalni rezultat = ", rank);
    for (i = 0; i < n; i++)
        printf("%4d", local_result[i]);
    printf("\n");

    int final_result[n];
    MPI_Reduce(local_result, final_result, n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("FINALNI REZULTAT A × b:\n");
        for (i = 0; i < n; i++)
            printf("rezultat[%d] = %d\n", i, final_result[i]);

        free(sendbuf_A);
        free(sendbuf_b);
    }

    MPI_Finalize();
    return 0;
}
