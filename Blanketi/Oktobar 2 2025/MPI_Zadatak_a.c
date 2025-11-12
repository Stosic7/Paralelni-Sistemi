#include <stdio.h>
#include <mpi.h>

#define n 3
#define k 4
#define m 2
#define p 2

int main(int argc, char** argv) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int A[n][k], B[k][m], C[n][m];
	int nizB[k][m / p];

    if (rank == 0) {
        printf("Inicijalna matrica A:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                A[i][j] = (i * 2 + j * j - 1) % 10 + 9;
                printf("%d\t", A[i][j]);
            }
            printf("\n");
        }
        fflush(stdout);

        printf("Inicijalna matrica B:\n");
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < m; j++) {
                B[i][j] = (i * 2 + j * j - 1) % 10 + 9;
                printf("%d\t", B[i][j]);
            }
            printf("\n");
        }
        fflush(stdout);
        printf("-------------------\n");
    }

    MPI_Bcast(A, n * k, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Datatype col_vec, col_type;
    MPI_Type_vector(k, m / p, m, MPI_INT, &col_vec);
    MPI_Type_create_resized(col_vec, 0, (m / p) * sizeof(int), &col_type);
    MPI_Type_free(&col_vec);
    MPI_Type_commit(&col_type);

    MPI_Scatter(B, 1, col_type, nizB, k * (m / p), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    printf("\nProces %d local columns:\n", rank);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m / p; j++) {
            printf("[%d]\t", nizB[i][j]);
        }
        printf("\n");
    }
    printf("\n-----------------------------");

    int parcijalno_c[n][m/p];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m/p; j++) {
            parcijalno_c[i][j] = 0;
            for (int t = 0; t < k; t++) {
                parcijalno_c[i][j] += A[i][t] * nizB[t][j];
            }
        }
    }
  
    MPI_Gather(parcijalno_c, n * (m / p), MPI_INT, C, n * (m / p), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    printf("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$");
    if (rank == 0) {
        printf("\nC: \n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%d\t", C[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
