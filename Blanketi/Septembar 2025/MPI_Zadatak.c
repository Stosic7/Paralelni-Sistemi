#include <stdio.h>
#include <mpi.h>
#include <limits.h>
#define n 3
#define m 2
#define k 4

int main(int argc, char** argv) {
    static int a[n][k];
    static int b[k][m];
    static int c[n][m];
    int minElement[n]; // min elementi svake vrste
    int rank, i, j, p;
    int nizA[n*(k/p)], nizB[(k/p)*m];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    MPI_Datatype col_vector, col_type;
    MPI_Type_vector(n, k/p, k, MPI_INT, &col_vector);
    MPI_Type_create_resized(col_vector, 0, (k/p)*sizeof(int), &col_type);
    MPI_Type_commit(&col_type);
    MPI_Type_free(&col_vector);

    if (rank == 0) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                a[i][j] = i * 2 + j;
            }
        }

        printf("MATRICA A\n\n");
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                printf("%4d", a[i][j]);
            }
            printf("\n");
        }

        for (i = 0; i < n; i++) {
            int min = INT_MAX;
            for (j = 0; j < k; j++) {
                if (a[i][j] < min) {
                    min = a[i][j];
                }
            }
            minElement[i] = min;
        }

        printf("\nminimalni elementi matrice A: \n");
        for (i = 0; i < n; i++)
            printf("%2d", minElement[i]);
        
        printf("\n\n");
        
        for (i = 0; i < k; i++) {
            for (j = 0; j < m; j++) {
                b[i][j] = i + j + 1;
            }
        }

        printf("MATRICA B\n\n");
        for (i = 0; i < k; i++) {
            for (j = 0; j < m; j++) {
                printf("%2d", b[i][j]);
            }
            printf("\n");
        }

        printf("----------\n");
    }

    MPI_Scatter(&a[0][0], 1, col_type, nizA, n*(k/p), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(&b[0][0], (k/p)*m, MPI_INT, nizB, (k/p)*m, MPI_INT, 0, MPI_COMM_WORLD);

    int parcijalno_c[n][m];
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            parcijalno_c[i][j] = 0;
            for (int t = 0; t < k/p; t++) {
                parcijalno_c[i][j] += nizA[i*(k/p) + t] * nizB[t*m + j];
            }
        }
    }

    MPI_Reduce(parcijalno_c, c, n*m, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nRezultujuca Matrica C (A x B):\n\n");
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                printf("%4d", c[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Type_free(&col_type);
    MPI_Finalize();
    return 0;
}
