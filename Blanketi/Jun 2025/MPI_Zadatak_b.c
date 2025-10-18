#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <limits.h>

#define n 5

int main(int argc, char** argv) {
    int rank, p, i, j;
    int x = 5; // ispitujemo deljivost

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    MPI_Datatype col_type, col_res;
    MPI_Type_vector(n, 1, n, MPI_INT, &col_type);
    MPI_Type_create_resized(col_type, 0, sizeof(int), &col_res);
    MPI_Type_free(&col_type);
    MPI_Type_commit(&col_res);

    int irow = rank / n;
    int jcol = rank % n;

    int in_new_comm = (rank % x == 0);
    printf("in_new_comm value for indexes: %d, [%d, %d]\n", in_new_comm, irow, jcol);

    MPI_Comm div_comm;
    int color = in_new_comm ? 0 : MPI_UNDEFINED;

    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &div_comm);

    if (div_comm != MPI_COMM_NULL) {
        int div_rank, div_size;
        MPI_Comm_rank(div_comm, &div_rank);
        MPI_Comm_size(div_comm, &div_size);

        printf("Process %d (globalni) je %d (novi komunikator)\n", rank, div_rank);
    }

    static int A[n][n];
    int kolona[n];
    
    if (rank == 0) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                A[i][j] = i * n + j + 1;
            }
        }
        
        printf("Matrica A:\n");
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                printf("%4d", A[i][j]);
            }
            printf("\n");
        }
        printf("-------------------\n");
    }

    MPI_Bcast(&A[0][0], n * n, MPI_INT, 0, MPI_COMM_WORLD);

    if (div_comm != MPI_COMM_NULL) {
        int div_rank;
        MPI_Comm_rank(div_comm, &div_rank);

        MPI_Scatter(&A[0][0], 1, col_res, kolona, n, MPI_INT, 0, div_comm);

        printf("Proces %d primio kolonu: ", rank);
        for (j = 0; j < n; j++) {
            printf("%4d", kolona[j]);
        }
        printf("\n");

        int proizvod = 1;
        for (j = 0; j < n; j++) {
            proizvod *= kolona[j];
        }

        printf("Process %d: proizvod = %d\n", rank, proizvod);

        int max_proizvod;
        MPI_Reduce(&proizvod, &max_proizvod, 1, MPI_INT, MPI_MAX, 0, div_comm);

        if (div_rank == 0) {
            printf("MAKSIMUM PROIZVODA: %d\n", max_proizvod);
        }

        MPI_Comm_free(&div_comm);
        MPI_Type_free(&col_res);
    }

    MPI_Finalize();
    return 0;
}
