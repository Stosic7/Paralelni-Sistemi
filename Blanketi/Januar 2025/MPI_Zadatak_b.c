#include <stdio.h>
#include <mpi.h>

#define n 6
#define m 6

int main(int argc, char** argv) {
    int size, rank;
    int irow, jcol;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int A[n][m];
    int vrsta[m];

    irow = rank / n;
    jcol = rank % n;

    int condition = (irow == jcol);
    int color;
    
    if (condition) {
        color = 1;
    } else {
        color = MPI_UNDEFINED;
    }

    MPI_Comm diag_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &diag_comm);

    MPI_Datatype row;
    MPI_Type_contiguous(m, MPI_INT, &row);
    MPI_Type_commit(&row);

    if (color != MPI_UNDEFINED) {
        int diagRank, diagSize;
        MPI_Comm_size(diag_comm, &diagSize);
        MPI_Comm_rank(diag_comm, &diagRank);

        if (diagRank == 0) {
            printf("Inicijalna matrica A:\n");
            for (int i = 0; i < n; i++){
                for (int j = 0; j < m; j++){
                    A[i][j] = (i + j) % 10;
                    printf("%d\t", A[i][j]);
                }
                printf("\n");
            }     
            fflush(stdout);
        }

        MPI_Scatter(A, 1, row, vrsta, m, MPI_INT, 0, diag_comm);
        MPI_Barrier(diag_comm);

        printf("\nProces %d elementi: ", rank);
        for (int i = 0; i < m; i++){
            printf("%d ", vrsta[i]);
            fflush(stdout);
        }
    }

    fflush(stdout);
    MPI_Type_free(&row);

    MPI_Finalize();
    return 0;
}
