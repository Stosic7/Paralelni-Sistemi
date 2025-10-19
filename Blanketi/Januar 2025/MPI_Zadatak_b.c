#include <stdio.h>
#include <mpi.h>
#include <limits.h>
#define n 5

int main(int argc, char** argv) {
    int rank, p, i ,j;
    int irow, jcol;
    int glavna_dijagonala;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    MPI_Comm main_diag_comm;
    irow = rank / n;
    jcol = rank % n;

    glavna_dijagonala = (irow == jcol);
    printf("main_diag_comm value for indexes: %d, [%d, %d]\n", glavna_dijagonala, irow, jcol);


    int color = glavna_dijagonala ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &main_diag_comm);

    if (main_diag_comm != MPI_COMM_NULL) {
        int main_rank, main_size;
        MPI_Comm_rank(main_diag_comm, &main_rank);
        MPI_Comm_size(main_diag_comm, &main_size);

        printf("Process %d (globalni) je %d (novi komunikator)\n", rank, main_rank);
    }

    static int A[n][n];
    int vrste[n];

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

    MPI_Bcast(&A[0][0], n*n, MPI_INT, 0, MPI_COMM_WORLD);

    if (main_diag_comm != MPI_COMM_NULL) {
        int main_rank;
        MPI_Comm_rank(main_diag_comm, &main_rank);

        MPI_Scatter(&A[0][0], n, MPI_INT, vrste, n, MPI_INT, 0, main_diag_comm);

        printf("Process %d, sa identifikatorom: %d je primio vrstu: ", rank, main_rank);
        for (i = 0; i < n; i++)
            printf("%4d", vrste[i]);
        printf("\n");

        int suma = 0;
        for (i = 0; i < n; i++)
            suma += vrste[i];
        
        printf("Suma vrste primljene od processa sa identifikatorom %d je: %d\n", main_rank, suma);

        MPI_Comm_free(&main_diag_comm);
    }

    MPI_Finalize();
    return 0;
}
