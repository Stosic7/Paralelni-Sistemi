#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <limits.h>

#define n 5

int main(int argc, char** argv) {
    int rank, p, i, j;
    int x = 10;  // Zadato x

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int irow = rank / n;  // Red
    int jcol = rank % n;  // Kolona

    int on_diagonal = (irow + jcol == n - 1); // sporedna dijagonala

    int in_new_comm = (on_diagonal && rank < x);
    printf("in_new_comm value for indexes: %d, [%d, %d]\n", in_new_comm, irow, jcol);

    MPI_Comm diag_comm;
    int color = in_new_comm ? 0 : MPI_UNDEFINED;
    
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &diag_comm);
    
    if (diag_comm != MPI_COMM_NULL) {
        int diag_rank, diag_size;
        MPI_Comm_rank(diag_comm, &diag_rank);
        MPI_Comm_size(diag_comm, &diag_size);
        
        printf("Proces %d (globalni) je %d (novi komunikator)\n", 
               rank, diag_rank);
    }

    static int A[n][n];
    int vrsta[n];
    
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
    
    if (diag_comm != MPI_COMM_NULL) {
        int diag_rank;
        MPI_Comm_rank(diag_comm, &diag_rank);
        
        // Scatter redove
        MPI_Scatter(&A[0][0], n, MPI_INT, vrsta, n, MPI_INT, 0, diag_comm);
        
        printf("Proces %d primio vrstu: ", rank);
        for (j = 0; j < n; j++) {
            printf("%4d", vrsta[j]);
        }
        printf("\n");

        // Proizvod elemenata vrste
        int proizvod = 1;
        for (j = 0; j < n; j++) {
            proizvod *= vrsta[j];
        }
        
        printf("Proces %d: proizvod = %d\n", rank, proizvod);

        // MPI_Reduce za maksimum proizvoda
        int max_proizvod;
        MPI_Reduce(&proizvod, &max_proizvod, 1, MPI_INT, MPI_MAX, 0, diag_comm);
        
        if (diag_rank == 0) {
            printf("MAKSIMUM PROIZVODA: %d\n", max_proizvod);
        }
        
        MPI_Comm_free(&diag_comm);
    }
    
    MPI_Finalize();
    return 0;
}
