#include <stdio.h>
#include <mpi.h>

#define n 6
#define m 6
#define p 3

int main(int argc, char** argv) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int A[n][m], B[m], C[n];
    int localColumns[n][m/p];
    int localVec[m/p];

    if (rank == 0) {
        printf("Inicijalna matrica A:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                A[i][j] = (i + j) % 10;
                printf("%d\t", A[i][j]);
            }
            printf("\n");
        }
        fflush(stdout);

        printf("Inicijalna vector B:");
        for (int i = 0; i < m; i++) {
            B[i] = i+1;
            printf("%d ", B[i]);
        }
        printf("\n");
    }

    MPI_Datatype col_vec, col_type;
    MPI_Type_vector(m/p * n, 1, p, MPI_INT, &col_vec);
    MPI_Type_create_resized(col_vec, 0, sizeof(int), &col_type);
    MPI_Type_free(&col_vec);
    MPI_Type_commit(&col_type);

    MPI_Scatter(A, 1, col_type, localColumns, n*(m/p), MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    printf("\nProces %d local columns:\n", rank);
        for (int i = 0; i < n; i++){
            for (int j = 0; j < m/p; j++){
                printf("%d\t", localColumns[i][j]);
            }
            printf("\n");
        }   
    printf("\n-----------------------------");

    MPI_Datatype vec, vec_type;
    MPI_Type_vector(m/p, 1, p, MPI_INT, &vec);
    MPI_Type_create_resized(vec, 0, sizeof(int), &vec_type);
    MPI_Type_free(&vec);
    MPI_Type_commit(&vec_type);

    MPI_Scatter(B, 1, vec_type, localVec, m/p, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    printf("\nProces %d local vector: ", rank);
        for (int i = 0; i < m/p; i++)
            printf("%d ", localVec[i]);
    printf("\n");

    int parcijalniRezultat[n];

    for (int i = 0; i < n; i++) {
        parcijalniRezultat[i] = 0;
        for (int j = 0; j < m/p; j++) {
            parcijalniRezultat[i] += localColumns[i][j] * localVec[j];
        }
    }

    MPI_Reduce(parcijalniRezultat, C, n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    printf("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$");
    if (rank == 0){
        printf("\nC: ");
        for (int i = 0; i < n; i++)
            printf("%d ", C[i]);
    }
    fflush(stdout);

    MPI_Finalize();
    return 0;
    
}
