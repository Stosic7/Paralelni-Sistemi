#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#define n 8 // matrica n x n, menjaj dimenziju po potrebi

int main(int argc, char** argv) {
    int i, j, k, l, idx, y;
    int p, rank, q;
    int irow, jcol, row_id, col_id;
    MPI_Comm row_comm, col_comm, com;
    MPI_Datatype blok;

    static int a[n][n], c[n];
    int b[n];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    q = (int)sqrt((double)p);
    if (q * q != p) {
        if (rank == 0) printf("Broj procesa mora biti kvadrat!\n");
        MPI_Finalize();
        return 0;
    }
    k = n / q; // blok dimenzija

    int* local_a = (int*)calloc(k * k, sizeof(int));
    int* local_b = (int*)calloc(k, sizeof(int));
    int* local_c = (int*)calloc(k, sizeof(int));

    // MPI blok tip za slanje podmatrica
    MPI_Type_vector(k, k, n, MPI_INT, &blok);
    MPI_Type_commit(&blok);

    // MASTER: inicijalizacija matrice i vektora b (A=[i+j], b=[i])
    if (rank == 0) {
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                a[i][j] = i + j;
        for (i = 0; i < n; i++)
            b[i] = i;
    }

    // MASTER šalje blokove A ostalim procesima
    if (rank == 0) {
        // master šalje svakom procesu njegov blok
        int proc = 1;
        for (irow = 0; irow < q; irow++) {
            for (jcol = 0; jcol < q; jcol++) {
                if (proc == 0) continue; // za master lokalno kopira
                MPI_Send(&a[irow * k][jcol * k], 1, blok, proc, 0, MPI_COMM_WORLD);
                proc++;
            }
        }
        // master lokalno kopira sebi blok
        idx = 0;
        for (i = 0; i < k; i++)
            for (j = 0; j < k; j++)
                local_a[idx++] = a[i][j];
    } else {
        MPI_Recv(local_a, k*k, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Prava paralelna distribucija vektora b prema zadatku
    int* recv_b = (int*)calloc(k, sizeof(int));
    int b_start = (rank % q) * k;
    if (rank == 0) {
        for (int proc = 0; proc < p; proc++) {
            int bstart = (proc % q) * k;
            if (proc == 0) {
                for (i = 0; i < k; i++)
                    local_b[i] = b[bstart + i];
            } else {
                MPI_Send(&b[bstart], k, MPI_INT, proc, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(local_b, k, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Komunikatori za red/kolonu
    irow = rank / q;
    jcol = rank % q;
    MPI_Comm_split(MPI_COMM_WORLD, irow, rank, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, jcol, rank, &col_comm);
    MPI_Comm_rank(row_comm, &row_id);
    MPI_Comm_rank(col_comm, &col_id);

    // Broadcast vektora b u svakom redu
    MPI_Bcast(local_b, k, MPI_INT, 0, row_comm);

    // Računaj lokalni rezultat (multiplikacija bloka matrice)
    for (i = 0; i < k; i++) {
        local_c[i] = 0;
        for (j = 0; j < k; j++)
            local_c[i] += local_a[i * k + j] * local_b[j];
    }

    // Prikupljanje rezultata putem Gather + Reduce
    int* result = NULL;
    if (col_id == 0) result = (int*)calloc(n, sizeof(int));
    MPI_Gather(local_c, k, MPI_INT, result, k, MPI_INT, 0, col_comm);
    if (col_id == 0)
        MPI_Reduce(result, c, n, MPI_INT, MPI_SUM, 0, row_comm);

    // MASTER upisuje rezultat u binarni fajl
    if (rank == 0) {
        FILE* f = fopen("okt2_c.dat", "wb");
        fwrite(c, sizeof(int), n, f);
        fclose(f);
        printf("Rezultat vektora c upisan u okt2_c.dat:\n");
        for (i = 0; i < n; i++)
            printf("c[%d] = %d\n", i, c[i]);
    }

    free(local_a); free(local_b); free(local_c); free(recv_b); if (result) free(result);
    MPI_Finalize();
    return 0;
}
