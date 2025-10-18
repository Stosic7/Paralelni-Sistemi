#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define m 3
#define k 4

int main(int argc, char** argv) {
    static int a[m][k];
    int b[k];
    int rank, p, i, j;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int num_rows = (m + p - 1) / p;
    int nizA[num_rows * k];
    int sendbuf[p * num_rows * k];

    if (rank == 0) {
        for (i = 0; i < m; i++)
            for (j = 0; j < k; j++)
                a[i][j] = i * k + j + 1;

        printf("Matrica A:\n");
        for (i = 0; i < m; i++) {
            for (j = 0; j < k; j++)
                printf("%4d", a[i][j]);
            printf("\n");
        }
        printf("-------------------\n");

        for (i = 0; i < k; i++)
            b[i] = i + 1;

        printf("Vektor b: ");
        for (i = 0; i < k; i++)
            printf("%4d", b[i]);
        printf("\n-------------------\n");

        int idx = 0;
        for (int proc = 0; proc < p; proc++)
            for (int row = proc; row < m; row += p)
                for (j = 0; j < k; j++)
                    sendbuf[idx++] = a[row][j];
        while (idx < p * num_rows * k)
            sendbuf[idx++] = 0;
    }

    MPI_Scatter(sendbuf, num_rows * k, MPI_INT, 
                nizA, num_rows * k, MPI_INT, 
                0, MPI_COMM_WORLD);

    MPI_Bcast(b, k, MPI_INT, 0, MPI_COMM_WORLD);

    int local_result[num_rows];  // Rezultat za lokalne vrste

    for (i = 0; i < num_rows; i++) {
        int global_row = rank + i * p;  // Koja je ovo vrsta globalno?
        
        if (global_row < m) {  // Proveri da li je validna vrsta
            local_result[i] = 0;
            
            // Skalarni proizvod vrste × vektor b
            for (j = 0; j < k; j++) {
                local_result[i] += nizA[i * k + j] * b[j];
            }
            
            printf("Proces %d: vrsta %d × b = %d\n", 
                rank, global_row, local_result[i]);
        }
    }

    // SABERI SVE PARCIJALNE REZULTATE (Gather ili Reduce)
    int final_result[m];

    // Opcija 1: Gather - sakupi sve rezultate na proces 0
    // (Složenije jer je ciklična distribucija)

    // Opcija 2: Allgatherv ili individualni Send/Recv
    if (rank == 0) {
        // Proces 0 već ima svoje rezultate
        for (i = 0; i < num_rows; i++) {
            int global_row = rank + i * p;
            if (global_row < m)
                final_result[global_row] = local_result[i];
        }
        
        // Prima rezultate od drugih procesa
        for (int proc = 1; proc < p; proc++) {
            int proc_rows = (m + proc) / p;  // Koliko vrsta ima taj proces
            int recv_buf[proc_rows];
            
            MPI_Recv(recv_buf, proc_rows, MPI_INT, proc, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Stavi na prava mesta
            for (i = 0; i < proc_rows; i++) {
                int global_row = proc + i * p;
                if (global_row < m)
                    final_result[global_row] = recv_buf[i];
            }
        }
        
        // Ispis finalnog rezultata
        printf("\n========================================\n");
        printf("REZULTAT A × b:\n");
        for (i = 0; i < m; i++)
            printf("rezultat[%d] = %d\n", i, final_result[i]);
        printf("========================================\n");
        
    } else {
        // Ostali procesi šalju svoje rezultate procesu 0
        int actual_rows = 0;
        for (i = 0; i < num_rows; i++) {
            int global_row = rank + i * p;
            if (global_row < m)
                actual_rows++;
        }
        
        MPI_Send(local_result, actual_rows, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }


    MPI_Finalize();
    return 0;
}
