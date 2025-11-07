#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <stddef.h>

#define n 3
#define p 9

typedef struct {
    int maticni_broj;
    char ime[20];
    char prezime[20];
    double prosecna_plata;
} Zaposleni;

int main(int argc, char** argv) {
    int rank, size, irow, jcol;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    irow = rank / n;
    jcol = rank % n;

    int condition = (irow > jcol);
    int color = condition ? 0 : MPI_UNDEFINED;

    MPI_Comm lower_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &lower_comm);

    MPI_Datatype zaposleni_type;
    int blocklengths[4] = { 1, 20, 20, 1 };
    MPI_Aint displacements[4];
    MPI_Datatype types[4] = { MPI_INT, MPI_CHAR, MPI_CHAR, MPI_DOUBLE };

    displacements[0] = offsetof(Zaposleni, maticni_broj);
    displacements[1] = offsetof(Zaposleni, ime);
    displacements[2] = offsetof(Zaposleni, prezime);
    displacements[3] = offsetof(Zaposleni, prosecna_plata);

    MPI_Type_create_struct(4, blocklengths, displacements, types, &zaposleni_type);
    MPI_Type_commit(&zaposleni_type);

    if (color != MPI_UNDEFINED) {
        int lower_rank, lower_size;
        MPI_Comm_rank(lower_comm, &lower_rank);
        MPI_Comm_size(lower_comm, &lower_size);

        Zaposleni svi_zaposleni[p];
        if (lower_rank == 0) {
            for (int i = 0; i < lower_size; i++) {
                svi_zaposleni[i].maticni_broj = 2000 + i;
                sprintf(svi_zaposleni[i].ime, "Ime%d", i);
                sprintf(svi_zaposleni[i].prezime, "Prezime%d", i);
                svi_zaposleni[i].prosecna_plata = 45000.0 + i * 3000;
            }

            printf("Procesi u donjoj trougaonoj matrici (bez dijagonale):\n");
            printf("lower_size = %d procesa\n", lower_size);
        }

        Zaposleni moj_zaposleni;
        MPI_Scatter(svi_zaposleni, 1, zaposleni_type,
                    &moj_zaposleni, 1, zaposleni_type,
                    0, lower_comm);

        printf("Proces rank=%d (irow=%d, jcol=%d, lower_rank=%d): MB=%d, %s %s, Plata=%.2f\n",
               rank, irow, jcol, lower_rank,
               moj_zaposleni.maticni_broj,
               moj_zaposleni.ime,
               moj_zaposleni.prezime,
               moj_zaposleni.prosecna_plata);

        Zaposleni grupa_zaposleni[p];
        MPI_Gather(&moj_zaposleni, 1, zaposleni_type,
                   grupa_zaposleni, 1, zaposleni_type,
                   0, lower_comm);

        if (lower_rank == 0) {
            printf("\n--- Svi zaposleni u donjoj trougaonoj grupi ---\n");
            for (int i = 0; i < lower_size; i++) {
                printf("  MB: %d, %s %s, Plata: %.2f\n",
                       grupa_zaposleni[i].maticni_broj,
                       grupa_zaposleni[i].ime,
                       grupa_zaposleni[i].prezime,
                       grupa_zaposleni[i].prosecna_plata);
            }
        }
    }

    MPI_Type_free(&zaposleni_type);
    MPI_Finalize();
    return 0;
}
