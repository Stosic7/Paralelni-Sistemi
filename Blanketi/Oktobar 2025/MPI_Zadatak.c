#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#define n 3

typedef struct {
    int maticni_broj;
    char ime[50];
    char prezime[50];
    float plata;
} Zaposleni;

int main(int argc, char** argv) {
    // inicijalizacija
    int a[n][n], b[n][n], c[n][n];
    int nizA[n], nizB[n]; // nizA ce da sadrzi vrste matrice A, nizB ce da se sadrzi kolone matrice B
    int rank, p, i, j, irow, jcol;
    MPI_Comm row_comm, col_comm, first_col_comm, first_row_comm; // potrebni komunikatori za vrste/kolone
    MPI_Datatype col_vec, col_type; // resized koristimo da sukcesivno odredimo kolone

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    p = n*n; // iz teksta jer je p = q^2, a q = n

    irow = rank / n; // boja vrste (prva vrsta 0, 0, 0, druga vrsta 1 1 1)
    jcol = rank % n; //  boja kolone (prva kolona 0 0 0 druga kolona 1 1 1)

    MPI_Comm_split(MPI_COMM_WORLD, irow, jcol, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, jcol, irow, &col_comm);

    // pravimo izvedeni tip za kolone jer su kolone udaljene n mesta jedna od druge, pa moramo da imamo izvedeni tip podatka, sto ce kod nas da bude col_type
    MPI_Type_vector(n, 1, n, MPI_INT, &col_vec);
    MPI_Type_create_resized(col_vec, 0, sizeof(int), &col_type);
    MPI_Type_commit(&col_type);
    MPI_Type_free(&col_vec);

    static int A[n][n], B[n][n];
    if (rank == 0) {
        for (i=0;i<n;i++) 
            for (j=0;j<n;j++) { 
                A[i][j]=i+j; 
                B[i][j]=i+j; 
            }
    }

    // pravimo first_col komunikator, znaci krenuce od prve kolone
    MPI_Comm_split(MPI_COMM_WORLD, jcol, irow, &first_col_comm);
    if (first_col_comm != MPI_COMM_NULL) {
        // odradice scutter (deli) elemenata celom redu za sve tri kolone
        // znaci p0 ce imati npr (0,1,2), p3 ce imati npr [3,4,5], p5 [6,7,8]... zavisi od n
        MPI_Scatter(&A[0][0], n, MPI_INT, nizA, n, MPI_INT, 0, first_col_comm);
    }
    MPI_Bcast(nizA, n, MPI_INT, 0, row_comm); // kada imamo u svakoj koloni elemente tog reda, broadcastujemo ih drugim processima u tom redu
    // ako p0 ima (0,1,2) --> to ce broadcastoviti i za p1 (0,1,2) i za p2 (0,1,2)...

    // isto sad samo za redove kojima prosledjujemo kolone matrice b
    MPI_Comm_split(MPI_COMM_WORLD, irow, jcol, &first_row_comm);
    if (first_row_comm != MPI_COMM_NULL) {
        MPI_Scatter(&B[0][0], 1, col_type, nizB, n, MPI_INT, 0, first_row_comm);
    }
    MPI_Bcast(nizB, n, MPI_INT, 0, col_comm);

    // na kraju imamo u svakom procenu nizA koji ima svoju vrstu i u nizB kolonu koja odgovara toj vrsti, njih mnozimo.

    int cij = 0;
    for (int t=0;t<n;t++) 
        cij += nizA[t]*nizB[t];

    int flatC[n*n];
    // Svi procesi salju svoje rezultate natrag glavnom procesu (broj 0) koji sve skuplja i smesta u matricu flatC
    MPI_Gather(&cij, 1, MPI_INT, flatC, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (i=0;i<n;i++) 
            for (j=0;j<n;j++) 
                c[i][j]=flatC[i*n+j];
        for (i=0;i<n;i++) { 
            for (j=0;j<n;j++) 
                printf("%4d", c[i][j]); 
            printf("\n"); 
        }
    }

    MPI_Datatype zaposleni_type;

    // koliko ima polja (4)
    int lengths[4] = {1, 50, 50, 1};

    // Pomak svakog polja u strukturi
    MPI_Aint displacements[4];
    Zaposleni dummy;
    MPI_Get_address(&dummy.maticni_broj, &displacements[0]);
    MPI_Get_address(&dummy.ime, &displacements[1]);
    MPI_Get_address(&dummy.prezime, &displacements[2]);
    MPI_Get_address(&dummy.plata, &displacements[3]);

    // Relativan pomak
    for(int i=n; i>=0; i--) {
        displacements[i] -= displacements[0];
    }

    // Tipovi polja
    MPI_Datatype types[4] = {MPI_INT, MPI_CHAR, MPI_CHAR, MPI_FLOAT};

    // Kreiraj tip
    MPI_Type_create_struct(4, lengths, displacements, types, &zaposleni_type);
    MPI_Type_commit(&zaposleni_type);

    // Gornja trougaona: jcol >= irow
    int color;
    if (jcol >= irow) {
        color = 0;  // Ucestvuje u novom komunikatoru
    } else {
        color = MPI_UNDEFINED;
    }

    MPI_Comm upper_triangle_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &upper_triangle_comm);

    if (upper_triangle_comm != MPI_COMM_NULL) {
    Zaposleni osoba;
    
    // Samo proces 0 u novom komunikatoru inicijalizuje
    int new_rank;
    MPI_Comm_rank(upper_triangle_comm, &new_rank);
    
    if (new_rank == 0) {
        // Inicijalizuj podatke
        osoba.maticni_broj = 12345;
        strcpy(osoba.ime, "Luka");
        strcpy(osoba.prezime, "Stosic");
        osoba.plata = 50000.0;
    }
    
    // Broadcast strukturu svima u gornjo-trougaonom komunikatoru
    MPI_Bcast(&osoba, 1, zaposleni_type, 0, upper_triangle_comm);
    
    // Svaki proces ispisuje
    printf("Proces %d (rang %d u MPI svetu): %s %s, plata: %.2f\n",
           new_rank, rank, osoba.ime, osoba.prezime, osoba.plata);
    }

    MPI_Finalize();
    return 0;
}
