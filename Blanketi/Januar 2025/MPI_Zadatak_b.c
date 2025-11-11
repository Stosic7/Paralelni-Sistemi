#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define n 5
#define p 25

int main(int argc, char** argv) {
	int rank, size, irow, jcol, color, key = 1;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	irow = rank / n;
	jcol = rank % n;

	int condition = (irow == jcol);
	color = condition ? 0 : MPI_UNDEFINED;

	MPI_Comm diag;
	MPI_Comm_split(MPI_COMM_WORLD, color, key, &diag);

	int A[n][n];
	int rows[n];

	if (color != MPI_UNDEFINED) {
		int diagRank, diagSize;
		MPI_Comm_rank(diag, &diagRank);
		MPI_Comm_size(diag, &diagSize);

		if (diagRank == 0) {
			printf("Inicijlizovana matrica A: \n");
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					A[i][j] = ((i * i + j * j) % 20) * (j * 5 + i % 2);
					printf("%d\t", A[i][j]);
				}
				printf("\n");
			}
			printf("----------------------------------------\n");
		}

		MPI_Scatter(A, n, MPI_INT, rows, n, MPI_INT, 0, diag);
		MPI_Barrier(diag);

		printf("Process ID: %d u novom komunikatoru ima vrstu: ", diagRank);
		for (int i = 0; i < n; i++) {
			printf("%2d, ", rows[i]);
		}
		printf("\n----------------------------------------------------------------------------------------------\n");

		int suma = 0;
		for (int i = 0; i < n; i++) {
			suma += rows[i];
		}
		
		int* sume = NULL;
		if (diagRank == 0) {
			sume = (int*)malloc(diagSize * sizeof(int));
		}
		MPI_Gather(&suma, 1, MPI_INT, sume, 1, MPI_INT, 0, diag);

		if (diagRank == 0) {
			printf("\n=== FINALNI REZULTATI ===\n");
			int ukupno = 0;
			for (int i = 0; i < diagSize; i++) {
				printf("Dijagonalni proces %d: suma = %d\n", i, sume[i]);
				ukupno += sume[i];
			}
			printf("Ukupan zbir dijagonale: %d\n", ukupno);
			free(sume);
		}
	}

	MPI_Finalize();
	return 0;
}
