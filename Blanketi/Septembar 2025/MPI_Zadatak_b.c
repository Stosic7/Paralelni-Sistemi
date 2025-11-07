#include <stdio.h>
#include <mpi.h>
#include <limits.h>

#define n 6
#define m 6
#define x 10
#define p 36

int main(int argc, char** argv) {
	int rank, size, irow, jcol, color, key = 1;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int A[n][m];

	irow = rank / n;
	jcol = rank % n;
	int vrsta[m];
	int prVrsta[m];

	int condition = (irow + jcol == n - 1);

	if (condition) {
		color = 0;
	} else {
		color = MPI_UNDEFINED;
	}

	MPI_Comm diag_comm;
	MPI_Comm_split(MPI_COMM_WORLD, color, key, &diag_comm);

	MPI_Datatype row;
	MPI_Type_contiguous(m, MPI_INT, &row);
	MPI_Type_commit(&row);

	if (color != MPI_UNDEFINED) {
		int diagRank, diagSize;
		MPI_Comm_size(diag_comm, &diagSize);
		MPI_Comm_rank(diag_comm, &diagRank);

		if (diagRank == 0) {
			printf("Inicijalna matrica A:\n");
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < m; j++) {
					A[i][j] = (i + j) % 15 + 1;
					printf("%d\t", A[i][j]);
				}
				printf("\n");
			}
			fflush(stdout);
		}

		MPI_Scatter(A, 1, row, vrsta, n, MPI_INT, 0, diag_comm);
		MPI_Barrier(diag_comm);

		int proizvod = 1;
		for (int i = 0; i < m; i++) {
			proizvod *= vrsta[i];
		}

		MPI_Gather(&proizvod, 1, MPI_INT, prVrsta, 1, MPI_INT, 0, diag_comm);

		if (diagRank == 0) {
			printf("PROIZVODI IZ PROCESSA: \n");
			for (int i = 0; i < m; i++) {
				printf("%d, ", prVrsta[i]);
			}
			printf("\n");

			int maxEl = INT_MIN;
			for (int i = 0; i < m; i++) {
				if (prVrsta[i] > maxEl) {
					maxEl = prVrsta[i];
				}
			}
			printf("Maksimalni proizvod: %d\n", maxEl);
		}
	}

	MPI_Finalize();
	return 0;
}
