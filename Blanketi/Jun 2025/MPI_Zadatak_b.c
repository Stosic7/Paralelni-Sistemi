#include <stdio.h>
#include <mpi.h>
#include <limits.h>

#define n 6
#define m 6
#define x 5

int main(int argc, char** argv) {
	int rank, size;
	int key = 1;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int A[n][m];
	int irow, jcol;
	int kolona[n];
	int primljeneSume[n];
	int l = 0;

	irow = rank / n;
	jcol = rank % n;

	int condition = (rank % x == 0);
	int color;

	if (condition) {
		color = 1;
	} else {
		color = MPI_UNDEFINED;
	}

	MPI_Comm div_comm;
	MPI_Comm_split(MPI_COMM_WORLD, color, key, &div_comm);

	MPI_Datatype col_row, col_vec;
	MPI_Type_vector(n, 1, m, MPI_INT, &col_row);
	MPI_Type_create_resized(col_row, 0, sizeof(int), &col_vec);
	MPI_Type_free(&col_row);
	MPI_Type_commit(&col_vec);

	if (color != MPI_UNDEFINED) {
		int divRank, divSize;
		MPI_Comm_rank(div_comm, &divRank);
		MPI_Comm_size(div_comm, &divSize);

		if (divrank == 0) {
			for (int i = 0; i < n; i++)
				for (int j = 0; j < m; j++)
					A[i][j] = (i + j) % 10;
			printf("Inicijalna matrica A:\n");
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < m; j++) printf("%d\t", A[i][j]);
				printf("\n");
			}
			fflush(stdout);
		}

		MPI_Scatter(A, 1, col_vec, kolona, n, MPI_INT, 0, div_comm);
		MPI_Barrier(div_comm);

		printf("\n[divRank %d] kolona:\n", divRank);
		for (int i = 0; i < n; i++) printf("%d ", kolona[i]);
		printf("\n");
		fflush(stdout);

		int suma = 0;
		for (int i = 0; i < n; i++) suma += kolona[i];
		
		int sumsBuf[m*n];

		MPI_Gather(&suma, 1, MPI_INT, sumsBuf, 1, MPI_INT, 0, div_comm);

		if (divRank == 0) {
			int minEl = INT_MAX;
			for (int i = 0; i < divSize; i++)
				if (sumsBuf[i] < minEl) minEl = sumsBuf[i];
			printf("Minimalna suma kolone: %d\n", minEl);
		}
	}

	MPI_Type_free(&col_vec);

	MPI_Finalize();
	return 0;

}
