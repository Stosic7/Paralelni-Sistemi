#include <stdio.h>
#include <mpi.h>

#define n 3
#define p 9

int main(int argc, char** argv) {
	int rank, size, irow, jcol;
	int A[n][n], B[n][n], C[n][n];

	int nizA[n], nizB[n];
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	irow = rank / n;
	jcol = rank % n;
	
	MPI_Comm row_comm, col_comm, first_col_comm, first_row_comm;
	MPI_Comm_split(MPI_COMM_WORLD, irow, jcol, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, jcol, irow, &col_comm);

	if (rank == 0) {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				A[i][j] = (i + j) % 10;
				B[i][j] = ((i * i + j * j) + 1) % 20;
			}
		}

		printf("Matrica A: \n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				printf("%d\t", A[i][j]);
			}
			printf("\n");
		}
		printf("-------------------\n");

		printf("Matrica B: \n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				printf("%d\t", B[i][j]);
			}
			printf("\n");
		}
		printf("-------------------\n");
	}

	MPI_Datatype col_vec, col_type;
	MPI_Type_vector(n, 1, n, MPI_INT, &col_vec);
	MPI_Type_create_resized(col_vec, 0, sizeof(int), &col_type);
	MPI_Type_free(&col_vec);
	MPI_Type_commit(&col_type);

	MPI_Comm_split(MPI_COMM_WORLD, jcol, irow, &first_col_comm);
	if (first_col_comm != MPI_COMM_NULL) {
		MPI_Scatter(A, n, MPI_INT, nizA, n, MPI_INT, 0, first_col_comm);
	}
	MPI_Bcast(nizA, n, MPI_INT, 0, row_comm);

	MPI_Comm_split(MPI_COMM_WORLD, irow, jcol, &first_row_comm);
	if (first_row_comm != MPI_COMM_NULL) {
		MPI_Scatter(B, 1, col_type, nizB, n, MPI_INT, 0, first_row_comm);
	}
	MPI_Bcast(nizB, n, MPI_INT, 0, col_comm);

	int cilj = 0;
	for (int t = 0; t < n; t++) {
		cilj += nizA[t] * nizB[t];
	}

	int flatC[n * n];

	MPI_Gather(&cilj, 1, MPI_INT, flatC, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				C[i][j] = flatC[i * n + j];
			}
		}
		printf("Matrica C: \n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				printf("%d\t", C[i][j]);
			}
			printf("\n");
		}
	}

	MPI_Finalize();
	return 0;
}
