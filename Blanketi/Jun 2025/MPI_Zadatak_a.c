#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define m 4
#define k 5
#define p 2
#define W MPI_COMM_WORLD

int main(int argc, char** argv) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(W, &rank);
	MPI_Comm_size(W, &size);

	int A[m][k], B[k], C[m];
	int nizA[m/p][k];

	if (rank == 0) {
		printf("Inicijalna matrica A:\n");
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < k; j++) {
				A[i][j] = (i + j) % 10;
				printf("%d\t", A[i][j]);
			}
			printf("\n");
		}
		fflush(stdout);
		printf("------------------------------------------------\n");

		printf("Inicijalna vector B:");
		for (int i = 0; i < k; i++) {
			B[i] = i + 1;
			printf("%d ", B[i]);
		}
		printf("\n");
		printf("------------------------------------------------\n");
	}

	MPI_Datatype row_vec, row_type;
	MPI_Type_vector(m / p, k, p * k, MPI_INT, &row_vec);
	MPI_Type_create_resized(row_vec, 0, k*sizeof(int), &row_type);
	MPI_Type_free(&row_vec);
	MPI_Type_commit(&row_type);

	MPI_Scatter(A, 1, row_type, nizA, (m / p) * k, MPI_INT, 0, W);

	printf("Process %d (matrica A): \n", rank);
	for (int i = 0; i < m/p; i++) {
		for (int j = 0; j < k; j++) {
			printf("%d\t", nizA[i][j]);
		}
		printf("\n");
	}
	printf("------------------------------------------------\n");

	MPI_Bcast(B, k, MPI_INT, 0, W);

	int parcijalni_C[m / p];
	for (int i = 0; i < m / p; i++) {
		parcijalni_C[i] = 0;
		for (int j = 0; j < k; j++) {
			parcijalni_C[i] += nizA[i][j] * B[j];
		}
	}

	MPI_Gather(parcijalni_C, m / p, MPI_INT, C, m / p, MPI_INT, 0, W);

	if (rank == 0) {
		printf("\nVektor C = A * B:\n");
		for (int i = 0; i < m; i++) {
			printf("%d ", C[i]);
		}
		printf("\n");
	}

	MPI_Finalize();
	return 0;
}
