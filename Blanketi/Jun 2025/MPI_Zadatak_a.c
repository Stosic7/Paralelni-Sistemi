#include <stdio.h>
#include <mpi.h>

#define m 6
#define k 6
#define p 3

int main(int argc, char** argv) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int A[m][k], B[k], C[m];
	int nizA[m][k / p];

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

		printf("Inicijalna vector B:");
		for (int i = 0; i < k; i++) {
			B[i] = i + 1;
			printf("%d ", B[i]);
		}
		printf("\n");
	}

	MPI_Datatype row_vec, row_type;
	MPI_Type_vector(k / p * m, 1, p, MPI_INT, &row_vec);
	MPI_Type_create_resized(row_vec, 0, sizeof(int), &row_type);
	MPI_Type_free(&row_vec);
	MPI_Type_commit(&row_type);

	MPI_Scatter(A, 1, row_type, nizA, m * k / p, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	printf("\nProces %d local columns:\n", rank);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < k / p; j++) {
			printf("%d\t", nizA[i][j]);
		}
		printf("\n");
	}
	printf("\n-----------------------------");

	MPI_Bcast(B, k, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	printf("\nProces %d local vector: ", rank);
	for (int i = 0; i < k; i++)
		printf("%d ", B[i]);
	printf("\n");

	int parcijalniRezultat[m];

	for (int i = 0; i < m; i++) {
		int s = 0;
		for (int j = 0; j < k/p; j++) {
			int jglob = j * p + rank;
			s += nizA[i][j] * B[jglob];
		}
		parcijalniRezultat[i] = s;
	}

	MPI_Reduce(parcijalniRezultat, C, m, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	printf("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$");
	if (rank == 0) {
		printf("\nC: ");
		for (int i = 0; i < m; i++)
			printf("%d ", C[i]);
	}
	fflush(stdout);

	MPI_Finalize();
	return 0;
}
