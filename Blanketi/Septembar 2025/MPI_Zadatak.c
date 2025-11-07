#include <stdio.h>
#include <mpi.h>
#include <limits.h>

#define n 3
#define m 4
#define k 2
#define p 2

int main(int argc, char** argv) {
	int rank, size;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int A[n][k], B[k][m], C[n][m];
	int nizA[n][k / p];
	int nizB[k / p][m];

	int minVrste[n];
	int l = 0;

	if (rank == 0) {
		printf("Inicijalna matrica A:\n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < k; j++) {
				A[i][j] = (i*j+1+(i*i) + j * j * (i + 1*j)) % 10;
				printf("%d\t", A[i][j]);
			}
			printf("\n");
		}
		fflush(stdout);

		for (int i = 0; i < n; i++) {
			int minEl = INT_MAX;
			for (int j = 0; j < k; j++) {
				if (A[i][j] < minEl) {
					minEl = A[i][j];
				}
			}
			minVrste[l++] = minEl;
		}

		printf("Minimalni elementi svake vrste matrice A: \n");
		for (int i = 0; i < n; i++) {
			printf("%2d", minVrste[i]);
		}
		printf("\n");

		printf("Inicijalna matrica B:\n");
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < m; j++) {
				B[i][j] = (i*j+1 + j*j*(i+1)) % 20;
				printf("%d\t", B[i][j]);
			}
			printf("\n");
		}
		fflush(stdout);
	}

	MPI_Datatype col_vec, col_type;
	MPI_Type_vector(n, k / p, k, MPI_INT, &col_vec);
	MPI_Type_create_resized(col_vec, 0, (k / p) * sizeof(int), &col_type);
	MPI_Type_free(&col_vec);
	MPI_Type_commit(&col_type);

	MPI_Scatter(A, 1, col_type, nizA, n * (k / p), MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	printf("\nProces %d (matrix A):\n", rank);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k / p; j++) {
			printf("%d\t", nizA[i][j]);
		}
		printf("\n");
	}
	printf("\n-----------------------------");

	MPI_Datatype row_vec;
	MPI_Type_vector(k / p, m, m, MPI_INT, &row_vec);
	MPI_Type_commit(&row_vec);

	MPI_Scatter(B, 1, row_vec, nizB, (k / p) * m, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	printf("\nProces %d (matrix B):\n", rank);
	for (int i = 0; i < k/p; i++) {
		for (int j = 0; j < m; j++) {
			printf("%d\t", nizB[i][j]);
		}
		printf("\n");
	}
	printf("\n-----------------------------");

	int parcijalno_c[n][m];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			parcijalno_c[i][j] = 0;
			for (int t = 0; t < k / p; t++) {
				parcijalno_c[i][j] += nizA[i][t] * nizB[t][j];
			}
		}
	}

	MPI_Reduce(parcijalno_c, C, n * m, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	printf("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$");
	if (rank == 0) {
		printf("\nC: \n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				printf("%d\t", C[i][j]);
			}
			printf("\n");
		}
	}

	MPI_Finalize();
	return 0;
}
