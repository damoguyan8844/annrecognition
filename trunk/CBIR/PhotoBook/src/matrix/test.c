#include <stdio.h>
#include "matrix.h"

main()
{
  Matrix A, T;
  Real *B;

  A = MatrixCreate(2,2);
  A->data[0][0] = 1.0;
  A->data[0][1] = 2.0;
  A->data[1][0] = 3.0;
  A->data[1][1] = 4.0;
  printf("A:\n");
  MatrixWrite(A, stdout);
  printf("Determinant: %g\n", MatrixDeterminant(A));

  B = Allocate(2, Real);
  B[0] = 1.0;
  B[1] = 2.0;
  printf("B: %g %g\n", B[0], B[1]);
  MatrixSolve(A, B);
  printf("Solve(A, B): %g %g\n", B[0], B[1]);

  T = MatrixCopy(A);
  MatrixInvert(T);
  printf("Invert(A):\n");
  MatrixWrite(T, stdout);
  MatrixFree(T);

  A->data[0][0] = 2.0;
  A->data[0][1] = 1.0;
  A->data[1][0] = 1.0;
  A->data[1][1] = 3.0;
  printf("A:\n");
  MatrixWrite(A, stdout);
  MatrixEigenvalues(A, B, NULL);
  printf("Eigenvalues(A): %g %g\n", B[0], B[1]);
  MatrixCholesky(A);
  printf("Cholesky(A):\n");
  MatrixWrite(A, stdout);

  MatrixFree(A);
  free(B);
}
