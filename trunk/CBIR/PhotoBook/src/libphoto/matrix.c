#include <tpm/tpm.h>
#include "matrix.h"

Matrix MatrixCreate(int height, int width)
{
  Matrix result;
  int i;

  result = Allocate(1, struct MatrixStruct);
  result->width = width;
  result->height = height;
  result->data = Allocate(height, Real*);
  result->data[0] = Allocate(width*height, Real);
  if(!result->data[0]) {
    fprintf(stderr, 
            "Cannot allocate %d byte matrix\n", width*height*sizeof(Real));
    exit(1);
  }
  for(i=1;i<height;i++) {
    result->data[i] = &result->data[0][i*width];
  }
  return result;
}

void MatrixFree(Matrix matrix)
{
  free(matrix->data[0]);
  free(matrix->data);
  free(matrix);
}
