typedef double Real;
typedef struct MatrixStruct {
  int width, height;
  Real **data;
} *Matrix;

Matrix MatrixCreate(int height, int width);
void MatrixFree(Matrix matrix);
