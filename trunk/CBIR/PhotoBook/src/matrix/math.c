/* This functions are distinguished by requiring libm.a */

#include <math.h>
#include "matrix.h"

/* Prototypes ****************************************************************/
void MatrixMagnitude(Matrix A, Matrix B);

/* Functions *****************************************************************/

/* Modifies A to be the magnitude of the complex matrix A + Bi. 
 */
void MatrixMagnitude(Matrix A, Matrix B)
{
  int i;
  for(i=0;i < A->height*A->width;i++) {
    A->data[0][i] = hypot(A->data[0][i], B->data[0][i]);
  }
}

