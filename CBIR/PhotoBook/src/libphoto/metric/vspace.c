#include "photobook.h"
#include "obj_data.h"
#include <math.h>

/* Functions *****************************************************************/

/* computes Euclidean distance from c to b*b'*c, 
 * i.e. the sum of squared entries in (I-b*b')*c.
 * v is a work array, of size rows by 1.
 * designed for matrices with rows >> cols.
 */
static double vs_dist(int rows, int cols, double *c, double *b, double *v)
{
  int i,j,k;
  double dot, d = 0.0;
  for(i=0;i<cols;i++) {
    for(k=0;k<rows;k++) v[k] = (c+k*cols)[i];
    for(j=0;j<cols;j++) {
      dot = 0.0;
      for(k=0;k<rows;k++) dot += (b+k*cols)[j] * (c+k*cols)[i];
      for(k=0;k<rows;k++) v[k] -= dot * (b+k*cols)[j];
    }
    for(k=0;k<rows;k++) d += v[k] * v[k];
  }
  return d;
}

/* requires: everything known */
void VSpaceDistance(Ph_Object self, Ph_Member query, 
		    Ph_Member *test, int count)
{
  int m,i,j;
  double *c1, *b1, *c2, *b2, *v;
  double d1, d2;
  struct VSpaceData *data = (struct VSpaceData *)self->data;

  PhGetField(query, data->corr_field, c1);
  PhGetField(query, data->basis_field, b1);
  v = Allocate(data->rows, double);

  for(m=0;m < count;m++) {
    PhGetField(test[m], data->corr_field, c2);
    PhGetField(test[m], data->basis_field, b2);

    /* compute distance both ways, take minimum */
    d1 = vs_dist(data->rows, data->cols, c1, b2, v);
    d2 = vs_dist(data->rows, data->cols, c2, b1, v);

    Ph_MemDistance(test[m]) = min(d1, d2);
  }
  free(v);
}
