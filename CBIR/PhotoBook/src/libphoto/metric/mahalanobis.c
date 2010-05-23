#include "photobook.h"
#include "obj_data.h"
#include <math.h>

/* Functions *****************************************************************/

/* requires: vector-size, coeff-field, icovar-field are set */
void MahalDistance(Ph_Object self, Ph_Member query, 
		   Ph_Member *test, int count)
{
  int m,i,j;
  double *va, *vb, *d;
  double *ma;
  double total, subtotal;
  struct MahalData *data = (struct MahalData *)self->data;

  PhGetField(query, data->coeff_field, va);
  PhGetField(query, data->icovar_field, ma);

  for(m=0;m < count;m++) {
    PhGetField(test[m], data->coeff_field, vb);

    /* Compute difference vector d = (va-vb). */
    d = Allocate(data->vector_size, double);
    for(i=0;i < data->vector_size;i++) {
      d[i] = va[i] - vb[i];
    }

    if(data->mask) {
      /* mask d by zeroing elements */
      for(i=0;i < data->vector_size;i++) {
	if(data->mask[i]) d[i] = 0.0;
      }
    }

    total = 0.0;
    /* Compute dT*ma*d */
    for(i=0;i < data->vector_size;i++) {
      if(d[i] == 0.0) continue;
      subtotal = 0.0;
      for(j=0;j < data->vector_size;j++) {
	subtotal += (ma+i*data->vector_size)[j] * d[j];
      }
      total += subtotal * d[i];
    }
    free(d);

    /* this is actually the squared Mahalanobis distance */
    Ph_MemDistance(test[m]) = fabs(total);
  }
}

void MahalCon(Ph_Object self)
{
  struct MahalData *data = (struct MahalData *)self->data;
  data->mask = NULL;
}

void MahalDes(Ph_Object self)
{
  struct MahalData *data = (struct MahalData *)self->data;
  if(data->mask) free(data->mask);
}
