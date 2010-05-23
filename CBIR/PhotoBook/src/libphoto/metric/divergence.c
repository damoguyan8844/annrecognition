#include "photobook.h"
#include "obj_data.h"
#include <math.h>

/* Functions *****************************************************************/

/* requires: vector-size, coeff-field, covar-field, icovar-field are set */
void DiverDistance(Ph_Object self, Ph_Member query, 
		   Ph_Member *test, int count)
{
  int m,i,j;
  double *va, *vb, *d;
  double *ca, *cb, *ica, *icb, *k;
  double total, subtotal;
  struct DiverData *data = (struct DiverData *)self->data;
  struct MahalData *mdata = (struct MahalData *)self->super->data;

  PhGetField(query, mdata->coeff_field, va);
  PhGetField(query, mdata->icovar_field, ica);
  PhGetField(query, data->covar_field, ca);

  for(m=0;m < count;m++) {
    PhGetField(test[m], mdata->coeff_field, vb);
    PhGetField(test[m], mdata->icovar_field, icb);
    PhGetField(test[m], data->covar_field, cb);

    total = 0.0;
    /* Compute trace term */
    for(i=0;i < mdata->vector_size * mdata->vector_size;i++) {
      total += ca[i] * icb[i];
      total += ica[i] * cb[i];
    }
    total -= 2*mdata->vector_size;

    /* Compute difference vector d = (va-vb). */
    d = Allocate(mdata->vector_size, double);
    for(i=0;i < mdata->vector_size;i++) {
      d[i] = va[i] - vb[i];
    }

    /* Compute combined covariance k = ica + icb */
    k = Allocate(mdata->vector_size * mdata->vector_size, double);
    for(i=0;i < mdata->vector_size * mdata->vector_size;i++) {
      k[i] = ica[i] + icb[i];
    }

    /* Compute dT*k*d */
    for(i=0;i < mdata->vector_size;i++) {
      subtotal = 0.0;
      for(j=0;j < mdata->vector_size;j++) {
	subtotal += (k+i*mdata->vector_size)[j] * d[j];
      }
      total += subtotal * d[i];
    }
    total /= 2;
    free(k);
    free(d);

    Ph_MemDistance(test[m]) = fabs(total);
  }
}
