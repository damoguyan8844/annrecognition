#include "photobook.h"
#include "obj_data.h"
#include <assert.h>
#include <math.h>

/* Functions *****************************************************************/

#define sqr(x) ((x)*(x))

double PeakMatch(int npeaks, double *pa, double *pb, 
		 int nbr_size, double **nbr)
{
  int i,j;
  double total, subtotal;

  total = 0.0;
  for(i=0;i < npeaks;i++) {
    double *a_peak = pa + 2 + i*3;
    if(a_peak[2] == 0) continue;
    subtotal = 0.0;
    for(j=0;j < npeaks;j++) {
      double *b_peak = pb + 2 + j*3;
      int xd, yd;
      double magsum;
      
      /* ignore zero-magnitude peaks */
      if(b_peak[2] == 0) continue;
      /* try peak */
      xd = (int)fabs(a_peak[0] - b_peak[0]);
      yd = (int)fabs(a_peak[1] - b_peak[1]);
      if((xd < nbr_size) &&
	 (yd < nbr_size)) {
	magsum = a_peak[2] + b_peak[2];
	subtotal += nbr[xd][yd] * b_peak[2] / sqr(magsum);
      }
      
      /* try peak reflected through origin */
      xd = (int)fabs(a_peak[0] - (pb[0] - b_peak[0]));
      yd = (int)fabs(a_peak[1] - (pb[1] - b_peak[1]));
      if((xd < nbr_size) &&
	 (yd < nbr_size)) {
	magsum = a_peak[2] + b_peak[2];
	subtotal += nbr[xd][yd] * b_peak[2] / sqr(magsum);
      }
    }
    total += subtotal * sqr(a_peak[2]);
  }
  return total;
}

void PeaksDistance(Ph_Object self, Ph_Member query, 
		   Ph_Member *test, int count)
{
  int m,i,j;
  double *pa, *pb;
  double total, subtotal;
  struct PeaksData *data = (struct PeaksData *)self->data;

  PhGetField(query, data->peaks, pa);

  for(m=0;m<count;m++) {
    PhGetField(test[m], data->peaks, pb);

    /* bogus peaks? */
    if((pa[0] < 0) || (pb[0] < 0)) {
      Ph_MemDistance(test[m]) = 1;
      continue;
    }

    Ph_MemDistance(test[m]) = -PeakMatch(data->num_peaks, pa, pb, 
					 data->nbr_size, data->nbr->data);
  }
}

static void watchProc(Ph_Object self, char *field, void *userData)
{
  struct PeaksData *data = (struct PeaksData *)self->data;
  if(!strcmp(field, "nbr-size")) {
    int i,j;
    /* create the neighborhood matrix */
    if(data->nbr) MatrixFree(data->nbr);
    data->nbr = MatrixCreate(data->nbr_size, data->nbr_size);
    for(i=0;i < data->nbr_size;i++) {
      for(j=0;j < data->nbr_size;j++) {
	if(!i && !j) 
	  data->nbr->data[i][j] = 1.0;
	else
	  data->nbr->data[i][j] = 1/sqrt((double)i+j);
      }
    }
  }
}

void PeaksCon(Ph_Object self)
{
  struct PeaksData *data = (struct PeaksData *)self->data;
  int i;
  
  data->nbr = NULL;
  /* set up watch callbacks */
  Ph_ObjWatch(self, "nbr-size", watchProc, NULL);

  /* set the default nbr-size */
  i = 2;
  Ph_ObjSet(self, "nbr-size", &i);
}

void PeaksDes(Ph_Object self)
{
  struct PeaksData *data = (struct PeaksData *)self->data;
  MatrixFree(data->nbr);
}
