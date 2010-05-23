#include "photobook.h"
#include "obj_data.h"
#include <assert.h>
#include <math.h>

/* Globals *******************************************************************/

static char *orien_strings[] = {
  "none", "gorkani", "tamura"
};
#define ORIEN(x) (data->orien_type == orien_strings[x])

typedef struct {
  char harmonic;
  union {
    char *orient;  /* harmonic == 0 && orien_type == gorkani */
    char dir;      /* harmonic == 0 && orien_type == tamura */
    double *peaks; /* harmonic == 1 */
  } data;
} Info;

/* Functions *****************************************************************/

static int GetInfo(struct WoldData *data, Ph_Member member, Info *info)
{
  double *v;
  char *s;

  s = Ph_MemGetAnn(member, "harmonic");
  info->harmonic = s? strcmp(s,"yes")? 0:1:0;

  if(!info->harmonic) {
    if(ORIEN(1)) {
      info->data.orient = Ph_MemGetAnn(member, data->orien_label);
      if(!info->data.orient) {
	fprintf(stderr, "no `%s' defined for `%s'\n",
		data->orien_label, Ph_MemName(member));
	return PH_ERROR;
      }
    }
    else if(ORIEN(2)) {
      if(Ph_ObjGet(member, data->tamura_vector, &v) == PH_ERROR) {
	fprintf(stderr, "Error getting %s field for %s\n",
		data->tamura_vector, Ph_MemName(member));
	return PH_ERROR;
      }
      info->data.dir = v[2] != 1.0;
    }
  }
  else {
    if(Ph_ObjGet(member, data->peaks, &info->data.peaks) == PH_ERROR) {
      fprintf(stderr, "Error getting %s field for %s\n",
	      data->peaks, Ph_MemName(member));
      return PH_ERROR;
    }
  }
  return PH_OK;
}

#define sqr(x) ((x)*(x))

void WoldDistance(Ph_Object self, Ph_Member query, 
		  Ph_Member *test, int count)
{
  int m;
  struct WoldData *data = (struct WoldData *)self->data;
  Info a, b;
  int i, j;
  double total, subtotal;

  if(GetInfo(data, query, &a) == PH_ERROR) return;

  for(m=0;m<count;m++) {
    if(GetInfo(data, test[m], &b) == PH_ERROR) return;
    if(a.harmonic ^ b.harmonic) {
      /* mismatched harmonicity */
      Ph_MemDistance(test[m]) = NOTADISTANCE;
      continue;
    }
    if(!a.harmonic) {
      if( (ORIEN(1) && strcmp(a.data.orient, b.data.orient)) ||
	  (ORIEN(2) && (a.data.dir ^ b.data.dir)) ) {
	Ph_MemDistance(test[m]) = NOTADISTANCE;
	continue;
      }
      /* Get distance from the alt_metric */
      data->alt_distance(data->alt_metric, query, &test[m], 1);
      continue;
    }

    /* Peak matching */
    total = 0.0;
    for(i=0;i < data->num_peaks;i++) {
      double *a_peak = &a.data.peaks[i*3];
      subtotal = 0.0;
      for(j=0;j < data->num_peaks;j++) {
	double *b_peak = &b.data.peaks[j*3];
	int xd, yd;
	double magsum;

	/* try peak */
        xd = (int)fabs(a_peak[0] - b_peak[0]);
        yd = (int)fabs(a_peak[1] - b_peak[1]);
        if((xd < data->nbr_size) &&
           (yd < data->nbr_size)) {
          magsum = a_peak[2] + b_peak[2];
          subtotal += data->nbr->data[xd][yd] * b_peak[2] / sqr(magsum);
        }

        /* try peak reflected through origin */
        /* warning: this assumes image size is 128x128 */
        xd = (int)fabs(a_peak[0] - (128-b_peak[0]));
        yd = (int)fabs(a_peak[1] - (128-b_peak[1]));
        if((xd < data->nbr_size) &&
           (yd < data->nbr_size)) {
          magsum = a_peak[2] + b_peak[2];
          subtotal += data->nbr->data[xd][yd] * b_peak[2] / sqr(magsum);
        }
      }
      total += subtotal * sqr(a_peak[2]);
    }
    Ph_MemDistance(test[m]) = -total;
  }
}

static void watchProc(Ph_Object self, char *field, void *userData)
{
  struct WoldData *data = (struct WoldData *)self->data;
  if(!strcmp(field, "nbr-size")) {
    int i,j;
    /* create the nbr matrix */
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
  else if(!strcmp(field, "alt-metric")) {
    /* look up the metric */
    data->alt_metric = PhLookupObject(self->phandle, data->alt_metric_name);
    if(!data->alt_metric) {
      fprintf(stderr, "No such object: %s\n", data->alt_metric_name);
      return;
    }
    /* cache the alt_metric distance function */
    data->alt_distance = PhObjFunc(data->alt_metric, "distance");
    assert(data->alt_distance);
  }
}

void WoldCon(Ph_Object self)
{
  struct WoldData *data = (struct WoldData *)self->data;
  int i;
  
  /* quark the orien_strings */
  for(i=0;i<2;i++) {
    orien_strings[i] = Ph_StringQuark(orien_strings[i]);
  }

  data->nbr = NULL;
  /* set up watch callbacks */
  Ph_ObjWatch(self, "nbr-size", watchProc, NULL);
  Ph_ObjWatch(self, "alt-metric", watchProc, NULL);

  /* set the default nbr-size */
  i = 2;
  Ph_ObjSet(self, "nbr-size", &i);
}

void WoldDes(Ph_Object self)
{
  struct WoldData *data = (struct WoldData *)self->data;
  MatrixFree(data->nbr);
}
