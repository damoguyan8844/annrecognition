#include "photobook.h"
#include "obj_data.h"
#include <math.h>
#include <assert.h>

/* Functions *****************************************************************/

/* requires: num_metrics is set */
void CombinationDistance(Ph_Object self, Ph_Member query, 
		       Ph_Member *test, int count)
{
  int m,i;
  double *distances;
  struct CombinationData *data = (struct CombinationData *)self->data;

  distances = Allocate(count, double);
  for(m=0;m < count;m++) distances[m] = 0.0;
  for(i=0;i < data->num_metrics;i++) {
    if(!data->distfuncs[i]) continue;
    data->distfuncs[i](data->metrics[i], query, test, count);
    for(m=0;m < count;m++) {
      distances[m] += 
	Ph_MemDistance(test[m]) * data->factors[i] * data->weights[i];
    }
  }
  for(m=0;m < count;m++) Ph_MemDistance(test[m]) = distances[m];
  free(distances);
}

static void RefreshArray(Ph_Object obj, char *field)
{
  struct CombinationData *data = (struct CombinationData *)obj->data;
  void **dptr;
  ObjField *of = Ph_ObjField(obj, field);
  Type t1, t2;
  int i;

  /* bind the type */
  t1 = of->type;
  t2 = TypeType(t1,0);
  t1 = _TypeInt(t2,0);
  TypeReplace(t1, TypeCreate(TYPE_INT, data->num_metrics));
  TypeComputeSize(t2);

  /* reallocate and initialize */
  dptr = (void**)of->data;
  if(*dptr) free(*dptr);
  *dptr = Allocate(TypeSize(t2), char);
  if(!strcmp(TypeClass(TypeType(t2,0)), "double")) {
    for(i=0;i<data->num_metrics;i++) ((double*)(*dptr))[i] = 1.0;
  }
  else {
    for(i=0;i<data->num_metrics;i++) ((void**)(*dptr))[i] = "";
  }

  /* do a set to call the callbacks */
  Ph_ObjSet(obj, field, of->data);
}

static void watchProc(Ph_Object self, char *field, void *userData)
{
  struct CombinationData *data = (struct CombinationData *)self->data;
  if(!strcmp(field, "metrics")) {
    /* convert names to objects */
    int i;
    for(i=0;i<data->num_metrics;i++) {
      data->distfuncs[i] = NULL;
      if(!data->metric_names[i][0]) continue;
      data->metrics[i] = PhLookupObject(self->phandle, data->metric_names[i]);
      if(!data->metrics[i]) {
	fprintf(stderr, "No such object: %s\n", data->metric_names[i]);
	continue;
      }
      /* cache the metric distance function */
      data->distfuncs[i] = PhObjFunc(data->metrics[i], "distance");
      assert(data->distfuncs[i]);
    }
  }
  else if(!strcmp(field, "num-metrics")) {
    if(data->metrics) free(data->metrics);
    data->metrics = Allocate(data->num_metrics, Ph_Object);
    if(data->distfuncs) free(data->distfuncs);
    data->distfuncs = Allocate(data->num_metrics, PhDistFunc*);
    RefreshArray(self, "factors");
    RefreshArray(self, "weights");
    RefreshArray(self, "metrics");
  }
}

void CombinationCon(Ph_Object self)
{
  struct CombinationData *data = (struct CombinationData *)self->data;
  data->metric_names = NULL;
  data->factors = NULL;
  data->weights = NULL;
  data->metrics = NULL;
  data->distfuncs = NULL;
  /* set up watch callbacks */
  Ph_ObjWatch(self, "num-metrics", watchProc, NULL);
  Ph_ObjWatch(self, "metrics", watchProc, NULL);
}

void CombinationDes(Ph_Object self)
{
  struct CombinationData *data = (struct CombinationData *)self->data;
  if(data->metric_names) free(data->metric_names);
  if(data->factors) free(data->factors);
  if(data->weights) free(data->weights);
  if(data->metrics) free(data->metrics);
  if(data->distfuncs) free(data->distfuncs);
}
