#include "photobook.h"
#include "obj_data.h"
#include <limits.h>
#include <assert.h>

typedef struct {
  int rank;
  Ph_Member member;
} RankRec;

/* Functions *****************************************************************/

static int Compare(RankRec *a, RankRec *b)
{
  return (Ph_MemDistance(a->member) > Ph_MemDistance(b->member)) ? 1 : -1;
}

/* requires: num_metrics is set */
void RankComboDistance(Ph_Object self, Ph_Member query, 
		       Ph_Member *test, int count)
{
  int m,i;
  int rank;
  double *distance, *weight, cur_dist;
  RankRec *ranks;
  struct RankComboData *data = (struct RankComboData *)self->data;

  ranks = Allocate(count, RankRec);
  for(m=0;m<count;m++) {
    ranks[m].rank = m;
    ranks[m].member = test[m];
  }
  distance = (double*)calloc(count, sizeof(double));

  PhGetField(query, data->weights, weight);

  for(i=0;i < data->num_metrics;i++) {
    if(!data->distfuncs[i]) continue;
    data->distfuncs[i](data->metrics[i], query, test, count);
    /* sort ranks array to get rank ordering */
    qsort((void*)ranks, count,
	  sizeof(RankRec), (CmpFunc*)Compare);
    /* add weighted ranks to distance array */
    rank = 0;
    cur_dist = -10e10;
    for(m=0;m<count;m++) {
      /* only increase rank if the distance has increased */
      if(Ph_MemDistance(ranks[m].member) > cur_dist) {
	rank++;
	cur_dist = Ph_MemDistance(ranks[m].member);
      }
      distance[ranks[m].rank] += weight[i] * rank;
    }
  }

  for(m=0;m < count;m++) Ph_MemDistance(test[m]) = distance[m];
  free(distance);
  free(ranks);
}

static void RefreshArray(Ph_Object obj, char *field)
{
  struct RankComboData *data = (struct RankComboData *)obj->data;
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
  struct RankComboData *data = (struct RankComboData *)self->data;
  if(!strcmp(field, "metrics")) {
    /* convert names to objects */
    int i;
    for(i=0;i<data->num_metrics;i++) {
      Ph_Object obj;
      data->distfuncs[i] = NULL;
      if(!data->metric_names[i][0]) continue;
      obj = PhLookupObject(self->phandle, data->metric_names[i]);
      if(!obj) {
	fprintf(stderr, "No such object: %s\n", data->metric_names[i]);
	continue;
      }
      data->metrics[i] = obj;
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
    RefreshArray(self, "metrics");
  }
}

void RankComboCon(Ph_Object self)
{
  struct RankComboData *data = (struct RankComboData *)self->data;
  data->metric_names = NULL;
  data->metrics = NULL;
  data->distfuncs = NULL;
  /* set up watch callbacks */
  Ph_ObjWatch(self, "num-metrics", watchProc, NULL);
  Ph_ObjWatch(self, "metrics", watchProc, NULL);
}

void RankComboDes(Ph_Object self)
{
  struct RankComboData *data = (struct RankComboData *)self->data;
  if(data->metric_names) free(data->metric_names);
  if(data->metrics) free(data->metrics);
  if(data->distfuncs) free(data->distfuncs);
}
