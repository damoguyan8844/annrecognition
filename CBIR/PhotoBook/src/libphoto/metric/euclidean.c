#include "photobook.h"
#include "obj_data.h"
#include <math.h>

/* Functions *****************************************************************/

/* requires: vector-size, field are set */
void EuclideanDistance(Ph_Object self, Ph_Member query, 
		       Ph_Member *test, int count)
{
  int m,i;
  double *va, *vb, distance;
  struct EuclideanData *data = (struct EuclideanData *)self->data;

  PhGetField(query, data->field, va);
  for(m=0;m < count;m++) {
    PhGetField(test[m], data->field, vb);
    distance = 0.0;
    for(i = data->from; i <= data->to; i++) {
      double d = va[i] - vb[i];
      d /= data->weights[i];
      distance += d*d;
    }
    Ph_MemDistance(test[m]) = sqrt(distance);
  }
}

static void watchProc(Ph_Object self, char *field, void *userData)
{
  struct EuclideanData *data = (struct EuclideanData *)self->data;
  /* no bounds checking on "from" and "to" */
  if(!strcmp(field, "from")) {
    if(data->from > data->to) Ph_ObjSet(self, "to", &data->from);
  }
  else if(!strcmp(field, "to")) {
    if(data->to < data->from) Ph_ObjSet(self, "from", &data->to);
  }
  else if(!strcmp(field, "vector-size")) {
    int i;
    Type t1, t2;

    /* reallocate weights to all 1.0 */
    if(data->weights) free(data->weights);
    data->weights = Allocate(data->vector_size, double);
    for(i=0;i<data->vector_size;i++) data->weights[i] = 1.0;
    /* bind the weight type */
    t1 = Ph_ObjField(self, "weights")->type;
    t2 = TypeType(t1,0);
    t1 = _TypeInt(t2,0);
    TypeReplace(t1, TypeCreate(TYPE_INT, data->vector_size));
    TypeComputeSize(t2);
    /* do a set to call the callbacks */
    Ph_ObjSet(self, "weights", &data->weights);

    /* set from = 0, to = vecsize-1 */
    i = 0;
    Ph_ObjSet(self, "from", &i);
    i = data->vector_size - 1;
    Ph_ObjSet(self, "to", &i);
    
    /* bind the vector type */
    t2 = TypeType(data->type,0);
    t1 = _TypeInt(t2,0);
    TypeReplace(t1, TypeCreate(TYPE_INT, data->vector_size));
    TypeComputeSize(t2);
  }
}

void EuclideanCon(Ph_Object self)
{
  struct EuclideanData *data = (struct EuclideanData *)self->data;
  data->weights = NULL;
  data->type = TypeParse("ptr array[?x] double");
  /* set up watch callbacks */
  Ph_ObjWatch(self, "vector-size", watchProc, NULL);
  Ph_ObjWatch(self, "from", watchProc, NULL);
  Ph_ObjWatch(self, "to", watchProc, NULL);
  Ph_ObjWatch(self, "field", watchProc, NULL);
}

void EuclideanDes(Ph_Object self)
{
  struct EuclideanData *data = (struct EuclideanData *)self->data;
  if(data->weights) free(data->weights);
  TypeFree(data->type);
}
