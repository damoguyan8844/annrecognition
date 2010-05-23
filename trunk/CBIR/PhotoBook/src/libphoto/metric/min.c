#include "photobook.h"
#include "obj_data.h"

/* Functions *****************************************************************/

void MinDistance(Ph_Object self, Ph_Member query,
		 Ph_Member *test, int count)
{
  int m,i, length;
  double *va, *vb, distance;
  struct MinData *data = (struct MinData *)self->data;

  PhGetField(query, data->field, va);
  length = TypeInt(TypeType(
	       Ph_ObjField(query, data->field)->type, 0), 0);

  for(m=0;m < count;m++) {
    PhGetField(test[m], data->field, vb);
    distance = 0.0;
    for(i = 0; i < length; i++) {
      if(va[i] > vb[i]) distance += (va[i] - vb[i]);
    }
    Ph_MemDistance(test[m]) = distance;
  }
}
