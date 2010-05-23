#include "photobook.h"
#include "obj_data.h"
#include <math.h> /* for pow() */

/* Prototypes ****************************************************************/

void TswCutTree(double *v, int levels, float cutoff);
static void Highest(double *v, int length, int keep);

/* Functions *****************************************************************/

/* requires: vector-size */
void TswDistance(Ph_Object self, Ph_Member query,
		 Ph_Member *test, int count)
{
  int m,i, vector_size;
  double *va, *vb, distance;
  struct TswData *data = (struct TswData *)self->data;

  vector_size = (int)((pow(4.0, (double)data->levels)-1)/3);

  PhGetField(query, data->field, va);
  va = ArrayCopy(va, vector_size, sizeof(double));
  TswCutTree(va, data->levels, data->cutoff);
  Highest(va, vector_size, data->keep);

/*
  for(i=0;i<85;i++) {
    printf("%g ", va[i]);
  }
  printf("\n");
*/

  for(m=0;m < count;m++) {
    PhGetField(test[m], data->field, vb);
    vb = ArrayCopy(vb, vector_size, sizeof(double));
    TswCutTree(vb, data->levels, data->cutoff);

    distance = 0.0;
    for(i=0;i<vector_size;i++) {
      double d;
      if(va[i] == 0.0) continue;
      if(vb[i] == 0.0) {
	distance += 1e6;  /* penalize for not having the node */
	continue;
      }
      d = va[i] - vb[i];
      distance += d*d;
    }

    free(vb);
    Ph_MemDistance(test[m]) = distance;
  }
  free(va);
}

void TswCon(Ph_Object self)
{
  struct TswData *data = (struct TswData *)self->data;
  data->cutoff = 0.3;
  data->keep = 5;
  data->levels = 4;
}

/* Modifies v so that all elements with value below the keep-th
 * highest value are zero.
 */
static void Highest(double *v, int length, int keep)
{
  double old_max, cur_max;
  int i;

  /* Find the keep-th highest value. */
  old_max = 1e10;
  for(;keep;keep--) {
    /* Find the maximum of all values < old_max */
    cur_max = 0.0;
    for(i=0;i<length;i++) {
      if(v[i] >= old_max) continue;
      if(v[i] > cur_max) cur_max = v[i];
    }
    old_max = cur_max;
  }
  /* Set all elements below this value to zero. */
  for(i=0;i<length;i++) {
    if(v[i] < old_max) v[i] = 0.0;
  }
}

/* Modifies v so that non-leaves are zero.
 * Leaves are nodes which
 *   1. Have no children, or
 *   2. Have energy < cutoff*(max energy of siblings)
 */
void TswCutTree(double *v, int levels, float cutoff)
{
  double maximum = 0.0;
  int i, size;

  if(levels == 1) return;
  /* ignore the root */
  *v = 0.0;
  v++;
  /* compute the size of the subtrees */
  size = (int)((pow(4.0, (double)levels-1)-1)/3);

  /* Find the maximum value at this level */
  for(i=0;i<4;i++) {
    if(v[i*size] > maximum) maximum = v[i*size];
  }
  /* Zero children of all nodes whose energy is < cutoff*maximum */
  maximum *= (double)cutoff;
  for(i=0;i<4;i++) {
    if(v[i*size] < maximum) {
      /* Doesn't zero the parent */
      int j; for(j=1;j<size;j++) v[i*size+j] = 0.0;
    }
    else TswCutTree(&v[i*size], levels-1, cutoff);
  }
}

