#include "photobook.h"
#include "obj_data.h"
#include <math.h>

/* Functions *****************************************************************/

#define Value(t) *(double*)(t)->data

static void ScoreLeaves(Ph_Handle phandle, Tree t, int score)
{
  if(TreeLeaf(t)) {
    Ph_Member m = phandle->total_set[(int)Value(t)-1];
    Ph_MemDistance(m) = score + RandReal()/2;
  }
  else {
    IterateChildren(t, child) {
      ScoreLeaves(phandle, child, score);
    }
  }
}

static Tree LeafWithValue(Tree t, double v)
{
  Tree l;
  
  if(TreeLeaf(t)) {
    if(Value(t) == v) return t;
  }
  else {
    IterateChildren(t,c) {
      if(l = LeafWithValue(c,v)) return l;
    }
  }
  return NULL;
}

/* assigns distances to the entire database, regardless of the count argument.
 * another implementation could store high/low values
 * at O(h) tree nodes and assign distances by comparing against 
 * successive nodes. This would avoid having to traverse the whole tree.
 */
void HierDistance(Ph_Object self, Ph_Member query, 
		  Ph_Member *test, int count)
{
  int score;
  Tree node;
  struct HierData *data = (struct HierData *)self->data;

  if(!data->tree) {
    fprintf(stderr, "%s metric: no tree defined\n", Ph_ObjName(self));
    return;
  }
  node = LeafWithValue(data->tree, Ph_MemIndex(query)+1);
  ScoreLeaves(self->phandle, node, 0);
  for(score=1;node->parent;score++) {
    IterateChildren(node->parent, child) {
      if(child != node) ScoreLeaves(self->phandle, child, score);
    }
    node = node->parent;
  }
}

static void watchProc(Ph_Object self, char *field, void *userData)
{
  struct HierData *data = (struct HierData *)self->data;
  FILE *fp;
  char str[100];

  /* load in the tree */
  fp = fopen(data->tree_file, "r");
  if(!fp) {
    sprintf(str, "%s/%s/%s", self->phandle->data_dir, self->phandle->db_name,
	    data->tree_file);
    fp = fopen(str, "r");
    if(!fp) {
      fprintf(stderr, "%s metric: cannot open tree file `%s'\n", 
	      Ph_ObjName(self), data->tree_file);
      return;
    }
  }
  if(debug) printf("loading tree file `%s'\n", data->tree_file);
  data->tree = TreeRead(fp, RealRead);
  fclose(fp);
}

void HierCon(Ph_Object self)
{
  struct HierData *data = (struct HierData *)self->data;
  data->tree = NULL;
  /* set up watch callback */
  Ph_ObjWatch(self, "tree", watchProc, NULL);
}

void HierDes(Ph_Object self)
{
  struct HierData *data = (struct HierData *)self->data;
  if(data->tree) TreeFree(data->tree, GenericFree);
}
