/* Partition implementation.
 * A partition is just a mapping from indices to class numbers.
 * Class numbers start at 1 (0 is reserved by the module).
 * A proper partition does not skip class numbers, i.e. the maximum
 * class number is == the number of distinct classes.
 * Use PartitionNormalize to enforce this property.
 */

#include "partition.h"

/* Prototypes ****************************************************************/

Partition PartitionCreate(int num_elts);
void PartitionFree(Partition p);

Partition PartitionCreateFromFile(char *fname);
void PartitionWrite(FILE *fp, Partition p);

void PartitionRefine(Partition p, void **samples, CmpFunc *eqv);
void PartitionMerge(Partition p, int a, int b);
void PartitionNormalize(Partition p);
int PartitionClassSize(Partition p, int c);
int PartitionIterate(Partition p, int class, int *elt);
double PartitionRandIndex(Partition p1, Partition p2);
Partition PartitionOr(Partition p1, Partition p2);

/* Functions *****************************************************************/

/* Free the storage used by p */
void PartitionFree(Partition p)
{
  free(p->class);
  free(p);
}

/* Read a partition from fp.
 * The format is the number of elements followed by 
 * a sequence of class numbers.
 * Returns NULL if an error occurred.
 */
Partition PartitionCreateFromFile(char *fname)
{
  Partition p;
  int i;
  FILE *fp;

  if(!strcmp(fname, "-")) fp = stdin;
  else fp = fopen(fname, "r");
  if(!fp) return NULL;
  /* read the number of elements */
  if(fscanf(fp, "%d", &i) < 1) return NULL;
  p = PartitionCreate(i);
  for(i=0;i < p->num_elements;i++) {
    fscanf(fp, "%d", &p->class[i]);
    if(p->class[i] > p->num_classes)
      p->num_classes = p->class[i];
  }
  fclose(fp);
  return p;
}

/* Write a partition in ASCII to fp.
 */
void PartitionWrite(FILE *fp, Partition p)
{
  int i;

  fprintf(fp, "%d\n", p->num_elements);
  for(i=0;i < p->num_elements;i++) {
    fprintf(fp, "%d\n", p->class[i]);
  }
}

/* Normalize p so it satifies the rep invariant, i.e.
 * for every c in [1, max(p[k])] there exists 
 * an i such that p[i] = c, and all p[k] are positive.
 */
void PartitionNormalize(Partition p)
{
  int i, j, empty, cur_class, max_class;

  max_class = 0;
  for(i=0;i < p->num_elements;i++) {
    if(p->class[i] > max_class) max_class = p->class[i];
  }

  cur_class = 1;
  for(j=1;j <= max_class;j++) {
    empty = 1;
    for(i=0;i < p->num_elements;i++) {
      if(p->class[i] == j) {
	empty = 0;
	p->class[i] = cur_class;
      }
    }
    if(!empty) cur_class++;
  }
  /* Singletons */
  for(i=0;i < p->num_elements;i++) {
    if(p->class[i] == 0) p->class[i] = cur_class++;
  }
  p->num_classes = cur_class-1;
}

/* Returns the number of samples in p in class c */
int PartitionClassSize(Partition p, int c)
{
  int i, count;

  count = 0;
  for(i=0;i < p->num_elements;i++) {
    if(p->class[i] == c) count++;
  }
  return count;
}

/* Used by the PartitionIter macro.
 * Sets *elt to the first sample in p in class "class", starting
 * at *elt.
 * Returns 1 iff a sample was found.
 */
int PartitionIterate(Partition p, int class, int *elt)
{
  for(;;(*elt)++) {
    if(*elt == p->num_elements) return 0;
    if(p->class[*elt] == class) return 1;
  }
}

/* Merges the class of a and the class of b to form a single class.
 * This function may break the rep invariant.
 */
void PartitionMerge(Partition p, int a, int b)
{
  int i,c;

  if(p->class[a] == 0) {
    if(p->class[b] == 0) {
      p->class[a] = p->class[b] = ++(p->num_classes);
    }
    else {
      p->class[a] = p->class[b];
    }
  }
  else if(p->class[b] == 0) {
    p->class[b] = p->class[a];
  }
  else if(p->class[a] != p->class[b]) {
    c = p->class[a];
    for(i=0;i < p->num_elements;i++) {
      if(p->class[i] == c) p->class[i] = p->class[b];
    }
  }
}

/* Merges all classes containing at least one equivalent member,
 * according to eqv.
 */
void PartitionRefine(Partition p, void **samples, CmpFunc *eqv)
{
  int i,j;

  for(i=0;i < p->num_elements;i++) {
    for(j=i+1;j < p->num_elements;j++) {
      if(!p->class[i] || (p->class[i] != p->class[j]))
	if(!eqv(samples[i], samples[j]))
	  PartitionMerge(p, i, j);
    }
  }
  PartitionNormalize(p);
}

/* Returns a NULL partition of num_elts elements.
 * The return value does not satisfy the rep invariant.
 */
Partition PartitionCreate(int num_elts)
{
  Partition p;
  int i;

  p = Allocate(1, struct Partition);
  p->num_elements = num_elts;
  p->num_classes = 0;
  p->class = Allocate(num_elts, int);

  for(i=0;i < num_elts;i++) {
    p->class[i] = 0;
  }
  return p;
}

/* This index was taken from Jain and Dubes. It ranges from -1 to 1.
 * Larger indices indicate stronger correspondence between p1 and p2.
 * It is invariant to permutations of the class numbers.
 *
 * Special cases:
 *   If p1 = p2, index = 1.
 * Otherwise,
 *   If p1->num_classes = p1->num_elements, index = 0. (no classes)
 *   If p1->num_classes = 1, index = 0. (one class)
 */
double PartitionRandIndex(Partition p1, Partition p2)
{
  int n, i, j;
  int **histo;
  double z,y,x,w,v,u;

  /* n = min(p1, p2) */
  n = p1->num_elements;
  if(p2->num_elements < n) n = p2->num_elements;
  if(n == 1) return 1;

  /* fast method; histogram of pairs */

  /* allocate and clear histogram */
  histo = Allocate(p1->num_classes, int*);
  histo[0] = Allocate(p1->num_classes * p2->num_classes, int);
  for(i=0;i < p1->num_classes * p2->num_classes;i++) {
    histo[0][i] = 0;
  }
  for(i=1;i < p1->num_classes;i++) {
    histo[i] = &histo[0][i * p2->num_classes];
  }

  /* fill histogram */
  for(i=0;i < n;i++) {
    histo[ p1->class[i]-1 ][ p2->class[i]-1 ]++;
  }

  z = y = 0.0;
  for(i=0;i < p1->num_classes;i++) {
    for(j=0;j < p2->num_classes;j++) {
      z += (double)histo[i][j] * histo[i][j];
    }
    j = PartitionClassSize(p1, i+1);
    y += (double)j * j;
  }
  x = 0.0;
  for(i=0;i < p2->num_classes;i++) {
    j = PartitionClassSize(p2, i+1);
    x += (double)j * j;
  }
  w = z - n;
  v = (x+y)/2 - n;
  u = (x-n)*(y-n)/n/(n-1);

  /* Check for p1 = p2 case */
  if((v == u) && (w == u)) return 1;

  return (w-u)/(v-u);
}

/* Produces a new partition where two elements are in the same class
 * if they were so in p1 or in p2.
 */
Partition PartitionOr(Partition p1, Partition p2)
{
  Partition p;
  int i;

  p = PartitionCreate(p1->num_elements);
  memcpy(p->class, p1->class, p->num_elements * sizeof(int));
  for(i=0;i<p->num_elements;i++) {
    PartitionIter(p2, p2->class[i], e) {
      PartitionMerge(p, i, e);
    }
  }
  PartitionNormalize(p);
  return p;
}
