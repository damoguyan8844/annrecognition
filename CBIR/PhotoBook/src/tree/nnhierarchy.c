/* Uses MakePartition to generate either a NN or MNV single-link
 * cluster hierarchy.
 */

#include <tpm/gtree.h>
#include <tpm/partition.h>

#define JP 0
#define BASE 1

int EquivNN(int *a, int *b);
int EquivMNV(int *a, int *b);
int ArrayPos(int *array, int length, int v);
int SizeOfIntersection(int *a1, int *a2, int length);

int dist_thres, shared_thres;

int main(int argc, char **argv)
{
  char *nn_file;
  FILE *fp;
  int **samples, **s1, **s2;
  int i,j,k,c;
  int num_samples;
  int nclusters;
  Partition p;
  int min_dist_thres, max_dist_thres;
  Tree *trees, tree;

  if(argc < 6) {
    printf("Usage:\n%s <nn_file> <num_samples> <shared_thres> <start_dist_thres> <end_dist_thres>\n",
	   argv[0]);
    printf("To generate a hierarchy from <num_samples> lines of the neighbor file <nn_file>.\n");
    printf("dist_thres = distance threshold (k-nearest)\n");
    printf("end_dist_thres <= neighbors in nn_file\n");
    printf("tree goes to stdout\n");
    exit(1);
  }

  nn_file = argv[1];
  num_samples = atoi(argv[2]);
  shared_thres = atoi(argv[3]);
  min_dist_thres = atoi(argv[4]);
  max_dist_thres = atoi(argv[5]);

  /* Read in neighbor file */
  fp = fopen(nn_file, "r");
  if(!fp) {
    printf("Cannot open `%s'\n", nn_file);
    exit(1);
  }
  samples = Allocate(num_samples, int*);
  for(s1=samples,i=0;i<num_samples;s1++,i++) {
    *s1 = Allocate(max_dist_thres+1, int);
    for(j=0;j<=max_dist_thres;j++) {
      fscanf(fp, "%d", &(*s1)[j]);
    }
    while(fgetc(fp) != '\n');
  }
  fclose(fp);

  /* Initialize the partition and trees */
  p = PartitionCreate(num_samples);
  trees = Allocate(num_samples, Tree);
  for(i=0;i<num_samples;i++) {
    trees[i] = TreeCreate(IntCreate(i+BASE));
  }

  /* loop all possible partitions */
#if JP
  dist_thres = max_dist_thres;
  for(;shared_thres >= 0;shared_thres--) {
#else
  for(dist_thres = min_dist_thres;dist_thres <= max_dist_thres;dist_thres++) {
#endif
    /* Form clusters */
    PartitionRefine(p, (void*)samples, (CmpFunc*)EquivNN);

    /* Set j = size of largest class */
    j = PartitionClassSize(p,1);
    for(c=2;c <= p->num_classes;c++) {
      i = PartitionClassSize(p,c);
      if(i > j) j = i;
    }
    fprintf(stderr, "%d %d %d %d ", 
	    dist_thres, shared_thres, p->num_classes, j);
    fprintf(stderr, "\n");
    fflush(stdout);

    /* create next level of tree */
    /* loop classes */
    for(c = 1;c <= p->num_classes;c++) {
      if(PartitionClassSize(p, c) <= 1) continue;
      tree = TreeCreate(IntCreate(dist_thres));
      {PartitionIter(p,c,e) {
        TreeAddChildUnique(tree, trees[e]);
        trees[e] = tree;
      }}
    }
  }
  PartitionFree(p);

  /* put all existing trees under one tree */
  tree = TreeCreate(IntCreate(dist_thres));
  for(i=0;i<num_samples;i++) {
    TreeAddChildUnique(tree, trees[i]);
  }

  TreeWrite(stdout, tree, IntWrite);
  TreeFree(tree, free);
  exit(0);
}

int EquivNN(int *a, int *b)
{
  if(ArrayPos(a, dist_thres+1, *b) == -1) return 1;
  if(ArrayPos(b, dist_thres+1, *a) == -1) return 1;
  if(SizeOfIntersection(a, b, dist_thres+1) < 
     shared_thres) return 1;
  return 0;
}

int EquivMNV(int *a, int *b)
{
  int pa, pb;

  pa = ArrayPos(a, dist_thres+1, *b);
  pb = ArrayPos(b, dist_thres+1, *a);
  if(pa == -1 || pb == -1) return 1;
  if(pa + pb > dist_thres) return 1;
  return 0;
}

int ArrayPos(int *array, int length, int v)
{
  int i;

  for(i=0;i<length;i++) if(array[i] == v) return i;
  return -1;
}

int SizeOfIntersection(int *a1, int *a2, int length)
{
  int i,j, count;

  count = 0;
  /* exclude the first elements */
  for(i=1;i<length;i++) {
    for(j=1;j<length;j++) {
      if(a1[i] == a2[j]) count++;
    }
  }
  return count;
}
