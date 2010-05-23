/* Definitions for the Partition data type */
#ifndef PARTITION_H_INCLUDED
#define PARTITION_H_INCLUDED

#include <tpm/tpm.h>

typedef struct Partition {
  int num_elements, num_classes;
  int *class;
} *Partition;

#define PartitionIter(p,c,e) int e;for(e=0;PartitionIterate(p,c,&e);e++)

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

#endif
