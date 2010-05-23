/* Standard definitions for the TPM library */
#ifndef TPM_H_INCLUDED
#define TPM_H_INCLUDED

/* include all the commonly needed headers */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <malloc.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <tpm/binary.h>
#include <tpm/memcheck.h>
#include <tpm/error.h>

#define Allocate(n,t) (t*)malloc((n)*sizeof(t))
#define AllocateC(n,t) (t*)calloc(n, sizeof(t))
#define Reallocate(p, n, t) ((p)=(t*)realloc((p),(n)*sizeof(t)))
#define AllocCopy(n,t,v) (t*)memcpy(Allocate(n,t),v,(n)*sizeof(t))

#ifndef max
#define max(a,b) ((a)>(b)?(a):(b))
#endif
#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#endif

/* Define TPM_SWAP if the machine is big-endian.
 * big: hp700, sun4, iris
 * little: alpha
 */
#if defined(hpux) || defined(sgi) || defined(sun)
#define TPM_SWAP
#endif

/* distinguished integer value */
#define NAI INT_MIN

/* Uses lowest bit to make a tagged union of pointer and integer.
 * (Uses the fact that pointers to aligned locations, like subroutine 
 *  entry points, must have LSB == 0).
 * Works for integers in the range -0x3ffffffe .. 0x3fffffff (for 32-bit longs)
 */
#define PValInt(i) (void*)(((long)(i) << 1)+1)
#define PValIsInt(p) ((long)(p) & 1)
#define PValAsInt(p) ((long)(p) >> 1)

#define MakeAligned(ptr, size) \
  if((long)(ptr) % (size))     \
  ptr = (void*)((long)(ptr) + (size) - ((long)(ptr) % (size))); else
#define MakeAlignedInt(x, size) \
  if((x) % (size)) x += (size) - ((x) % (size)); else

typedef void *Any;
typedef void *Ptr;

typedef struct {
  char *name;
  void *value;
} Association;
  
/* tpm.c *********************************************************************/
double RandReal(void);
float RandFloat(void);
int RandInt(int start, int end);
double ipow(double b, int e);
unsigned long isqrt(unsigned long x);
int gcd(int a, int b);

typedef void FreeFunc(void *);
typedef int CmpFunc(const void *, const void *);
typedef void WriteFunc(FILE *, void *);
typedef void *ReadFunc(FILE *);
typedef void *CopyFunc(void *p);

ReadFunc RealRead, IntRead;
WriteFunc RealWrite, IntWrite;
CmpFunc RealCmp, IntCmp, FloatCmp, AllMatchCmp;
CopyFunc IntCopy;
void *RealCreate(double r);
void *IntCreate(int i);

void TpmSplit(char *string, char split, int *argc_p, char ***argv_p);
Association *AssociationCreate(char *name, void *value);
void AssociationFree(void *a);
void *ArrayCopy(void *array, int length, int item_size);
void *AllocMatrix(int rows, int cols, int item_size);
void FreeMatrix(void *m);
FILE *SafeOpen(char *fname, char *mode);
void *GenericMalloc(size_t size);
FreeFunc GenericFree;

/* path.c ********************************************************************/
char *Path_LastName(char *path);
char *Path_TrueName(char *path);
char *Path_Ups(char *path);
char *Path_Home(char *user);

/* Machine patches ***********************************************************/
#ifdef sun
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#endif
#if defined(ultrix)
char *strdup(char *s);
#endif

#ifndef RAND_MAX
#if defined ( SYSV )
#define RAND_MAX ((1<<15)-1)
#else
#define RAND_MAX ((1<<31)-1)
#endif
#endif

#endif 
