/* Miscellaneous routines.
 * Includes random number generators, create/read/write/compare functions
 * for common types, and allocation/deallocation convenience routines.
 */
#include <tpm/tpm.h>

typedef double Real;

/* Prototypes ****************************************************************/
Real RandReal(void);
float RandFloat(void);
int RandInt(int start, int end);
Real ipow(Real b, int e);
unsigned long isqrt(unsigned long x);
int gcd(int a, int b);

void TpmSplit(char *string, char split, int *argc_p, char ***argv_p);
Association *AssociationCreate(char *name, void *value);
void AssociationFree(void *a);
void *ArrayCopy(void *array, int length, int item_size);
void *AllocMatrix(int rows, int cols, int item_size);
void FreeMatrix(void *m);
FILE *SafeOpen(char *fname, char *mode);
void *GenericMalloc(size_t size);
FreeFunc GenericFree;

/* RandReal function *********************************************************/

/* Returns a Real in the range [0,1] */
Real RandReal(void)
{
  return (Real)rand()/RAND_MAX;
}

/*
*  Generates a uniformly distributed r.v. between 0 and 1.
*  Kris Popat  6/85
*  Ref: Applied Statistics, 1982 vol 31 no 2 pp 188-190
*  Based on FORTRAN routine by H. Malvar.
*/

static long ix = 101;
static long iy = 1001;
static long iz = 10001;

float RandFloat(void)
{
  static float u;
  
  ix = 171*(ix % 177)-2*(ix/177);
  iy = 172*(iy % 176)-2*(iy/176);
  iz = 170*(iz % 178)-2*(iz/178);
  
  if (ix<0) ix = ix + 30269;
  if (iy<0) iy = iy + 30307;
  if (iz<0) iz = iz + 30323;
  
  u = ((float) ix)/30269.0 +
                ((float) iy)/30307.0 + ((float) iz)/30323.0;
  u -= (float)(int)u;
  return(u);
}

/* Generates an integer r.v. uniformly distributed in [start, end] */
int RandInt(int start, int end)
{
  return (int)(RandReal()*(end-start)+start+0.5);
}

/* integer math functions ****************************************************/

/* Fast power to integer exponent.
 * Taken from Ammeraal, "Programs and Data Structures in C".
 */
Real ipow(Real b, int e)
{
  int negexp = e < 0;
  double y;
  if(e == 0) return 1.0;
  if(negexp) e = -e;
  while(!(e&1)) { b*=b; e>>=1; }
  y = b;
  while(e >>= 1) {
    b *= b;
    if(e&1) y*=b;
  }
  return negexp ? 1.0/y : y;
}

/* Integer square root.
 * Adapted from Bentley, "More Programming Pearls".
 * Valid for x = 0..2147483648
 * On machines with good FPUs, is slower than (double)sqrt((double)x).
 * On machines without hardware divide (like HP), is really slow.
 */
unsigned long isqrt(unsigned long x)
{
  unsigned long z, nz;
  if(!x) return 0;
  /* Get a good approximation to the root by halving the number of bits */
  nz = x >> 2;
  z = 1;
  while(nz) { nz >>= 2; z <<= 1; }
  /* Now drive to the root with four Newton iterations */
  z = (z + x/z) >> 1;
  z = (z + x/z) >> 1;
  z = (z + x/z) >> 1;
  z = (z + (x+z-1)/z) >> 1;
  return z;
}

/* Greatest positive common factor. E.g. gcd(12, -18) == 6. */
int gcd(int a, int b)
{
  int g;
  if(b == 0) g = a;
  else g = gcd(b, a%b);
  if(g < 0) return -g;
  return g;
}

/* Create/Read/Write/Cmp functions *******************************************/

void *RealCreate(Real r)
{
  Real *p = Allocate(1,Real);
  *p = r;
  return (void*)p;
}

void *RealRead(FILE *fp)
{
  Real *p = Allocate(1,Real);
  int status = fscanf(fp, "%lg", p);
  if(!status || (status == EOF)) {
    fprintf(stderr, "RealRead: Bad floating point number\n");
  }
  return (void*)p;
}

void RealWrite(FILE *fp, void *p)
{
  fprintf(fp, "%g", *(Real*)p);
}

int RealCmp(const void *va, const void *vb)
{
  const Real *a = va, *b = vb;
  
  if(*a < *b) return -1;
  else if(*a > *b) return 1;
  else return 0;
}

int FloatCmp(const void *va, const void *vb)
{
  const float *a = va, *b = vb;
  if(*a < *b) return -1;
  else if(*a > *b) return 1;
  else return 0;
}

void *IntCreate(int i)
{
  int *p = Allocate(1,int);
  *p = i;
  return (void*)p;
}

void *IntRead(FILE *fp)
{
  int *p = Allocate(1,int);
  fscanf(fp, "%d", p);
  return (void*)p;
}

void IntWrite(FILE *fp, void *p)
{
  fprintf(fp, "%d", *(int*)p);
}

int IntCmp(const void *va, const void *vb)
{
  const int *a = va, *b = vb;

  if(*a < *b) return -1;
  else if(*a > *b) return 1;
  else return 0;
}

void *IntCopy(void *v)
{
  return IntCreate(*(int*)v);
}

int AllMatchCmp(const void *va, const void *vb)
{
  return 0;
}

/* Misc **********************************************************************/

void TpmSplit(char *string, char split, int *argc_p, char ***argv_p)
{
  char *p, **word;
  /* at most len/2 words in string */
  *argv_p = word = Allocate((strlen(string)+1)/2, char*);
  for(;*string;string++) {
    /* scan until split char */
    for(p=string;*p && *p != split;p++);
    if(p == string) continue;
    /* make a copy */
    *word = AllocCopy(p-string+1, char, string);
    (*word)[p-string] = 0;
    word++;
    if(!*p) break;
    string = p;
  }
  *argc_p = word - *argv_p;
  Reallocate(*argv_p, *argc_p, char*);
}

Association *AssociationCreate(char *name, void *value)
{
  Association *a = Allocate(1, Association);
  a->name = name;
  a->value = value;
  return a;
}

void AssociationFree(void *a)
{
  free(((Association*)a)->name);
  free(a);
}

void *ArrayCopy(void *array, int length, int item_size)
{
  void *result;

  result = malloc(length * item_size);
  memcpy(result, array, length * item_size);
  return result;
}

void *AllocMatrix(int rows, int cols, int item_size)
{
  char **p;
  int i;

  p = Allocate(rows, char*);
  p[0] = Allocate(rows * cols * item_size, char);
  memset(p[0], 0, rows*cols*item_size);
  for(i=1;i<rows;i++)
    p[i] = &p[0][i*cols*item_size];
  return (void*)p;
}

void FreeMatrix(void *m)
{
  free(((void**)m)[0]);
  free(m);
}

FILE *SafeOpen(char *fname, char *mode)
{
  FILE *fp = fopen(fname, mode);
  if(fp == NULL) {
    fprintf(stderr, "Could not open file `%s': ", fname);
    perror("");
    fflush(stderr);
    exit(1);
  }
  return fp;
}

void *GenericMalloc(size_t size)
{
  void *p = malloc(size);
  if(!p) {
    fprintf(stderr, "Failed allocating %d bytes\nOut of memory\n", size);
    exit(1);
  }
  return p;
}

void GenericFree(void *p)
{
  free(p);
}

/* Every ANSI compiler should have this function!! */
#if defined(ultrix)
char *strdup(char *s)
{
  char *str;

  str = malloc(strlen(s)+1);
  strcpy(str, s);
  return str;
}
#endif
