/* Machine-independent I/O. Everything is stored little-endian; applications
 * do the swapping.
 * To read in an array of 10 integers:
 *   ReadAdjust(buffer, TPM_INT, 10, fileHandle);
 * To write an array of 10 integers:
 *   WriteAdjust(buffer, TPM_INT, 10, fileHandle);
 */

typedef enum { TPM_CHAR, TPM_INT, TPM_FLOAT, TPM_DOUBLE } TpmType;
#define TPM_REAL TPM_DOUBLE
extern char normal_type_sizes[];
extern char native_type_sizes[];
#define TpmTypeSize(t) native_type_sizes[t]

#define ReadAdjust(p,t,n,fp) \
  fread(p,normal_type_sizes[t],n,fp),Normal2Native(p,n,t)

/* The second Native2Normal is to swap the buffer back again */
#define WriteAdjust(p,t,n,fp) \
  fwrite(Native2Normal(p,n,t),normal_type_sizes[t],n,fp),Native2Normal(p,n,t)

void *Native2Normal(void *p, unsigned num_items, TpmType item_type);
void *Normal2Native(void *p, unsigned num_items, TpmType item_type);
void *SwapBuffer(void *p, unsigned num_items, unsigned item_size);
void *AdjustBuffer(void *p, unsigned num_items, unsigned item_size);
