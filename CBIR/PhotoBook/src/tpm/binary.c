/* Machine-independent I/O routines. Applications should not call these 
 * directly, but use the ReadAdjust and WriteAdjust macros.
 */

#include "binary.h"

/* Globals ******************************************************************/

char normal_type_sizes[] = { 1, 4, 4, 8 };
char native_type_sizes[] = { sizeof(char), sizeof(int), 
			     sizeof(float), sizeof(double) };

/* Prototypes ***************************************************************/

void *Native2Normal(void *p, unsigned num_items, TpmType item_type);
void *Normal2Native(void *p, unsigned num_items, TpmType item_type);
void *SwapBuffer(void *p, unsigned num_items, unsigned item_size);
void *AdjustBuffer(void *p, unsigned num_items, unsigned item_size);

/* Functions ****************************************************************/

void *Native2Normal(void *p, unsigned num_items, TpmType item_type)
{
  return AdjustBuffer(p, num_items, native_type_sizes[item_type]);
}

void *Normal2Native(void *p, unsigned num_items, TpmType item_type)
{
  return AdjustBuffer(p, num_items, normal_type_sizes[item_type]);
}

#define swap(a,b) t=a;a=b;b=t
void *SwapBuffer(void *p, unsigned num_items, unsigned item_size)
{
  char *buffer = (char*)p, t, i;
  if(item_size > 1) {
    for(;num_items;num_items--) {
      for(i=0;i<item_size/2;i++) {
	swap(*(buffer+i), *(buffer+item_size-i-1));
      }
      buffer += item_size;
    }
  }
  return p;
}

/* If the current machine is big-endian, swap the buffer.
 * Otherwise do nothing.
 * BTW: big-endian is "network byte order"
 */
void *AdjustBuffer(void *p, unsigned num_items, unsigned item_size)
{
  int z=1;
  unsigned char *c=(unsigned char *)&z;

  if(c[0]==0) return SwapBuffer(p, num_items, item_size);
  return p;
}
