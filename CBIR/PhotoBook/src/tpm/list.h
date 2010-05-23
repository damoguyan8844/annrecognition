/* Definitions for linked lists */
#ifndef TPM_LIST_H_INCLUDED
#define TPM_LIST_H_INCLUDED

#include <tpm/tpm.h>

/* ListNode structure (transparent) */
typedef struct ListNodeStruct {
  void *data;
  struct ListNodeStruct *next;
} *ListNode;

/* List structure (opaque) */
typedef struct ListStruct {
  ListNode front;
  unsigned size;
  FreeFunc *fp;
} *List;

/* Macros */
#define ListFront(list) ((list)->front)
#define ListFreeFunc(list) ((list)->fp)
#define ListSize(list) ((list)->size)
#define ListEmpty(list) (ListFront(list) == NULL)
#define ListFlush(list) ListRemoveAll(list, NULL, AllMatchCmp)
/* A typical use of this iterator is
 * {ListNode p;ListIterate(p, list) {
 *   Print(p->data);
 * }}
 */
#define ListIterate(p, list) \
  for(p = ListFront(list); p; p = (p)->next)
#define ListIter(p, d, list) \
  ListNode p;for(p = ListFront(list); p && (d=(p)->data,p);p=(p)->next)

/* list.c ********************************************************************/
ListNode ListRear(List list);
List ListCopy(List list, CopyFunc *cf);
List ListAddInOrder(List list, void *value, CmpFunc *cf);
void *ListValueAtIndex(List list, int index);
int ListIndexAtValue(List list, void *value, CmpFunc *cf);
List ListAddAtIndex(List list, int index, void *value);
List ListAddFront(List list, void *value);
List ListAddRear(List list, void *value);
List ListRemoveFront(List list, void **value_return);
List ListRemoveRear(List list, void **value_return);
List ListRemoveAtIndex(List list, int index, 
		       void **value_return);
int  ListRemoveValue(List list, void *value, CmpFunc *cf);
int  ListRemoveAll(List list, void *value, CmpFunc *cf);
List ListAppend(List first, List second);
List ListVaAppend(List first, ...);
List ListCreate(FreeFunc *fp);
void ListFree(List list);
void *ListToArray(List list, int *nitems_return, int item_size);
void *ListToPtrArray(List list, int *nitems_return);
List ListFromArray(void *array, int nitems, int item_size);
List ListFromPtrArray(void *array, int nitems);
List ListReverse(List list);

#endif
