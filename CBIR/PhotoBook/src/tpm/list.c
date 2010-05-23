/* Linked list implementation. 
 * Each node has data and a next link. The end of the list has a NULL link.
 * There are convenience routines for adding/removing to the front and end,
 * for simulating queues, stacks, dequeues, etc.
 */

#include <stdarg.h>
#include <malloc.h>
#include <tpm/list.h>

/* Prototypes ****************************************************************/

List ListAddInOrder(List list, void *value, CmpFunc *cf);
ListNode ListRear(List list);
List ListCopy(List list, CopyFunc *cf);
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

/* Private */
static ListNode NodeAlloc(void);

/* Functions *****************************************************************/

/* Returns the last node in the list */
ListNode ListRear(List list)
{
  ListNode ptr;
  if(ListEmpty(list)) return NULL;
  for(ptr = ListFront(list); ptr->next; ptr = ptr->next);
  return ptr;
}

/* Returns a copy of the list. Uses cf to copy data fields.
 * If cf == NULL, the same data is referenced by the copy and the original.
 */
List ListCopy(List list, CopyFunc *cf)
{
  List result = ListCreate(ListFreeFunc(list));
  ListNode *p1, p2;
  for(p1 = &result->front, p2 = list->front;
      p2; p1 = &(*p1)->next, p2 = p2->next) {
    *p1 = NodeAlloc();
    (*p1)->next = NULL;
    (*p1)->data = cf ? cf(p2->data) : p2->data;
  }
  result->size = list->size;
  return result;
}

/* Adds a new node so that the list remains in ascending order according
 * to a comparison function (cf). Assumes that the list is already ordered.
 */
List ListAddInOrder(List list, void *value, CmpFunc *cf)
{
  ListNode node, *ptr;

  assert(list);
  node = NodeAlloc();
  node->data = value;
  node->next = NULL;
  for(ptr = &ListFront(list);
      *ptr && (cf((*ptr)->data, value) < 0);
      ptr = &(*ptr)->next);
  node->next = *ptr;
  *ptr = node;
  ListSize(list)++;
  return list;
}

/* Add a new node so that ListValueAtIndex(list, index) = value.
 */
List ListAddAtIndex(List list, int index, void *value)
{
  ListNode node, *ptr;

  assert(list);
  node = NodeAlloc();
  node->data = value;
  for(ptr = &ListFront(list);*ptr && index;ptr = &(*ptr)->next, index--);
  node->next = *ptr;
  *ptr = node;
  ListSize(list)++;
  return list;
}

/* Add a new node to the front of list, with data field equal to value.
 * Returns the resulting list, which is also placed in list.
 */
List ListAddFront(List list, void *value)
{
  ListNode node;

  assert(list);
  node = NodeAlloc();
  node->data = value;
  node->next = ListFront(list);
  ListFront(list) = node;
  ListSize(list)++;
  return list;
}

/* Add a new node to the end of list, with data field equal to value.
 * Returns the resulting list, which is also placed in list.
 */
List ListAddRear(List list, void *value)
{
  ListNode node, *ptr;

  assert(list);
  node = NodeAlloc();
  node->data = value;
  node->next = NULL;
  for(ptr = &ListFront(list);*ptr;ptr = &(*ptr)->next);
  *ptr = node;
  ListSize(list)++;
  return list;
}

/* Removes a node from the front of list.
 * If value_return is NULL, the node is destroyed. 
 * Otherwise, *value_return points to the data field of the removed node.
 * If the list is empty, *value_return will be set to NULL and the list will
 * be unchanged.
 */
List ListRemoveFront(List list, void **value_return)
{
  ListNode temp;

  assert(list);
  temp = ListFront(list);
  if(!temp) {
    if(value_return)
      *value_return = NULL;
  }
  else {
    ListFront(list) = temp->next;
    if(value_return)
      *value_return = temp->data;
    else if(ListFreeFunc(list)) {
      ListFreeFunc(list)(temp->data);
    }
    free(temp);
    ListSize(list)--;
  }
  return list;
}

/* Removes a node from the end of list. 
 * If value_return is NULL, the node is destroyed. 
 * Otherwise, *value_return points to the data field of the removed node.
 * If the list is empty, *value_return will be set to NULL and the list will
 * be unchanged.
 */
List ListRemoveRear(List list, void **value_return)
{
  ListNode temp, *ptr;

  assert(list);
  if(ListEmpty(list)) {
    if(value_return)
      *value_return = NULL;
  }
  else {
    for(ptr = &ListFront(list);(*ptr)->next;ptr = &(*ptr)->next);
    temp = *ptr;
    *ptr = temp->next;
    if(value_return)
      *value_return = temp->data;
    else if(ListFreeFunc(list)) {
      ListFreeFunc(list)(temp->data);
    }
    free(temp);
    ListSize(list)--;
  }
  return list;
}

/* Returns the index of the node such that cf(data, value) == 0.
 * If cf == NULL, returns the index such that data == value. (pointer equality)
 * Numbering starts at zero. If no node has that value, returns -1.
 */
int ListIndexAtValue(List list, void *value, CmpFunc *cf)
{
  ListNode ptr;
  int index;

  assert(list);
  index = 0;
  ListIterate(ptr, list) {
    if(!cf) { if(ptr->data == value) return index; }
    else if(!cf(ptr->data, value)) return index;
    index++;
  }
  return -1;
}

/* Returns the data field of the node at a specific index in a list, 
   starting from zero. If index is out of range, returns NULL. */
void *ListValueAtIndex(List list, int index)
{
  ListNode ptr;

  assert(list);
  if(index < 0) return NULL;
  ListIterate(ptr, list) if(!index--) break;
  if(ptr) return ptr->data;
  else return NULL;
}

/* Remove the node of list at index index. Numbering starts at zero.
 * If value_return is NULL, the node is destroyed. 
 * Otherwise, *value_return points to the data field of the removed node.
 * If the list is empty, *value_return will be set to NULL and the list will
 * be unchanged.
 */
List ListRemoveAtIndex(List list, int index, 
		       void **value_return)
{
  ListNode *ptr, temp;

  assert(list);
  for(ptr = &ListFront(list);index && *ptr;index--,ptr = &(*ptr)->next);
  if(!*ptr) {
    if(value_return)
      *value_return = NULL;
  }
  else {
    temp = *ptr;
    *ptr = temp->next;
    if(value_return)
      *value_return = temp->data;
    else if(ListFreeFunc(list)) {
      ListFreeFunc(list)(temp->data);
    }
    free(temp);
    ListSize(list)--;
  }
  return list;
}

/* Remove the first node of list such that cf(data, value) == 0.
 * If cf == NULL, uses pointer equality (data == value).
 * Returns TRUE if a node was removed.
 */
int ListRemoveValue(List list, void *value, CmpFunc *cf)
{
  ListNode *ptr, temp;
  
  assert(list);
  for(ptr= &ListFront(list);*ptr;ptr = &(*ptr)->next) {
    if(((*ptr)->data == value) || (cf && !cf((*ptr)->data, value))) {
      temp = *ptr;
      *ptr = temp->next;
      if(ListFreeFunc(list)) {
	ListFreeFunc(list)(temp->data);
      }
      free(temp);
      ListSize(list)--;
      return 1;
    }
  }
  return 0;
}

/* Remove all nodes of list such that cf(data, value) == 0.
 * If cf == NULL, uses pointer equality (data == value).
 * Returns the number of nodes removed.
 */
int ListRemoveAll(List list, void *value, CmpFunc *cf)
{
  ListNode *ptr, temp;
  int count = 0;
  
  assert(list);
  for(ptr=&ListFront(list);*ptr;) {
    if(((*ptr)->data == value) || (cf && !cf((*ptr)->data, value))) {
      temp = *ptr;
      *ptr = temp->next;
      if(ListFreeFunc(list)) {
	ListFreeFunc(list)(temp->data);
      }
      free(temp);
      ListSize(list)--;
      count++;
    }
    else ptr = &(*ptr)->next;
  }
  return count;
}

/* Concatenate two lists. Arguments must be non-NULL.
 * Either list may be empty. 
 * Returns the resulting list, which is also placed in the first list.
 * The second list is deallocated and should no longer be used.
 */
List ListAppend(List first, List second)
{
  ListNode *ptr;

  assert(first && second);
  for(ptr=&ListFront(first);*ptr;ptr=&(*ptr)->next);
  *ptr = ListFront(second);
  ListSize(first) += ListSize(second);
  free(second);
  return first;
}

/* Concatenate multiple lists. Arguments must all be non-NULL lists,
 * terminated by the value NULL. Lists, including the first, may be empty. 
 * Returns the resulting list, which is also placed in the first list.
 * All other lists have been deallocated and should no longer be used.
 */
List ListVaAppend(List first, ...)
{
  ListNode *ptr;
  List second;
  va_list args;

  assert(first);
  va_start(args, first);
  for(ptr=&ListFront(first);;) {
    second = va_arg(args, List);
    if(!second) break;
    for(;*ptr;ptr=&(*ptr)->next);
    if(!ListEmpty(second)) {
      *ptr = ListFront(second);
      ptr = &(*ptr)->next;
      ListSize(first) += ListSize(second);
    }
    free(second);
  }
  va_end(args);
  return(first);
}

/* Allocate an empty list. 
 * If fp != NULL, it will be used to free nodes for ListRemove or ListFree.
 */
List ListCreate(FreeFunc *fp)
{
  List list;

  list = (List)malloc(sizeof(struct ListStruct));
  if(!list) {
    fprintf(stderr, "ListCreate: Cannot allocate memory\n");
    fflush(stderr);
    exit(1);
  }
  list->front = NULL;
  list->fp = fp;
  list->size = 0;
  return list;
}

/* Free all memory associated with a list variable, using the FreeFunc,
 * if it was specified, to free data fields.
 */
void ListFree(List list)
{
  ListNode temp, ptr;

  assert(list);
  ptr = ListFront(list);
  while(ptr) {
    temp = ptr;
    ptr = ptr->next;
    if(ListFreeFunc(list)) {
      ListFreeFunc(list)(temp->data);
    }
    free(temp);
  }
  free(list);
}

/* Create an array from the contents of list.
 * The structures pointed to by the data fields are copied into consecutive
 * locations of the array. List is not modified. The number of elements
 * in the array is returned in *nitems_return, if nitems_return != NULL.
 * If the list is NULL, a NULL is returned.
 * If the list is empty, a malloc(0) is returned.
 */
void *ListToArray(List list, int *nitems_return, int item_size)
{
  int count;
  ListNode ptr;
  char *array, *p;

  if(!list) return NULL;
  count = ListSize(list);
  if(nitems_return != NULL) *nitems_return = count;

  array = (char*)malloc(count*item_size);
  if(!array) {
    fprintf(stderr, "ListToArray: Cannot allocate array\n");
    fflush(stderr);
    exit(1);
  }

  p = array;
  ListIterate(ptr, list) {
    memcpy(p, ptr->data, item_size);
    p += item_size;
  }

  return(array);
}

/* Create a List by copying values from an arbitrary array. */
List ListFromArray(void *array, int nitems, int item_size)
{
  List list;
  void *value;

  list = ListCreate(NULL);
  for(;nitems;nitems--) {
    value = (void*)malloc(item_size);
    if(!value) {
      fprintf(stderr, "ListFromArray: Cannot allocate value\n");
      fflush(stderr);
      exit(1);
    }
    memcpy(value, (char*)array, item_size);
    ListAddRear(list, value);
    array = (char*)array + item_size;
  }
  return list;
}

/* Create a List of values from an array of pointers to values. */
List ListFromPtrArray(void *array, int nitems)
{
  List list;

  list = ListCreate(NULL);
  for(;nitems;nitems--) {
    ListAddRear(list, *(void**)array);
    array = ((char**)array)+1;
  }
  return list;
}

/* Create an array of pointers to the contents of list.
 * The data pointers of the list nodes are copied into consecutive
 * locations of the array. List is not modified. The number of elements
 * in the array is returned in *nitems_return, if nitems_return != NULL.
 * If the list is NULL, a NULL is returned.
 * If the list is empty, a malloc(0) is returned.
 */
void *ListToPtrArray(List list, int *nitems_return)
{
  int count;
  ListNode ptr;
  void **array, **p;

  if(!list) return NULL;
  count = ListSize(list);
  if(nitems_return != NULL) *nitems_return = count;

  array = (void**)malloc(count*sizeof(void*));
  if(!array) {
    fprintf(stderr, "ListToPtrArray: Cannot allocate array\n");
    fflush(stderr);
    exit(1);
  }

  p = array;
  ListIterate(ptr, list) {
    *p++ = ptr->data;
  }

  return(array);
}

/* Reverse, in place, the order of nodes in a list. */
List ListReverse(List list)
{
  ListNode *f, p, pn;
  f = &list->front;
  for(p=*f,*f=NULL;p;p=pn) {
    pn = p->next;
    p->next = *f;
    *f = p;
  }
  return list;
}

/* Private *******************************************************************/

/* Allocate one uninitialized list node. Checks for allocation failure. */
static ListNode NodeAlloc(void)
{
  ListNode node;

  node = (ListNode)malloc(sizeof(struct ListNodeStruct));
  if(!node) {
    fprintf(stderr, "NodeAlloc: Cannot allocate node for list\n");
    fflush(stderr);
    exit(1);
  }
  return node;
}
