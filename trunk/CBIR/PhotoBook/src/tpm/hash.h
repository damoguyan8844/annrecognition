/* Definition of HashTable */
#ifndef HASH_H_INCLUDED
#define HASH_H_INCLUDED

#include <tpm/tpm.h>

/* Key types */
enum {
  HASH_STRING,
  HASH_WORD
  };

typedef char *HashKey;

typedef struct HashEntryStruct {
  void *value;
  struct HashEntryStruct *next;
  char *key;
  int bucket;
  char keydata[sizeof(int)]; /* must be last field */
} HashEntry;

typedef struct HashTableStruct {
  int key_type;
  int num_buckets, num_entries;
  int mask, down_shift;
  HashEntry **buckets;
  FreeFunc *ff;
} *HashTable;

#define HashEntryValue(h) (h)->value
#define HashEntryKey(h) (h)->key
#define HashTableSize(ht) (ht)->num_entries

/* These iterators don't support:
 * 1. Modification of the table.
 * 2. The "break" statement (will only stop the inner loop).
 */
#define HashIterate(b, e, ht) \
  for(b=0; b<(ht)->num_buckets; b++) \
     for(e=(ht)->buckets[b]; e; e=e->next)

#define HashIter(b, e, v, ht) \
  int b; HashEntry *e; for(b=0; b<(ht)->num_buckets; b++) \
          for(e=(ht)->buckets[b]; e && (v=(e)->value,e); e=e->next)

/* Could also use:
 *   for(e=NULL;e=HashTableNextEntry(ht,e);)
 */

HashTable HashTableCreate(int key_type, FreeFunc *ff);
void HashTableFree(HashTable ht);
void HashTableFlush(HashTable ht);
HashTable HashTableCopy(HashTable ht, CopyFunc *cf);

HashEntry *HashTableAddEntry(HashTable ht, HashKey key, int *found_return);
HashEntry *HashTableFindEntry(HashTable ht, HashKey key);
void HashTableRemoveEntry(HashTable ht, HashEntry *entry);
void HashTableRemoveNoFree(HashTable ht, HashEntry *entry);
int HashTableRemoveValueAll(HashTable ht, void *value, CmpFunc *cf);
HashEntry *HashTableNextEntry(HashTable ht, HashEntry *entry);

#endif
