/* Hash table implementation. Based on TCL hash tables.
 * Tables may be indexed in three ways:
 *   1. by string.
 *   2. by integer.
 *   3. by array of integers. ((2) is the special case where length == 1)
 * Hash tables automatically grow to keep the value/bucket ratio below
 * the REBUILD_MULTIPLIER. However, they will not shrink.
 */

#include <tpm/hash.h>

#define INITIAL_BUCKETS 4
#define REBUILD_MULTIPLIER 3

/* Prototypes ****************************************************************/

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

/* Private */
static void RebuildTable(HashTable ht);

/* Functions *****************************************************************/

/* used by HashTableCreate and HashTableFlush */
static HashTable initialize(HashTable ht)
{
  int i;
  ht->num_buckets = INITIAL_BUCKETS;
  ht->mask = ht->num_buckets - 1; /* assumes num_buckets is power of two */
  ht->down_shift = 28;
  ht->num_entries = 0;
  ht->buckets = Allocate(ht->num_buckets, HashEntry *);
  if(!ht->buckets) {
    fprintf(stderr, "Out of memory; cannot create hash table\n");
    return NULL;
  }
  for(i=0;i<ht->num_buckets;i++) ht->buckets[i] = NULL;
  return ht;
}

/* Create an empty hash table.
 * If key_type == 0, then it will use string keys.
 * Otherwise, it will use integer arrays of length key_type.
 * If ff != NULL, it will be used to free values in the table.
 */
HashTable HashTableCreate(int key_type, FreeFunc *ff)
{
  HashTable ht = Allocate(1, struct HashTableStruct);
  ht->key_type = key_type;
  ht->ff = ff;
  return initialize(ht);
}

/* used by HashTableFlush and HashTableFree */
static void HashTableRemoveAll(HashTable ht)
{
  int b;
  HashEntry **ep, *t;

  if(!ht) return;
  for(b=0;b<ht->num_buckets;b++) {
    for(ep=&ht->buckets[b];*ep;) {
      /* have to keep the ht consistent, since ht->ff may inspect us */
      t = *ep;
      *ep = t->next;
      ht->num_entries--;
      if(ht->ff) ht->ff(t->value);
      free(t);
    }
  }
}

/* Destroys the contents of the HashTable, but not the HashTable itself. */
void HashTableFlush(HashTable ht)
{
  if(!ht) return;
  HashTableRemoveAll(ht);
  free(ht->buckets);
  initialize(ht);
}

/* Destroys a HashTable, using the ff to free the stored values. */
void HashTableFree(HashTable ht)
{
  if(!ht) return;
  HashTableRemoveAll(ht);
  free(ht->buckets);
  free(ht);
}

HashTable HashTableCopy(HashTable ht, CopyFunc *cf)
{
  HashTable new = HashTableCreate(ht->key_type, ht->ff);
  void *t;
  HashIter(b, e, t, ht) {
    HashEntry *entry = HashTableAddEntry(new, HashEntryKey(e), NULL);
    HashEntryValue(entry) = cf ? cf(t) : t;
  }
  return new;
}

/* taken from Tcl7.3. Produces a hash key for a string. */
static int HashString(char *string)
{
  int result = 0;
  char c;
  for(;;) {
    c = *string++;
    if(!c) break;
    result += (result<<3) + c;
  }
  return result;
}

/* Computes the hash bucket for the given key */
int HashBucket(HashTable ht, HashKey key)
{
  int i, b;
  if(ht->key_type == HASH_STRING) b = HashString(key);
  else {
    if(ht->key_type > 1) {
      b = *(int*)key;
      for(i=1;i<ht->key_type;i++) {
	b += ((int*)key)[i];
      }
    }
    else b = (int)key;
    b = (b * 1103515245) >> ht->down_shift;
  }
  return (b & ht->mask);
}

/* Used by HashTableFindEntry and HashTableLookupEntry. 
 * Returns the bucket index and HashEntry corresponding to key.
 */
static HashEntry *LookupEntry(HashTable ht, HashKey key, int *bucket)
{
  int b, i;
  HashEntry *e;

  b = HashBucket(ht, key);
  *bucket = b;
  if(ht->key_type == HASH_STRING) {
    for(e = ht->buckets[b]; e; e=e->next)
      if(!strcmp(key, e->key)) break;
  }
  else if(ht->key_type == HASH_WORD) {
    for(e = ht->buckets[b]; e; e=e->next)
      if(key == e->key) break;
  }
  else {
    for(e = ht->buckets[b]; e; e=e->next) {
      /* compare the arrays (key, e->key) */
      for(i=0;i<ht->key_type;i++) {
	if(((int*)key)[i] != ((int*)e->key)[i]) break;
      }
      if(i == ht->key_type) break;
    }
  }
  return e;
}

/* If ht has no entries matching the key, returns a new HashEntry 
 * corresponding to that key (which has been entered into the hash table) 
 * into which data may be placed, and sets *found_return to 0.
 * Otherwise, returns the matching entry and sets *found_return to 1.
 * found_return may be NULL.
 */
HashEntry *HashTableAddEntry(HashTable ht, HashKey key, int *found_return)
{
  int b;
  HashEntry *e;
  if(!ht) return NULL;
  e = LookupEntry(ht, key, &b);
  if(!e) {
    if(found_return) *found_return = 0;
    if(ht->key_type == HASH_STRING) {
      e = (HashEntry*)malloc(sizeof(HashEntry) + 
			     strlen(key) + 1 - sizeof(e->keydata));
      if(!e) {
	fprintf(stderr, "Out of memory; cannot add hash entry\n");
	return NULL;
      }
      e->key = e->keydata;
      strcpy(e->keydata, key);
    }
    else if(ht->key_type == HASH_WORD) {
      e = Allocate(1, HashEntry);
      if(!e) {
	fprintf(stderr, "Out of memory; cannot add hash entry\n");
	return NULL;
      }
      e->key = key;
    }
    else {
      e = (HashEntry*)malloc(sizeof(HashEntry) + 
			     ht->key_type*sizeof(int) - sizeof(e->keydata));
      if(!e) {
	fprintf(stderr, "Out of memory; cannot add hash entry\n");
	return NULL;
      }
      e->key = e->keydata;
      memcpy(e->keydata, key, ht->key_type*sizeof(int));
    }
    e->bucket = b;
    e->next = ht->buckets[b];
    ht->buckets[b] = e;

    ht->num_entries++;
    if(ht->num_entries >= REBUILD_MULTIPLIER * ht->num_buckets) 
      RebuildTable(ht);
  }
  else if(found_return) *found_return = 1;
  return e;
}

/* Returns the HashEntry matching the key, or NULL if there is none. */
HashEntry *HashTableFindEntry(HashTable ht, HashKey key)
{
  int b;
  if(!ht) return NULL;
  return LookupEntry(ht, key, &b);
}

/* Removes the entry from the hash table, and uses the ff to free the 
 * entry data.
 */
void HashTableRemoveEntry(HashTable ht, HashEntry *entry)
{
  HashEntry **e;

  if(!ht) return;
  for(e = &ht->buckets[entry->bucket]; *e != entry; e = &(*e)->next);
  *e = entry->next;
  ht->num_entries--;
  if(ht->ff) ht->ff(entry->value);
  free(entry);
}

/* Removes the entry from the hash table, without freeing the entry data. */
void HashTableRemoveNoFree(HashTable ht, HashEntry *entry)
{
  FreeFunc *ff = ht->ff;
  ht->ff = NULL;
  HashTableRemoveEntry(ht, entry);
  ht->ff = ff;
}

/* Remove all entries such that cf(data, value) == 0.
 * If cf == NULL, uses pointer equality (data == value).
 * Returns the number of nodes removed.
 */
int HashTableRemoveValueAll(HashTable ht, void *value, CmpFunc *cf)
{
  int b, count = 0;
  HashEntry **ep, *t;
  for(b = 0; b < ht->num_buckets; b++) {
    for(ep = &ht->buckets[b]; *ep;) {
      if((cf && !cf((*ep)->value, value)) ||
	 (!cf && ((*ep)->value == value))) {
	t = *ep;
	*ep = t->next;
	ht->num_entries--;
	if(ht->ff) ht->ff(t->value);
	free(t);
	count++;
      }
      else ep = &(*ep)->next;
    }
  }
  return count;
}

/* used to expand the table. */
static void RebuildTable(HashTable ht)
{
  HashEntry **old_buckets;
  HashEntry *e, *t;
  int i,b, old_size;

  old_size = ht->num_buckets;
  old_buckets = ht->buckets;

  ht->num_buckets *= 4;
  ht->buckets = Allocate(ht->num_buckets, HashEntry *);
  if(!ht->buckets) {
    fprintf(stderr, "Out of memory; cannot rebuild hash table\n");
    ht->buckets = old_buckets;
    ht->num_buckets = old_size;
    return;
  }
  ht->down_shift -= 2;
  ht->mask = (ht->mask << 2) + 3;
  for(i=0;i<ht->num_buckets;i++) ht->buckets[i] = NULL;
  
  /* rehash all entries */
  for(i=0;i<old_size;i++) {
    for(e=old_buckets[i];e;e=t) {
      b = HashBucket(ht, e->key);
      e->bucket = b;
      t = e->next;
      e->next = ht->buckets[b];
      ht->buckets[b] = e;
    }
  }
  free(old_buckets);
}

/* Returns the HashEntry immediately after entry in the HashTable,
 * or NULL if no such entry. 
 * If entry == NULL, returns the first entry in the table.
 */
HashEntry *HashTableNextEntry(HashTable ht, HashEntry *entry)
{
  int b;
  HashEntry *e;

  if(!entry) b = 0;
  else { 
    if(entry->next) return entry->next;
    b = entry->bucket + 1;
  }
  for(e = NULL; b < ht->num_buckets; b++) {
    if(e = ht->buckets[b]) break;
  }
  return e;
}
