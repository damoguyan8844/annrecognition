#include <tpm/hash.h>
#include <type/type.h>

/* Globals *******************************************************************/

static HashTable quarks = NULL;

/* Prototypes ****************************************************************/
void TypeQuarksCreate(void);
void TypeQuarksFree(void);
void TypeQuarksFlush(void);
Type TypeQuark(char *type_s);

/* Functions *****************************************************************/

void TypeQuarksCreate(void)
{
  quarks = HashTableCreate(HASH_STRING, (FreeFunc*)TypeFree);
}

void TypeQuarksFree(void)
{
  HashTableFree(quarks);
  quarks = NULL;
}

void TypeQuarksFlush(void)
{
  HashTableFlush(quarks);
}

/* Returns the quarked version of TypeParse(type_s) */
Type TypeQuark(char *type_s)
{
  int found;
  Type type;
  HashEntry *e;

  /* is this string in the table? */
  e = HashTableFindEntry(quarks, type_s);
  if(e) return HashEntryValue(e);

  /* parse it into a type */
  type = TypeParse(type_s);
  if(!type) return NULL;

  /* enter it into the table */
  e = HashTableAddEntry(quarks, type_s, &found);
  return HashEntryValue(e) = type;
}
