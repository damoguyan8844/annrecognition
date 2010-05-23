/* StringQuark implementation. Useful for applications which do a lot of
 * string comparisons on a small set of strings. By quarking the strings,
 * strcmp can be replaced by ==.
 */

#include <tpm/hash.h>

/* Globals *******************************************************************/

static HashTable quarks = NULL;

/* Prototypes ****************************************************************/
void StringQuarksCreate(void);
void StringQuarksFree(void);
void StringQuarksFlush(void);
char *StringQuark(char *s);

/* Functions *****************************************************************/

void StringQuarksCreate(void)
{
  quarks = HashTableCreate(HASH_STRING, NULL);
}

void StringQuarksFree(void)
{
  HashTableFree(quarks);
  quarks = NULL;
}

void StringQuarksFlush(void)
{
  HashTableFlush(quarks);
}

/* Returns a quarked version of s, i.e. returns identical pointer values
 * for all equivalent strings.
 * (This naming convention is in keeping with the quantum mechanical theory 
 * that all equivalent quarks are identical.)
 */
char *StringQuark(char *s)
{
  int found;
  HashEntry *e = HashTableAddEntry(quarks, s, &found);
  return HashEntryKey(e);
}
