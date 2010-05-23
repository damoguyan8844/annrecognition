#include <ctype.h>
#if __vax
#include <stdlib.h>
#include <string.h>

/*
>     files.vaxo: Undefined symbol _remove referenced from text segment
>     parsers.vaxo: Undefined symbol _strtol referenced from text segment
>     parsers.vaxo: Undefined symbol _strtod referenced from text segment
>     streams.vaxo: Undefined symbol _strtoul referenced from text segment
>     arlotje.vaxo: Undefined symbol _strstr referenced from text segment
>     arlotje.vaxo: Undefined symbol _strstr referenced from text segmen
*/

void remove(filename)
     char *filename;
{
  char *command; command=malloc((strlen(filename)+3)*sizeof(char));
  strcpy(command,"rm "); strcat(command,filename);
  system(command);
  free(command);
}

char *strstr(src,sub)
     char *src, *sub;
{
  int size; size=strlen(sub);
  while ((*src) != '\0')
    if (strncmp(src,sub,size) == 0) return src;
    else src++;
  return NULL;
}

#endif

/* TPM: 4/13/94 */
#ifdef sun
unsigned long strtoul(nptr, eptr, base)
char *nptr, **eptr;
int base;
{
   return strtol(nptr, eptr, base);
}

void *memmove(s1, s2, n)
void *s1;
const void *s2;
int n;
{
  char *p1=(char*)s1, *p2=(char*)s2;

  if(s1 > s2) {
    p1+=n-1; p2+=n-1;
    while(n--) {
      *p1-- = *p2--;
    }
  }
  else {
    while(n--) {
      *p1++ = *p2++;
    }
  }
}
#endif

int framer_strcmp_ci(char *ptr1,char *ptr2)
{
   register char c1, c2;
   while (((c1=(tolower(*(ptr1++)))) != '\0') && ((c2=(tolower((*(ptr2++))))) != '\0') &&
          (c1 == c2));
   if (c1 == '\0') c2=(*(ptr2++));
   return ((c1 == c2) ? 0 : (((c1 > c2) ? 1 : -1)));
}

