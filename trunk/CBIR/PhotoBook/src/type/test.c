#include "type.h"

main()
{
  char str[100];
  Type t;

  TypeTableCreate();
  TypeDefineTpm();

  printf("Enter a type:\n");
  fgets(str, 100, stdin);
  t = TypeParse(str);

  printf("Type was: ");
  TypeUnparse(stdout, t);
  printf("\n");
  printf("align: %d\n", TypeAlign(t));
  printf("size: %d\n", TypeSize(t));

  TypeFree(t);
  TypeTableFree();
}

