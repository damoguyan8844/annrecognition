#include "parse_value.h"

main()
{
  char str[100], *s;
  Type t;
  void *data;

  TypeTableCreate();
  TypeParseTpm();
  TypeDefineTpm();

  printf("Enter a type:\n");
  fgets(str, 100, stdin);
  t = TypeParse(str);

  printf("Enter a value:\n");
  fgets(str, 100, stdin);
  data = NULL;
  if(!ParseValue(t, str, &data)) {
    printf("Parse error\n");
    exit(1);
  }

  printf("Type was: ");
  TypeUnparse(stdout, t);
  printf("\n");
  printf("Value was: ");
  s = UnparseValue(t, data);
  if(!s) {
    printf("Unparse error\n");
    exit(1);
  }
  printf("%s\n", s);
  free(s);
  free(data);

  TypeFree(t);
  TypeTableFree();
}

