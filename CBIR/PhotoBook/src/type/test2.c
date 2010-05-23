#include "type.h"
#include "match.h"

main()
{
  char str[100];
  Type t1, t2;
  Binding b1, b2;

  TypeTableCreate();
  TypeDefineTpm();

  printf("Enter actual type:\n");
  fgets(str, 100, stdin);
  t1 = TypeParse(str);

/*
  b1 = BindingUnknowns(t1);
  BindingWrite(stdout, b1);
  BindingFree(b1);
*/

  printf("Enter desired type:\n");
  fgets(str, 100, stdin);
  t2 = TypeParse(str);

  if(!TypeMatch(t1, t2, &b1, &b2)) {
    printf("Do not match\n");
  }
  else {
    BindingWrite(stdout, b1);
    printf("actual, after binding: ");
    BindingApply(b1, t1);
    TypeUnparse(stdout, t1);
    printf("\n");
    BindingWrite(stdout, b2);
    printf("desired, after binding: ");
    BindingApply(b2, t2);
    TypeUnparse(stdout, t2);
    printf("\n");
    printf("b1 without vars:\n");
    BindingNoVars(b1);
    BindingWrite(stdout, b1);
  }
  BindingFree(b1);
  BindingFree(b2);

  TypeFree(t1);
  TypeFree(t2);
  TypeTableFree();
}

