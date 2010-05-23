#include "photobook.h"
#include <type/parse_value.h>

void main(void)
{
  char *db_name = "textures";
  Ph_Handle phandle;
  Ph_Member member;
  char str[1000], *s;
  void *data;
  ObjField *of;
  ObjTField *otf;
  Type vtype, type;
  List members;

  debug = 1;
  phandle = Ph_Startup();
  Ph_SetDatabase(phandle, db_name);

#if 0
  /* add transient field to all members */
  vtype = Ph_TypeQuark("ptr array[?x] double");
  members = Ph_ListMembers(phandle);
  {ListIter(p, member, members) {
    Ph_ObjAddField(member, "tamura_coeff", vtype, NULL, NULL, NULL, 0, 0);
  }}
  ListFree(members);
#endif

  for(;;) {
    printf("\nMember > ");
    gets(str);
    if(!str[0]) break;
    member = Ph_MemberWithName(phandle, str);
    if(!member) {
      printf("No such member\n");
      continue;
    }

    printf("Field > ");
    gets(str);
    if(Ph_ObjGet(member, str, &data) == PH_ERROR) {
      printf("No data\n");
      continue;
    }

    of = PhObjField(member, str);
    if(!of) {
      otf = PhObjTField(member, str);
      of = (ObjField*)otf;
    }
    type = of->type;
    printf("Type is ");
    TypeUnparse(stdout, type);
    printf("\n");
    s = UnparseValue(type, &data);
    if(!s) {
      printf("Cannot unparse\n");
    }
    else {
      printf("Value: %s\n", s);
      free(s);
    }
  }
#if 0
  TypeFree(vtype);
#endif
  Ph_Shutdown(phandle);
}
