#include "photobook.h"

void main(void)
{
  char *db_name = "textures";
  Ph_Handle phandle;
  Ph_Member member;
  Type type;
  char str[1000], field[1000], *s;
  
  phandle = Ph_Startup();
  Ph_SetDatabase(phandle, db_name);

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
    gets(field);

    printf("Value > ");
    gets(str);

    if(!str[0]) {
      s = Ph_MemGetAnn(phandle, member, field);
      if(!s) {
	printf("No such field\n");
      }
      else {
	printf("%s\n", s);
      }
    }
    else {
      Ph_MemSetAnn(phandle, member, field, str, 1);
    }
  }
  Ph_Shutdown(phandle);
}
