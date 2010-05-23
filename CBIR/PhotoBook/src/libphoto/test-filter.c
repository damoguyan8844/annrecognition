#include "photobook.h"

void main(int argc, char *argv[])
{
  char *db_name = "textures";
  Ph_Handle phandle;
  List members;
  char filter[100];
  int i;

  if(argc > 1) db_name = argv[1];
  phandle = Ph_Startup();
  Ph_SetDatabase(phandle, db_name);

  for(;;) {
    printf("Filter > ");
    gets(filter);
    if(!strcmp(filter, "exit")) break;
    if(Ph_SetFilter(phandle, filter) == PH_ERROR) {
      printf("%s\n", phandle->error_string);
      continue;
    }
    members = Ph_ListWorkingSet(phandle);
/*
    for(i=0;i<phandle->ws_members;i++)
      printf("%s\n", Ph_MemName(phandle->working_set[i]));
*/
    {Ph_Member member;ListIter(p, member, members) {
      printf("%s\n", Ph_MemName(member));
    }}
    printf("\n");
    ListFree(members);
  }
  Ph_Shutdown(phandle);
}
