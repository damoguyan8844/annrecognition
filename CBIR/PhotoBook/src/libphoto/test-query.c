#include "photobook.h"

void PrintList(List list)
{
  char *s;

  if(!list) printf("<none>\n");
  else {
    ListIter(p, s, list) {
      printf("%s\n", s);
    }
    ListFree(list);
  }
}

void main(int argc, char *argv[])
{
  char *db_name = "textures";
  char *metric_name = NULL;
  Ph_Object metric;
  Ph_Handle phandle;
  List query_list;
  char query[100];
  Ph_Member member;
  List members;
  Type type;
  char *str;
  
  if(argc > 1) {
    db_name = argv[1];
  }
  if(argc > 2) {
    metric_name = argv[2];
  }
  phandle = Ph_Startup();

  printf("Metrics:\n");
  PrintList(Ph_GetMetrics());
  printf("Views:\n");
  PrintList(Ph_GetViews());

  Ph_SetDatabase(phandle, db_name);

  printf("DB Metrics:\n");
  PrintList(Ph_GetDBMetrics(phandle));
  printf("DB Views:\n");
  PrintList(Ph_GetDBViews(phandle));

  metric = Ph_SetMetric(phandle, metric_name);
  if(!metric) exit(1);
  printf("Metric: %s\n", Ph_ObjName(metric));

  query_list = ListCreate(NULL);
  ListAddRear(query_list, query);

  for(;;) {
    printf("Query > ");
    gets(query);
    if(!query[0]) break;
    if(Ph_SetQuery(phandle, query_list) == PH_ERROR) {
      printf("Erroneous query\n");
      continue;
    }

    members = Ph_ListWorkingSet(phandle);
    {ListIter(p, member, members) {
      printf("%s\n", Ph_MemName(member));
    }}
    printf("\n");
    ListFree(members);
  }
  ListFree(query_list);
  Ph_Shutdown(phandle);
}
