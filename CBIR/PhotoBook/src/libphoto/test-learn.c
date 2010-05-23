#include "photobook.h"
#include <tpm/stream.h>
#include <tpm/gtree.h>

#define CART 0
#define WITHLABEL 0

typedef struct CoverStruct {
  int low, high;
  int tree, label;
  Tree node;
} Cover;

static void wf(FileHandle h, Cover *d)
{
  fprintf(h, "%d", d->label);
}

extern Tree LearnedTree(void);

void main(void)
{
  char *db_name = "textures";
  Ph_Handle phandle;
  char str[100];
  int i;
  char *labels;
  Ph_Member member;
  List members, trees;
  FileHandle fp;
  
  phandle = Ph_Startup();
  Ph_SetDatabase(phandle, db_name);

#if 0
  trees = ListCreate(NULL);
  ListAddRear(trees, "tree1");
  ListAddRear(trees, "tree2");
#elif 1
  trees = ListCreate(NULL);
  ListAddRear(trees, "sar.tree");
#else
  trees = Ph_GetDBTrees(phandle);
#endif
  LearnInit(phandle, trees, 2);
  ListFree(trees);
  AddLabel();
  AddLabel();

  LearnReadBias("bias.in");
  LearnSaveBias("bias.out");
  for(;;) {
    printf("Member > ");
    gets(str);
    if(!str[0]) break;
    member = Ph_MemberWithName(phandle, str);
    if(!member) {
      printf("No such member\n");
      continue;
    }
    printf("Class? > ");
    gets(str);
    if(!str[0]) {
      LearnPosEx(member, 0);
      LearnNegEx(member, 1);
    }
    else {
      LearnNegEx(member, 0);
      LearnPosEx(member, 1);
    }
    LearnUpdate();
    /* show all label bindings */
    members = Ph_ListMembers(phandle);
#if WITHLABEL
    labels = Ph_MemWithLabel(NULL, 0);
#endif
    {ListIter(p, member, members) {
#if WITHLABEL
      if(labels[Ph_MemIndex(member)]) printf("%s\n", Ph_MemName(member));
#else
      labels = Ph_MemLabels(member, NULL);
      if(labels[0]) printf("%s\n", Ph_MemName(member));
      free(labels);
#endif
    }}
#if WITHLABEL
    free(labels);
#endif
    ListFree(members);

#if CART
    fp = FileOpen("tree", "w");
    TreeWrite(fp, LearnedTree(), (WriteFunc*)wf);
    FileClose(fp);
#endif
  }
  LearnSaveBias("bias.out");
  LearnFree();
  Ph_Shutdown(phandle);
}
