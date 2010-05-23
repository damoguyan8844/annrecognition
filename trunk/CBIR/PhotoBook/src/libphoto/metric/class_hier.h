/* Object class definition for Hierarchical */

ObjClassField HierFields[] = {
  { "tree", "quark" },
  { NULL, NULL }
};

/* Implemented in hier.c */
extern GenericFunc HierCon, HierDes, HierDistance;

ObjClassFunc HierFuncs[] = {
  { "distance", HierDistance },
  { "constructor", HierCon },
  { "destructor", HierDes },
  { NULL, NULL }
};

struct ObjClassStruct Hierarchical = {
  "tree", sizeof(struct HierData), 
  HierFields, HierFuncs, NULL
};

