/* Object class definition for Combination */

ObjClassField CombinationFields[] = {
  { "num-metrics", "int" },
  { "metrics", "ptr array[?x] quark" },
  { "factors", "ptr array[?x] double" },
  { "weights", "ptr array[?x] double" },
  { NULL, NULL }
};

/* Implemented in combination.c */
extern GenericFunc CombinationCon, CombinationDes, CombinationDistance;

ObjClassFunc CombinationFuncs[] = {
  { "distance", CombinationDistance },
  { "constructor", CombinationCon },
  { "destructor", CombinationDes },
  { NULL, NULL }
};

struct ObjClassStruct Combination = {
  "combination", sizeof(struct CombinationData), 
  CombinationFields, CombinationFuncs, NULL
};

