/* Object class definition for Min */

ObjClassField MinFields[] = {
  { "field", "quark" },
  { NULL, NULL }
};

/* Implemented in min.c */
extern GenericFunc MinDistance;

ObjClassFunc MinFuncs[] = {
  { "distance", MinDistance },
  { NULL, NULL }
};

struct ObjClassStruct MinClass = {
  "min", sizeof(struct MinData), 
  MinFields, MinFuncs, NULL
};

