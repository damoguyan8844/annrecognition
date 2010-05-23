/* Object class definition for TSW */

ObjClassField TswFields[] = {
  { "field", "quark" },
  { "cutoff", "float" },
  { "keep", "int" },
  { "levels", "int" },
  { NULL, NULL }
};

/* Implemented in tsw.c */
extern GenericFunc TswDistance, TswCon;

ObjClassFunc TswFuncs[] = {
  { "distance", TswDistance },
  { "constructor", TswCon },
  { NULL, NULL }
};

struct ObjClassStruct Tsw = {
  "tsw", sizeof(struct TswData), 
  TswFields, TswFuncs, NULL
};

