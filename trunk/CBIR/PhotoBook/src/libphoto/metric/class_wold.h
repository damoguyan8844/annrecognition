/* Object class definition for Wold */

ObjClassField WoldFields[] = {
  { "num-peaks", "int" },
  { "nbr-size", "int" },
  { "peaks", "quark" },
  { "alt-metric", "quark" },
  { "orien-type", "quark" },
  { "orien-label", "quark" },
  { "tamura-vector", "quark" },
  { NULL, NULL }
};

/* Implemented in wold.c */
extern GenericFunc WoldCon, WoldDes, WoldDistance;

ObjClassFunc WoldFuncs[] = {
  { "distance", WoldDistance },
  { "constructor", WoldCon },
  { "destructor", WoldDes },
  { NULL, NULL }
};

struct ObjClassStruct Wold = {
  "wold", sizeof(struct WoldData), 
  WoldFields, WoldFuncs, NULL
};

