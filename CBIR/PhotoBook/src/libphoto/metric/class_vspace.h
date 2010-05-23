/* Object class definition for VSpace */

ObjClassField VSpaceFields[] = {
  { "rows", "int" },
  { "cols", "int" },
  { "corr", "quark" },
  { "basis", "quark" },
  { NULL, NULL }
};

/* Implemented in vspace.c */
extern GenericFunc VSpaceDistance;

ObjClassFunc VSpaceFuncs[] = {
  { "distance", VSpaceDistance },
  { NULL, NULL }
};

struct ObjClassStruct VSpace = {
  "vspace", sizeof(struct VSpaceData), 
  VSpaceFields, VSpaceFuncs, NULL
};

