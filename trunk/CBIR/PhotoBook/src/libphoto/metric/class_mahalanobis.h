/* Object class definition for Mahalanobis */

ObjClassField MahalFields[] = {
  { "vector-size", "int" },
  { "coeff", "quark" },
  { "icovar", "quark" },
  { "mask", "ptr array[?x] char" },
  { NULL, NULL }
};

/* Implemented in mahalanobis.c */
extern GenericFunc MahalDistance, MahalCon, MahalDes;

ObjClassFunc MahalFuncs[] = {
  { "distance", MahalDistance },
  { "constructor", MahalCon },
  { "destructor", MahalDes },
  { NULL, NULL }
};

struct ObjClassStruct Mahalanobis = {
  "mahalanobis", sizeof(struct MahalData), 
  MahalFields, MahalFuncs, NULL
};

