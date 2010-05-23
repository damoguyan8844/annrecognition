/* Object class definition for Euclidean */

ObjClassField EuclideanFields[] = {
  { "vector-size", "int" }, /* will be initialized first */
  { "from", "int" },
  { "to", "int" },
  { "weights", "ptr array[?x] double" },
  { "field", "quark" },
  { NULL, NULL }
};

/* Implemented in euclidean.c */
extern GenericFunc EuclideanCon, EuclideanDes, EuclideanDistance;

ObjClassFunc EuclideanFuncs[] = {
  { "distance", EuclideanDistance },
  { "constructor", EuclideanCon },
  { "destructor", EuclideanDes },
  { NULL, NULL }
};

struct ObjClassStruct Euclidean = {
  "euclidean", sizeof(struct EuclideanData), 
  EuclideanFields, EuclideanFuncs, NULL
};

