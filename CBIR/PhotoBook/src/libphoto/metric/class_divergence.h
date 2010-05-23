/* Object class definition for Divergence */
/* Inherits from Mahalanobis */

ObjClassField DiverFields[] = {
  { "covar", "quark" },
  { NULL, NULL }
};

/* Implemented in divergence.c */
extern GenericFunc DiverDistance;

ObjClassFunc DiverFuncs[] = {
  { "distance", DiverDistance },
  { NULL, NULL }
};

struct ObjClassStruct Divergence = {
  "divergence", sizeof(struct DiverData), 
  DiverFields, DiverFuncs, &Mahalanobis
};

