/* Object class definition for Peaks (part of Wold) */

ObjClassField PeaksFields[] = {
  { "num-peaks", "int" },
  { "nbr-size", "int" },
  { "peaks", "quark" },
  { NULL, NULL }
};

/* Implemented in peaks.c */
extern GenericFunc PeaksCon, PeaksDes, PeaksDistance;

ObjClassFunc PeaksFuncs[] = {
  { "distance", PeaksDistance },
  { "constructor", PeaksCon },
  { "destructor", PeaksDes },
  { NULL, NULL }
};

struct ObjClassStruct Peaks = {
  "peaks", sizeof(struct PeaksData), 
  PeaksFields, PeaksFuncs, NULL
};

