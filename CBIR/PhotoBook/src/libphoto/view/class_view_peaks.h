/* Object class definition for View/Peaks */

ObjClassField ViewPeaks_Fields[] = {
  { "peaks", "quark" },
  { NULL, NULL }
};

/* Implemented in view/view_peaks.c */
extern GenericFunc ViewPeaks_Image, ViewPeaks_Con;

ObjClassFunc ViewPeaks_Funcs[] = {
  { "image", ViewPeaks_Image },
  { "constructor", ViewPeaks_Con },
  { NULL, NULL }
};

struct ObjClassStruct ViewPeaks = {
  "peaks", sizeof(struct ViewPeaksData),
  ViewPeaks_Fields, ViewPeaks_Funcs, &View
};
