/* Object class definition for View/Stretch */

ObjClassField ViewStretch_Fields[] = {
  { "field", "quark" },
  { NULL, NULL }
};

/* Implemented in view/stretch.c */
extern GenericFunc ViewStretch_Image, ViewStretch_Con;

ObjClassFunc ViewStretch_Funcs[] = {
  { "image", ViewStretch_Image },
  { "constructor", ViewStretch_Con },
  { NULL, NULL }
};

struct ObjClassStruct ViewStretch = {
  "stretch", sizeof(struct ViewStretchData),
  ViewStretch_Fields, ViewStretch_Funcs, &View
};
