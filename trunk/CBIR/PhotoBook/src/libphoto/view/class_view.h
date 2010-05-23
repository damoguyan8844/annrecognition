/* Object class definition for View */

ObjClassField View_Fields[] = {
  { "height", "int" },
  { "width", "int" },
  { "channels", "int" },
  { NULL, NULL }
};

/* Implemented in view.c */
extern GenericFunc View_Image, View_Con;

ObjClassFunc View_Funcs[] = {
  { "image", View_Image },
  { "constructor", View_Con },
  { NULL, NULL }
};

struct ObjClassStruct View = {
  "view", sizeof(struct ViewData),
  View_Fields, View_Funcs, NULL
};
