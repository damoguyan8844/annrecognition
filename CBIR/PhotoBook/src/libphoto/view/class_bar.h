/* Object class definition for View/Bar */
/* inherits from View/Image */

ObjClassField ViewBar_Fields[] = {
  { "vector", "quark" },
  { "spacing", "int" },
  { "maximum", "double" },
  { "minimum", "double" },
  { "color", "3 char" },
  { NULL, NULL }
};

/* Implemented in view/bar.c */
extern GenericFunc ViewBar_Image, ViewBar_Con;

ObjClassFunc ViewBar_Funcs[] = {
  { "image", ViewBar_Image },
  { "constructor", ViewBar_Con },
  { NULL, NULL }
};

struct ObjClassStruct ViewBar = {
  "bar_graph", sizeof(struct ViewBarData),
  ViewBar_Fields, ViewBar_Funcs, &ViewImage
};
