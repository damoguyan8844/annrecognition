/* Object class definition for View/Tsw */

ObjClassField ViewTsw_Fields[] = {
  { "field", "quark" },
  { "maximum", "double" },
  { NULL, NULL }
};

/* Implemented in view/tsw.c */
extern GenericFunc ViewTsw_Image, ViewTsw_Con;

ObjClassFunc ViewTsw_Funcs[] = {
  { "image", ViewTsw_Image },
  { "constructor", ViewTsw_Con },
  { NULL, NULL }
};

struct ObjClassStruct ViewTsw = {
  "tsw_tree", sizeof(struct ViewTswData),
  ViewTsw_Fields, ViewTsw_Funcs, &View
};
