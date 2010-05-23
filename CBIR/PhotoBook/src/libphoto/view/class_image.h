/* Object class definition for View/Image */

ObjClassField ViewImage_Fields[] = {
  { "field", "quark" },
  { NULL, NULL }
};

/* Implemented in view/image.c */
extern GenericFunc ViewImage_Image, ViewImage_Con;

ObjClassFunc ViewImage_Funcs[] = {
  { "image", ViewImage_Image },
  { "constructor", ViewImage_Con },
  { NULL, NULL }
};

struct ObjClassStruct ViewImage = {
  "image", sizeof(struct ViewImageData),
  ViewImage_Fields, ViewImage_Funcs, &View
};
