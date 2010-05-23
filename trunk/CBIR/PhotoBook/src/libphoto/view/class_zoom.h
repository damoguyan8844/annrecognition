/*Object class definition for View/Zoom */

ObjClassField ViewZoom_Fields[] =
{
  { "field","quark"},
  { "zfact","int"},
  { NULL,NULL}
};

/*Implemented in view/zoom.c */
extern GenericFunc ViewZoom_Image, ViewZoom_Con;
 
ObjClassFunc ViewZoom_Funcs[] =
{
  { "image", ViewZoom_Image},
  { "constructor", ViewZoom_Con},
  { NULL,NULL}
};

struct ObjClassStruct ViewZoom =
{
  "zoom", sizeof(struct ViewZoomData),
  ViewZoom_Fields, ViewZoom_Funcs, &View
};

