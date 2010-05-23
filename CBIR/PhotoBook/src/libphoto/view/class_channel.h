/* Object class definition for View/Channel */

ObjClassField ViewChannel_Fields[] =
{
  { "field", "quark" },
  { "channel", "int"},
  { NULL, NULL},
};

/* implemented in view/channel.c */
extern GenericFunc ViewChannel_Image, ViewChannel_Con;

ObjClassFunc ViewChannel_Funcs[] =
{
  { "image", ViewChannel_Image},
  { "constructor", ViewChannel_Con},
  { NULL,NULL}
}; 

struct ObjClassStruct ViewChannel =
{
  "channel", sizeof(struct ViewChannelData),
  ViewChannel_Fields, ViewChannel_Funcs, &View
};
