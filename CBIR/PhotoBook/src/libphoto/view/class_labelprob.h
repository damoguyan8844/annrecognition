/* Object class definition for View/LabelProb */

ObjClassField ViewLabelProb_Fields[] = {
  { "label", "int" },
  { NULL, NULL }
};

/* Implemented in view/labelprob.c */
extern GenericFunc ViewLabelProb_Image, ViewLabelProb_Con;

ObjClassFunc ViewLabelProb_Funcs[] = {
  { "image", ViewLabelProb_Image },
  { "constructor", ViewLabelProb_Con },
  { NULL, NULL }
};

struct ObjClassStruct ViewLabelProb = {
  "label_prob", sizeof(struct ViewLabelProbData),
  ViewLabelProb_Fields, ViewLabelProb_Funcs, &View
};
