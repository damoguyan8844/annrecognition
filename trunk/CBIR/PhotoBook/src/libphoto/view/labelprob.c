#include "photobook.h"
#include "obj_data.h"
#include <assert.h>

/* Functions *****************************************************************/

Ph_Image ViewLabelProb_Image(Ph_Object self, Ph_Member m)
{
  PhImageFunc *func;
  Ph_Image image;
  struct ViewLabelProbData *data = (struct ViewLabelProbData *)self->data;
  float *prob;
  uchar value;

  /* call the superclass to initialize the image */
  func = (PhImageFunc*)PhObjFunc(self->super, "image");
  assert(func);
  image = func(self->super, m);

  /* get the label probability */
  prob = Ph_MemLabelProb(m);
  value = (uchar)(prob[data->label]*255);
  Ph_ImagePutBlock(image, 0, 0, image->height, image->width,
		   value, value, value);

  return image;
}

void ViewLabelProb_Con(Ph_Object self)
{
  struct ViewLabelProbData *data = (struct ViewLabelProbData *)self->data;
  /* set defaults */
  data->label = 0;
}
