#include "photobook.h"
#include "obj_data.h"
#include <math.h> /* for ceil() */
#include <assert.h>

/* Functions *****************************************************************/

Ph_Image ViewBar_Image(Ph_Object self, Ph_Member m)
{
  PhImageFunc *func;
  Ph_Image image;
  struct ViewImageData *sdata = (struct ViewImageData *)self->super->data;
  struct ViewBarData *data = (struct ViewBarData *)self->data;
  int bar_width, vec_length, space, x, i, inc;
  double *vector, factor;

  if(sdata->field[0]) {
    /* call the superclass (image) to initialize the image */
    func = (PhImageFunc*)PhObjFunc(self->super, "image");
    assert(func);
    image = func(self->super, m);
  }
  if(!sdata->field[0] || !image) {
    /* call the super-superclass (view) to initialize the image */
    func = (PhImageFunc*)PhObjFunc(self->super->super, "image");
    assert(func);
    image = func(self->super->super, m);
    assert(image);
    /* clear to black */
    memset(image->data[0], 0, image->width*image->height*image->channels);
  }

  if(Ph_ObjGet(m, data->vector_field, &vector) == PH_ERROR) {
    fprintf(stderr, "Error getting %s field for %s\n", 
	    data->vector_field, Ph_MemName(m));
    Ph_ImageFree(image);
    return NULL;
  }
  vec_length = TypeInt(TypeType(
		 Ph_ObjField(m, data->vector_field)->type, 0), 0);

  /* determine the bar_width and space */
  x = 0;
  inc = 1;
  bar_width = image->width / vec_length;
  if(bar_width == 0) {
    bar_width = 1;
    space = 0;
    inc = (int)ceil((double)vec_length / image->width);
  }
  else {
    bar_width /= data->spacing + 1;
    if(bar_width == 0) {
      space = image->width / vec_length / 2;
      bar_width = space;
      if(bar_width == 0) {
	bar_width = 1;
	x = (image->width - vec_length) / 2;
      }
    }
    else 
      space = bar_width * data->spacing;
  }

  factor = image->height / (data->maximum - data->minimum);
  for(i=0;i<vec_length;i+=inc,x+=bar_width+space) {
    int j, height;
    double sum = 0.0;
    /* average together inc elements */
    for(j=0;j<inc;j++) sum += vector[i+j];
    sum /= inc;
    height = (int)((sum - data->minimum) * factor + 0.5);
    if(height <= 0) continue;
    if(height > image->height) height = image->height;
    Ph_ImagePutBlock(image, image->height-height, x, height, bar_width, 
		     data->color[0],
		     data->color[1],
		     data->color[2]);
  }

  return image;
}

void ViewBar_Con(Ph_Object self)
{
  struct ViewBarData *data = (struct ViewBarData *)self->data;
  /* set defaults */
  ((struct ViewImageData *)self->super->data)->field = "";
  data->color[0] = 255;
  data->color[1] = 255;
  data->color[2] = 255;
  data->spacing = 1;
  data->maximum = 1.0;
  data->minimum = 0.0;
}
