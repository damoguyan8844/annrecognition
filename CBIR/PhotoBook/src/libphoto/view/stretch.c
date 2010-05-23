#include "photobook.h"
#include "obj_data.h"
#include <assert.h>

/* Functions *****************************************************************/

Ph_Image ViewStretch_Image(Ph_Object self, Ph_Member m)
{
  Ph_Image image;
  struct ViewStretchData *data = (struct ViewStretchData *)self->data;
  int i,j;
  uchar maximum, minimum;

  /* get the image from the member's field */
  PhGetField(m, data->field, image);
  assert(image);

  /* find max and min (assumes gray image) */
  maximum = minimum = image->data[0][0];
  for(i=0;i < image->height;i++) {
    for(j=0;j < image->width;j++) {
      if(image->data[i][j] > maximum) 
	maximum = image->data[i][j];
      else if(image->data[i][j] < minimum) 
	minimum = image->data[i][j];
    }
  }

  /* stretch the image */
  for(i=0;i < image->height;i++) {
    for(j=0;j < image->width;j++) {
      image->data[i][j] = 
	(uchar)((double)(image->data[i][j] - minimum) 
		/ (maximum - minimum) * 255 + 0.5);
    }
  }
  return image;
}

static void watchProc(Ph_Object self, char *field, void *userData)
{
  struct ViewStretchData *data = (struct ViewStretchData *)self->data;
  int a;
  Ph_Object obj;

  obj = PhLookupObject(self->phandle, data->field);
  if(!obj) {
    fprintf(stderr, "No such object: %s\n", data->field);
    return;
  }

  Ph_ObjGet(obj, Ph_StringQuark("height"), &a);
  Ph_ObjSet(self,Ph_StringQuark("height"), &a);

  Ph_ObjGet(obj, Ph_StringQuark("width"), &a);
  Ph_ObjSet(self,Ph_StringQuark("width"), &a);

  Ph_ObjGet(obj, Ph_StringQuark("channels"), &a);
  Ph_ObjSet(self,Ph_StringQuark("channels"), &a);
}

void ViewStretch_Con(Ph_Object self)
{
  Ph_ObjWatch(self, "field", watchProc, NULL);
}
