#include "photobook.h"
#include "obj_data.h"
#include <assert.h>

/**********Functions***********/

Ph_Image ViewChannel_Image(Ph_Object self, Ph_Member m)
{
  Ph_Image imageold, imagenew;
  struct ViewChannelData *data = (struct ViewChannelData *)self->data;
  int i, j;
  
  /*Get the image from the members field*/
  PhGetField(m, data->field, imageold);
  assert (imageold);
  
  /*Creates new image that contains only one channel*/
  imagenew= Ph_ImageCreate(imageold->height,imageold->width,1);

  /*Copies the correct channel from imageold to imagenew*/
  for(i=0;i<imageold->height;i++) {
    for(j=0;j<imageold->width;j++) {
      imagenew->data[i][j]=imageold->data[i][3*j-1+ data->channel];
    }
  }

  /*Get rid of old image*/
  Ph_ImageFree(imageold);

  return imagenew;
}

static void watchProc(Ph_Object self, char *field, void *userData)
{
  struct ViewChannelData *data = (struct ViewChannelData *)self->data;
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

  a = 1;
  Ph_ObjSet(self,Ph_StringQuark("channels"),&a);
}

void ViewChannel_Con(Ph_Object self)
{
  Ph_ObjWatch(self, "field", watchProc, NULL);
}


























