#include "photobook.h"
#include "obj_data.h"
#include <assert.h>

/*************Functions*************/

Ph_Image ViewZoom_Image(Ph_Object self, Ph_Member m)
{
  Ph_Image imageold, imagenew;
  struct ViewZoomData *data =(struct ViewZoomData *)self->data;
  int f;

  /*Get the image from the members field*/
  PhGetField(m, data->field, imageold);
  assert (imageold);

  imagenew = Ph_ImageZoom(imageold,data->zfact);
  Ph_ImageFree(imageold);
  return imagenew;
}

static void watchProc(Ph_Object self, char *field, void *userData)
{
  struct ViewZoomData *data =(struct ViewZoomData *)self->data;
  int a;
  double factor;
  Ph_Object obj;

  obj = PhLookupObject(self->phandle, data->field);
  if(!obj) {
    fprintf(stderr, "No such object: %s\n", data->field);
    return;
  }

  if (data->zfact==0) data->zfact=1;
  if(data->zfact >= 0) factor = data->zfact;
  else factor = -1.0/data->zfact;

  Ph_ObjGet(obj, Ph_StringQuark("height"), &a);
  a *= factor;
  Ph_ObjSet(self,Ph_StringQuark("height"), &a);

  Ph_ObjGet(obj, Ph_StringQuark("width"), &a);
  a *= factor;
  Ph_ObjSet(self,Ph_StringQuark("width"),&a);

  Ph_ObjGet(obj, Ph_StringQuark("channels"), &a);
  Ph_ObjSet(self,Ph_StringQuark("channels"),&a);
}

void ViewZoom_Con(Ph_Object self)
{
  Ph_ObjWatch(self, "field", watchProc, NULL);
  Ph_ObjWatch(self, "zfact", watchProc, NULL);
}





