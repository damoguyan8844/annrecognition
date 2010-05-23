#include "photobook.h"
#include "obj_data.h"
#include <math.h> /* for pow() */

extern void TswCutTree(double *v, int levels, float cutoff);

/* Functions *****************************************************************/

static void DrawQuadTree(Ph_Image image, int y, int x, int height, int width,
			 double *v, int levels, double max_value)
{
  int size;
  unsigned char c;
  if(*v > 0.0) {
    if(*v > max_value) c = 255;
    else c = RealToGray(*v / max_value);
    Ph_ImagePutBlock(image, y, x, height, width, c, c, c);
  }
  if(levels == 1) return;
  size = (int)((pow(4.0, (double)levels-1)-1)/3);
  v++;
  DrawQuadTree(image, y,          x, 
               height/2, width/2, v, levels-1, max_value);
  v+=size;
  DrawQuadTree(image, y,          x+width/2, 
               height/2, width/2, v, levels-1, max_value);
  v+=size;
  DrawQuadTree(image, y+height/2, x, 
               height/2, width/2, v, levels-1, max_value);
  v+=size;
  DrawQuadTree(image, y+height/2, x+width/2, 
               height/2, width/2, v, levels-1, max_value);
}

Ph_Image ViewTsw_Image(Ph_Object self, Ph_Member m)
{
  Ph_Image image;
  struct ViewTswData *data = (struct ViewTswData *)self->data;
  struct ViewData *sdata = (struct ViewData *)self->super->data;
  struct TswData *tdata;
  Ph_Object obj;
  double *v, maximum;
  char *s;

  /* Get the metric object */
  obj = PhLookupObject(self->phandle, data->field);
  if(!obj) {
    fprintf(stderr, "No such object: %s\n", data->field);
    return NULL;
  }
  /* better be a Tsw object! */
  if(strcmp("tsw", Ph_ObjClass(obj))) {
    fprintf(stderr, "ViewTsw: field must be a tsw metric\n");
    return NULL;
  }
  tdata = (struct TswData *)obj->data;

  /* Get the tree vector */
  PhGetField(m, tdata->field, v);
  v = ArrayCopy(v, (int)((pow(4.0, (double)tdata->levels)-1)/3), sizeof(double));
  TswCutTree(v, tdata->levels, tdata->cutoff);

  image = Ph_ImageCreate(sdata->height, sdata->width, sdata->channels);

  DrawQuadTree(image, 0, 0, image->height, image->width,
	       v, tdata->levels, data->maximum);
  free(v);

  return image;
}

static void watchProc(Ph_Object self, char *field, void *userData)
{
  struct ViewTswData *data = (struct ViewTswData *)self->data;
  Ph_Object obj;

  obj = PhLookupObject(self->phandle, data->field);
  if(!obj) {
    fprintf(stderr, "No such object: %s\n", data->field);
    return;
  }
}

void ViewTsw_Con(Ph_Object self)
{
  struct ViewTswData *data = (struct ViewTswData *)self->data;
  data->maximum = 200;
  Ph_ObjWatch(self, "field", watchProc, NULL);
}
