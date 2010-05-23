#include "photobook.h"
#include "obj_data.h"
#include <assert.h>

/* Functions *****************************************************************/

Ph_Image ViewPeaks_Image(Ph_Object self, Ph_Member m)
{
  PhImageFunc *func;
  Ph_Image image;
  struct ViewPeaksData *data = (struct ViewPeaksData *)self->data;
  struct ViewData *sdata = (struct ViewData *)self->super->data;
  struct PeaksData *mdata;
  Ph_Object obj;
  double *peaks, max_value;
  int i;

  /* Get the metric object */
  obj = PhLookupObject(self->phandle, data->peaks);
  if(!obj) {
    fprintf(stderr, "No such object: %s\n", data->peaks);
    return NULL;
  }
  /* better be a Peaks object! */
  if(strcmp("peaks", Ph_ObjClass(obj))) {
    fprintf(stderr, "ViewPeaks: peaks must be a peaks metric\n");
    return NULL;
  }
  mdata = (struct PeaksData *)obj->data;

  /* Get the peaks vector */
  PhGetField(m, mdata->peaks, peaks);

  /* call the superclass to initialize the image */
  func = (PhImageFunc*)PhObjFunc(self->super, "image");
  assert(func);
  image = func(self->super, m);
  /* clear to black */
  memset(image->data[0], 0, image->width*image->height*image->channels);

  /* are these bogus peaks? */
  if(peaks[0] < 0) return image;

  /* determine the maximum peak magnitude (dynamic scaling) */
  max_value = 0.0;
  for(i=0;i<mdata->num_peaks;i++) {
    double *peak = peaks + 2 + i*3;
    if(peak[2] > max_value) max_value = peak[2];
  }
  /* draw the peaks as points whose brightness is peak magnitude */
  for(i=0;i<mdata->num_peaks;i++) {
    double *peak = peaks + 2 + i*3;
    int gray = RealToGray(peak[2]/max_value);
    if(gray == 0) continue;
    Ph_ImagePutPixel(image, (int)peak[0], (int)peak[1], gray, gray, gray);
    Ph_ImagePutPixel(image, (int)peak[0]-1, (int)peak[1], gray, gray, gray);
    Ph_ImagePutPixel(image, (int)peak[0]+1, (int)peak[1], gray, gray, gray);
    Ph_ImagePutPixel(image, (int)peak[0], (int)peak[1]-1, gray, gray, gray);
    Ph_ImagePutPixel(image, (int)peak[0], (int)peak[1]+1, gray, gray, gray);
    Ph_ImagePutPixel(image, peaks[0] - (int)peak[0], 
		     peaks[1] - (int)peak[1], gray, gray, gray);
    Ph_ImagePutPixel(image, peaks[0] - (int)peak[0] - 1, 
		     peaks[1] - (int)peak[1], gray, gray, gray);
    Ph_ImagePutPixel(image, peaks[0] - (int)peak[0] + 1, 
		     peaks[1] - (int)peak[1], gray, gray, gray);
    Ph_ImagePutPixel(image, peaks[0] - (int)peak[0], 
		     peaks[1] - (int)peak[1] - 1, gray, gray, gray);
    Ph_ImagePutPixel(image, peaks[0] - (int)peak[0], 
		     peaks[1] - (int)peak[1] + 1, gray, gray, gray);
  }

  return image;
}

static void watchProc(Ph_Object self, char *field, void *userData)
{
  struct ViewPeaksData *data = (struct ViewPeaksData *)self->data;
  Ph_Object obj;

  obj = PhLookupObject(self->phandle, data->peaks);
  if(!obj) {
    fprintf(stderr, "No such object: %s\n", data->peaks);
    return;
  }
}

void ViewPeaks_Con(Ph_Object self)
{
  struct ViewPeaksData *data = (struct ViewPeaksData *)self->data;
  Ph_ObjWatch(self, "peaks", watchProc, NULL);
}
