#include "photobook.h"
#include "obj_data.h"

/* Functions *****************************************************************/

/* Function corresponding to the "image" method in ViewFuncs.
 * Returns an uninitialized image with the height, width, and channels
 * specified by self->data.
 */
Ph_Image View_Image(Ph_Object self, Ph_Member m)
{
  struct ViewData *data = (struct ViewData *)self->data;
  return Ph_ImageCreate(data->height, data->width, data->channels);
}

/* Function corresponding to the "constructor" method in ViewFuncs.
 * Initializes the default image parameters.
 */
void View_Con(Ph_Object self)
{
  struct ViewData *data = (struct ViewData *)self->data;
  data->height = 128;
  data->width = 128;
  data->channels = 1;
}
