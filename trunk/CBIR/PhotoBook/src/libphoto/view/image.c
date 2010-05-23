#include "photobook.h"
#include "obj_data.h"
#include <assert.h>

/* Functions *****************************************************************/

/* returns 0 if an error occurred */
int skipPnmHeader(FILE * fileptr, Ph_Image image)
{
  int magic;
  int c;
  unsigned int garbage;
  int width, height, depth;

  /* - - GET INFO FROM PPM HEADER - - */
  if((fscanf(fileptr,"P%d", &magic) < 1) || (magic < 4) || (magic > 6)) {
    /* not a pnm file */
    goto error;
  }

  c = getc(fileptr);
  while (1) {
    /* - - KILL A COMMENT - - */
    if (c == '#') {
      while (1) {
	c = getc(fileptr);
	if (c == '\n' || c == EOF)
	  break;
      }
    }
    if (c==EOF)
      return(0);
    /* found a number? */
    if (c>='0' && c<='9') {
      /* rewind file pointer 1 char position */
      fseek(fileptr, -1, SEEK_CUR);
      break;
    }
    /* see if we are getting garbage (non-whitespace) */
    if (c!=' ' && c!='\t' && c!='\r' && c!='\n' && c!=',')
      garbage=1;
    c = getc(fileptr);
  }
  if(fscanf(fileptr, "%d %d %d", &width, &height, &depth) < 3) {
    fprintf(stderr, "Pnm header error\n");
    goto error;
  }
  if((width != image->width) || (height != image->height)) {
    fprintf(stderr, "Error: bad pnm dimensions (%d by %d)\n", width, height);
    goto error;
  }
  /* read the newline */
  getc(fileptr);
  
  /* - - fileptr should be at beginning of the data - - */
  return(1);

 error:
  rewind(fileptr);
  return 0;
}

/* Function corresponding to "image" method in ViewImageFuncs.
 * Takes the view object (self) and a member, returning a Ph_Image
 * read in from the directory specified by self->data->field.
 * Returns NULL if the file could not be read.
 */
Ph_Image ViewImage_Image(Ph_Object self, Ph_Member m)
{
  PhImageFunc *func;
  Ph_Image image;
  FILE *fp;
  char str[1000];
  struct ViewImageData *data = (struct ViewImageData *)self->data;
  unsigned image_size;
  int status;

  /* call the superclass to initialize the image */
  func = (PhImageFunc*)PhObjFunc(self->super, "image");
  assert(func);
  image = func(self->super, m);

  /* should be using transient data fields here */
  /* open the image file */
  sprintf(str, "%s/%s/%s/%s", 
	  self->phandle->data_dir, self->phandle->db_name, 
	  data->field, Ph_MemName(m));
  fp = fopen(str, "r");
  if(!fp) {
    fprintf(stderr, "Could not open image file `%s'\n", str);
    perror("");
    fflush(stderr);
    Ph_ImageFree(image);
    return NULL;
  }

  /* is this a pnm file? if so, skip the header info */
  status = skipPnmHeader(fp, image);
  if(debug) {
    if(status) fprintf(stderr, "reading pnm file\n");
    else fprintf(stderr, "reading RGB file\n");
  }

  /* read in the file */
  image_size = image->height * image->width;
/*
  if(appData->image_desc.non_interleaved) {
    plane = Allocate(image_size, uchar);
    for(i=0;i < channels;i++) {
      fread(plane, sizeof(uchar), image_size, fp);
      ptr = image[0]+i;
      for(j=0;j < image_size;j++) {
	*ptr = plane[j];
	ptr += channels;
      }
    }
    free(plane);
  }
  else {
    fread(image[0], channels, image_size, fp);
  }
*/
  fread(image->data[0], image->channels, image_size, fp);
  fclose(fp);

  return image;
}

/* Function corresponding to "constructor" method in ViewImageFuncs.
 * Initializes self->data->field (the image directory) to "image".
 */
void ViewImage_Con(Ph_Object self)
{
  struct ViewImageData *data = (struct ViewImageData *)self->data;
  /* set defaults */
  data->field = "image";
}
