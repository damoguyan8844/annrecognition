#include "photobook.h"

/* Prototypes ****************************************************************/

Ph_Image Ph_ImageCreate(int height, int width, int channels);
void Ph_ImageFree(Ph_Image image);

Ph_Image Ph_ImageCopy(Ph_Image image);
void Ph_ImagePutPixel(Ph_Image image, int y, int x, 
		      uchar red, uchar green, uchar blue);
void Ph_ImagePutRow(Ph_Image image, int y, int x, 
		    int width, 
		    uchar red, uchar green, uchar blue);
void Ph_ImagePutBlock(Ph_Image image, int y, int x, 
		      int height, int width, 
		      uchar red, uchar green, uchar blue);
void Ph_ImageRGBFunc(Ph_Image image, RGBFunc *func);
uchar RGB_Gray(uchar red, uchar green, uchar blue);
uchar RGB_Ohta1(uchar red, uchar green, uchar blue);
uchar RGB_Ohta2(uchar red, uchar green, uchar blue);
uchar RGB_Ohta3(uchar red, uchar green, uchar blue);
Ph_Image Ph_ImageZoom(Ph_Image image, int zfact);

/* Functions *****************************************************************/

Ph_Image Ph_ImageCreate(int height, int width, int channels)
{
  int i;
  Ph_Image image = Allocate(1, struct Ph_ImageStruct);
  image->height = height;
  image->width = width;
  image->channels = channels;
  image->data = Allocate(height, uchar*);
  image->data[0] = Allocate(height*width*channels, uchar);
  for(i=1;i<height;i++) {
    image->data[i] = &image->data[0][i*width*channels];
  }
  return image;
}

void Ph_ImageFree(Ph_Image image)
{
  free(image->data[0]);
  free(image->data);
  free(image);
}

Ph_Image Ph_ImageCopy(Ph_Image image)
{
  Ph_Image result;

  result = Ph_ImageCreate(image->height, image->width, image->channels);
  memcpy(result->data[0], image->data[0], 
	 image->height * image->width * image->channels);
  return result;
}

void Ph_ImagePutPixel(Ph_Image image, int y, int x, 
		      uchar red, uchar green, uchar blue)
{
  if(image->channels == 3) {
    image->data[y][3*x] = red;
    image->data[y][3*x+1] = green;
    image->data[y][3*x+2] = blue;
  }
  else {
    image->data[y][x] = ColorToGray(red, green, blue);
  }
}

void Ph_ImagePutRow(Ph_Image image, int y, int x, 
		    int width, 
		    uchar red, uchar green, uchar blue)
{
  int i;
  uchar *p;

  if(image->channels == 3) {
    p = &image->data[y][3*x];
    for(i=0;i<width;i++) {
      *p++ = red;
      *p++ = green;
      *p++ = blue;
    }
  }
  else {
    uchar gray = ColorToGray(red, green, blue);
    p = &image->data[y][x];
    for(i=0;i<width;i++) {
      *p++ = gray;
    }
  }
}

void Ph_ImagePutBlock(Ph_Image image, int y, int x, 
		      int height, int width, 
		      uchar red, uchar green, uchar blue)
{
  int i;
  for(i=0;i<height;i++) {
    Ph_ImagePutRow(image, i+y, x, width, red, green, blue);
  }
}

uchar RGB_Gray(uchar red, uchar green, uchar blue)
{
  return ColorToGray(red, green, blue);
}

uchar RGB_Ohta1(uchar red, uchar green, uchar blue)
{
  return (uchar)(((int)red + green + blue) / 3);
}

uchar RGB_Ohta2(uchar red, uchar green, uchar blue)
{
  return (uchar)(((int)red - blue + 255)/2);
}

uchar RGB_Ohta3(uchar red, uchar green, uchar blue)
{
  return (uchar)(((int)red - 2*green + blue + 510)/4);
}

static uchar RedValue(Ph_Image image, int y, int x)
{
  if(image->channels == 3) {
    return image->data[y][3*x];
  }
  else {
    return image->data[y][x];
  }
}

static uchar GreenValue(Ph_Image image, int y, int x)
{
  if(image->channels == 3) {
    return image->data[y][3*x+1];
  }
  else {
    return image->data[y][x];
  }
}

static uchar BlueValue(Ph_Image image, int y, int x)
{
  if(image->channels == 3) {
    return image->data[y][3*x+2];
  }
  else {
    return image->data[y][x];
  }
}

void Ph_ImagePutGrayPixel(Ph_Image image, int y, int x, uchar gray)
{
  if((y < 0) || (y >= image->height) ||
     (x < 0) || (x >= image->width)) {
    fprintf(stderr, "Ph_ImagePutGrayPixel: bad coordinates (%d, %d)\n", x, y);
    return;
  }
  if(image->channels == 3) {
    image->data[y][3*x] = gray;
    image->data[y][3*x+1] = gray;
    image->data[y][3*x+2] = gray;
  }
  else {
    image->data[y][x] = gray;
  }
}

void Ph_ImageRGBFunc(Ph_Image image, RGBFunc *func)
{
  int y,x;

  for(y=0;y<image->height;y++) {
    for(x=0;x<image->width;x++) {
      Ph_ImagePutGrayPixel(image, y, x, 
		    func(RedValue(image, y, x),
			 GreenValue(image, y, x),
			 BlueValue(image, y, x)));
    }
  }
}

Ph_Image Ph_ImageZoom(Ph_Image image, int zfact)
{
  int i,j,k,l,m;
  int p;
  Ph_Image zimage;
  
  if(zfact == 0) zfact=1;
  if((zfact == 1) || (zfact == -1)) return Ph_ImageCopy(image);
  
  if(zfact > 1) {
    zimage=Ph_ImageCreate((image->height)*zfact,(image->width)*zfact,
			  image->channels);
    /* Puts enlarged copy of image->data into zimage->data */
    for(i=0;i<image->height;i++) {
      for(j=0;j<image->width;j++) {
	for(m=0;m<zimage->channels;m++) {
	  p=image->data[i][image->channels*j + m];
	  for(k=0;k<zfact;k++) {
	    for(l=0;l<zfact;l++) {
	      zimage->data[zfact*i+k][image->channels*(zfact*j+l)+m]=p; 
	    }
	  }
	}
      }
    }
  }
  else { /* zfact is < -1 */
    zfact = -zfact;
    zimage=Ph_ImageCreate((image->height)/zfact,(image->width)/zfact,
			  image->channels);
    /* Puts shrunken copy of image->data  onto zimage->data */
    for(i=0;i<zimage->height;i++) {
      for(j=0;j<zimage->width;j++) {
	for(m=0;m<zimage->channels;m++) {
	  p=0;
	  for(k=0;k<zfact;k++) {
	    for(l=0;l<zfact;l++) {
	      p += image->data[zfact*i+k][(zimage->channels)*(zfact*j+l)+m];
	    }
	  }
	  zimage->data[i][zimage->channels*j+m]=p/(zfact*zfact);
	}
      }
    }
  }
  return zimage;
}
