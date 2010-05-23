/* Definition of Ph_Image */

#if !(defined(__GNUC__) && defined(__alpha__))
typedef unsigned char uchar;
#endif

typedef uchar RGBFunc(uchar red, uchar green, uchar blue);
#define ColorToGray(r,g,b) (uchar)(0.30*(r)+0.59*(g)+0.11*(b)+0.5)
#define RealToGray(r) ((uchar)((r)*255+0.5))

typedef struct Ph_ImageStruct {
  int height, width, channels;
  uchar **data;
} *Ph_Image;

/* image.c *******************************************************************/

Ph_Image Ph_ImageCreate(int height, int width, int channels);
void Ph_ImageFree(Ph_Image image);

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
