/* Module: Pixmap Cache and related functions */

#include <math.h>
#include "photobook.h"
#include "ui.h"

/* Globals *******************************************************************/

typedef struct CacheRec {
  char *name;
  int width, height;
  Pixmap pixmap;
} *CacheNode;

List pixCache; /* List of CacheNode */
int cacheSize = 100;

/* alpha: if "str" is called "s" then this breaks! */
#define CacheKey(str,m,v) sprintf(str, "%s/%s", Ph_MemName(m), v);

/* Prototypes ****************************************************************/
void PixCacheCreate(void);
void PixCacheFree(void);
Pixmap GetMemPixmap(Ph_Member member);
void UncacheMemPixmap(Ph_Member member, char *view);
void PixCacheSize(int size);

/* Private */
static Pixmap Image2Pixmap(Ph_Image image, int pad);
static void XImageFill(XImage *ximage, uchar *dptr, int img_channels);
static Pixmap CopyPixmap(Pixmap pixmap, int width, int height);
static CacheNode InstallPixmap(char *name, int width, int height, Pixmap pixmap);

/* Functions *****************************************************************/

void PixCacheSize(int bytes)
{
  cacheSize = bytes/appData->pix_width/appData->pix_height/appData->im_channels;
  if(debug) fprintf(stderr, "PixCache size = %d\n", cacheSize);
}

static void DestroyCacheNode(CacheNode node)
{
  free(node->name);
  XFreePixmap(appData->display, node->pixmap);
  free(node);
}

/* Initialize the image cache */
void PixCacheCreate(void)
{
  pixCache = ListCreate((FreeFunc*)DestroyCacheNode);
}

/* Destroy the contents of the image cache */
void PixCacheFree(void)
{
  ListFree(pixCache);
}

static int CompareCacheNode(CacheNode a, String b)
{
  return strcmp(a->name, b);
}

void UncacheMemPixmap(Ph_Member member, char *view)
{
  char cache_key[1000];

  CacheKey(cache_key, member, view);
  ListRemoveValue(pixCache, cache_key, (CmpFunc*)CompareCacheNode);
}

/* Get a pixmap from the cache.
 * Returns a new pixmap (caller must free it).
 */
Pixmap GetMemPixmap(Ph_Member member)
{
  char cache_key[1000];
  Pixmap pixmap;
  int cache_it;
  ListNode p;
  Ph_Image image;
  CacheNode node;

  CacheKey(cache_key, member, Ph_ObjName(Ph_GetView(phandle)));

  /* Check if the key is in the cache */
  ListIterate(p, pixCache) {
    if(!strcmp((node=(CacheNode)p->data)->name, cache_key)) {
      break;
    }
  }

  if(p) {
    pixmap = node->pixmap;
  }
  else {
    image = Ph_MemImage(member);
    if(image) {
      pixmap = Image2Pixmap(image, appData->y_pad);
      node = InstallPixmap(cache_key, 
			   image->width, image->height+appData->y_pad, pixmap);
      Ph_ImageFree(image);
    }
    else {
      return CopyPixmap(appData->no_image_pixmap, 
			appData->pix_width, appData->pix_height);
    }
  }
  return CopyPixmap(pixmap, node->width, node->height);
}

/* Private *******************************************************************/

static void XImageLocalByteOrder(XImage *ximage)
{
  int z=1;
  unsigned char *c=(unsigned char *)&z;
  ximage->byte_order = (c[0] == 0) ? MSBFirst : LSBFirst;
}

static Pixmap Image2Pixmap(Ph_Image image, int pad)
{
  XImage *ximage;
  Pixmap pixmap;
  uchar *ptr;
  int width, height, total_size, i;

  if(!image) return (Pixmap)0;

  if(appData->gamma_table) {
    /* perform gamma correction on image data */
    ptr = (uchar*)image->data[0];
    total_size = image->height*image->width*image->channels;
    for(i=0;i < total_size;i++) {
      *ptr = appData->gamma_table[*ptr];
      ptr++;
    }
  }

  /* copy the image into an ximage */
  width = image->width;
  height = image->height;
  ximage = XCreateImage(appData->display, 
			appData->visual,
			appData->depth,
			ZPixmap, 
			0, 
			NULL,
			image->width,
			image->height,
			32,
			0);
  XImageLocalByteOrder(ximage);
  total_size = ximage->bytes_per_line * image->height;
  ximage->data = Allocate(total_size, char);
  if(!ximage->data) {
    fprintf(stderr, "Could not allocate %d by %d ximage\n",
	    ximage->width, ximage->height);
    XtError("Image2Pixmap: XCreateImage failed\n");
  }
  XImageFill(ximage, image->data[0], image->channels);

  if(appData->color_table) {
    /* map each byte through the color_table */
    ptr = (uchar*)ximage->data;
    for(i=0;i < total_size;i++) {
      *ptr = appData->color_table[*ptr];
      ptr++;
    }
  }

  /* Allocate pixmap */
  pixmap = XCreatePixmap(appData->display,
			 appData->root,
			 width,
			 height+pad,
			 appData->depth);
  if(!pixmap) {
    XtError("Image2Pixmap: XCreatePixmap failed\n");
  }

  /* Clear the pixmap */
  XSetForeground(appData->display, appData->gc, 
		 appData->color[COLOR_PHOTOBG].pixel);
  XFillRectangle(appData->display,
		 pixmap,
		 appData->gc,0,0,
		 width,
		 height+pad); 
  
  /* Copy the image to the pixmap */
  XPutImage(appData->display, 
	    pixmap,
	    appData->gc,
	    ximage, 
	    0, 0, 
	    0, 0,
	    image->width,
	    image->height);

  DestroyXImage(ximage);
  return pixmap;
}

/* Fill an XImage with multi-byte data */
/* If the pixel size is greater than 'channels', 
 *   the byte values will be replicated across multiple bytes.
 * If the pixel size is less than 'channels',
 *   the low order bits (or bytes) of the data will be used.
 * Should work regardless of byte ordering (MSBfirst or LSBfirst).
 */
#define LSB3to4_1(r,g,b) \
  ((r) + ((g) << 8) + ((b) << 16))
#define LSB3to4_2(r,g,b) \
  (LSB3to4_1(r,g,b) << 8)
#define MSB3to4_1(r,g,b) \
  (((r) << 16) + ((g) << 8) + (b))
#define MSB3to4_2(r,g,b) \
  (MSB3to4_1(r,g,b) << 8)
#define LSB3to2(r,g,b) \
  ((((r) >> 3) & 0x1f) + (((g) << 3) & 0x7e0) + (((b) << 8) & 0xf800))
#define MSB3to2(r,g,b) \
  ((((r) << 8) & 0xf800) + (((g) << 3) & 0x7e0) + (((b) >> 3) & 0x1f))

static void XImageFill(XImage *ximage, uchar *dptr, int img_channels)
{
  int xi_channels, i, j, image_size;
  uchar *xdptr;
  int pad; /* extra bytes at end of each ximage row */

  xdptr = (uchar *)ximage->data;
  xi_channels = ximage->bytes_per_line / ximage->width;
  pad = ximage->bytes_per_line - ximage->width*xi_channels;

  if(debug)
    printf("%d -> %d;pad = %d\n", img_channels, xi_channels, pad);

  image_size = ximage->width * ximage->height;
  if(xi_channels > 0) {
    if(xi_channels == img_channels) {
      if(pad) {
	for(i=0;i<ximage->height;i++) {
	  memcpy(xdptr, dptr, ximage->width*img_channels);
	  xdptr += ximage->width*img_channels + pad;
	  dptr += ximage->width*img_channels;
	}
      }
      else 
	memcpy(xdptr, dptr, image_size*img_channels);
    }
    else if(xi_channels == 4) {
      if(img_channels == 3) {
	if(ximage->red_mask == 0xff0000) {
	  for(i=0;i<ximage->height;i++) {
	    for(j=0;j<ximage->width;j++) {
	      XPutPixel(ximage, j, i, MSB3to4_1(*dptr, *(dptr+1), *(dptr+2)));
	      dptr+=3;
	    }
	  }
	}
	else {
	  for(i=0;i<ximage->height;i++) {
	    for(j=0;j<ximage->width;j++) {
	      XPutPixel(ximage, j, i, LSB3to4_1(*dptr, *(dptr+1), *(dptr+2)));
	      dptr+=3;
	    }
	  }
	}
/*
	if(ximage->byte_order == LSBFirst) {
	  for(i=0;i<image_size;i++,xdptr+=4,dptr+=3) {
	    *(int*)xdptr) = LSB3to4_1(*dptr, *(dptr+1), *(dptr+2));
	  }
	}
	else {
	  for(i=0;i<image_size;i++,xdptr+=4,dptr+=3) {
	    *(int*)xdptr) = MSB3to4_1(*dptr, *(dptr+1), *(dptr+2));
	  }
	}
*/
      }
      else if(img_channels == 1) {
	/* put same byte in every channel */
	for(i=0;i<image_size;i++,xdptr+=4,dptr++) {
	  *(int*)xdptr = MSB3to4_2(*dptr, *dptr, *dptr) + *dptr;
	}
      }
    }
    else if(xi_channels == 2) {
      unsigned short *uptr = (unsigned short *)xdptr;
      if(img_channels == 3) {
	if(ximage->red_mask == 0xf800) {
	  for(i=0;i<ximage->height;i++) {
	    for(j=0;j<ximage->width;j++) {
	      *uptr++ = MSB3to2(*dptr, *(dptr+1), *(dptr+2));
	      dptr+=3;
	    }
	    uptr += pad/2;
	  }
	}
	else {
	  for(i=0;i<ximage->height;i++) {
	    for(j=0;j<ximage->width;j++) {
	      *uptr++ = LSB3to2(*dptr, *(dptr+1), *(dptr+2));
	      dptr+=3;
	    }
	    uptr += pad/2;
	  }
	}
      }
      else {
	printf("Unexpected img_channels (%d) for xi_channels = %d\n", 
	       img_channels, xi_channels);
      }
    }
    else if(xi_channels == 1) {
      if(img_channels == 3) {
	/* Compute NTSC gray = 0.30 red, 0.59 green, 0.11 blue */
	for(i=0;i<ximage->height;i++) {
	  for(j=0;j<ximage->width;j++) {
	    *xdptr++ = (uchar)
	      (0.30 * *dptr + 0.59 * *(dptr+1) + 0.11 * *(dptr+2));
	    dptr += 3;
	  }
	  xdptr += pad;
	}
      }
      else {
	printf("Unexpected img_channels (%d) for xi_channels = %d\n", 
	       img_channels, xi_channels);
      }
    }
    else {
      printf("Unexpected xi_channels: %d\n", xi_channels);
    }
  }
  else {
    /* Assume two pixels per byte */
    for(i=0;i<image_size;i+=2,dptr+=2*img_channels) {
      *xdptr++ = (*dptr << 4) | *(dptr+img_channels);
    }
  }
} 

/* Allocate a copy of a pixmap */
/* Returns a new pixmap */
static Pixmap CopyPixmap(Pixmap pixmap, int width, int height)
{
  Pixmap new;

  new = XCreatePixmap(appData->display, appData->root,
		      width, height,
		      appData->depth);
  XCopyArea(appData->display, pixmap, new, appData->gc,
	    0,0, width, height, 0,0);
  return new;
}

/* Install a pixmap into the cache */
static CacheNode InstallPixmap(char *name, 
			       int width, int height, Pixmap pixmap)
{
  CacheNode new;

  /* Don't allow more than cacheSize nodes */
  while(ListSize(pixCache) > cacheSize) {
    ListRemoveRear(pixCache, NULL);
  }

  /* Allocate a new node */
  do {
    new = Allocate(1,struct CacheRec);
    if(!new) {
      /* If the cache is empty, give up */
      if(ListEmpty(pixCache)) {
	fprintf(stderr, "Cannot cache '%s'\n", name);
	fflush(stderr);
      }
      /* Free a cache node to make room */
      ListRemoveRear(pixCache, NULL);
    }
  } while(!new);

  /* Put onto the front of the cache */
  new->name = strdup(name);
  new->width = width;
  new->height = height;
  new->pixmap = pixmap;
  ListAddFront(pixCache, new);
  return new;
}

