/******************************************************************************
Copyright 1995 by the Massachusetts Institute of Technology.  All
rights reserved.

Developed by Thomas P. Minka and Rosalind W. Picard at the Media
Laboratory, MIT, Cambridge, Massachusetts, with support from BT, PLC,
Hewlett-Packard, and NEC.

This distribution is approved by Nicholas Negroponte, Director of
the Media Laboratory, MIT.

Permission to use, copy, or modify this software and its documentation
for educational and research purposes only and without fee is hereby
granted, provided that this copyright notice and the original authors'
names appear on all copies and supporting documentation.  If individual
files are separated from this distribution directory structure, this
copyright notice must be included.  For any other uses of this software,
in original or modified form, including but not limited to distribution
in whole or in part, specific prior permission must be obtained from
MIT.  These programs shall not be used, rewritten, or adapted as the
basis of a commercial software or hardware product without first
obtaining appropriate licenses from MIT.  MIT. makes no representations
about the suitability of this software for any purpose.  It is provided
"as is" without express or implied warranty.
******************************************************************************/

#include <stdio.h>
#include <matrix.h>

#define RGB_I1(r,g,b) (((r)+(g)+(b))/3)
#define RGB_I2(r,g,b) (((r)-(b)+256)/2)
#define RGB_I3(r,g,b) (((r)-2*(g)+(b)+512)/4)

int main(int argc, char *argv[])
{
  Matrix rgb_image, ohta_image;
  int width, height, x,y;
  double *p, *q;

  if(argc != 3) {
    fprintf(stderr, "Usage: %s <width> <height> < infile > outfile\n", argv[0]);
    fprintf(stderr, "To convert a 3-channel RGB image to a 3-channel Ohta I1,I2,I3 image\n");
    exit(1);
  }

  width = atoi(argv[1]);
  height = atoi(argv[2]);

  rgb_image = MatrixCreate(height,width*3);
  MatrixReadImage(rgb_image, stdin);
  ohta_image = MatrixCreate(height,width*3);

  p = rgb_image->data[0];
  q = ohta_image->data[0];
  for(y=0;y<height;y++) {
    for(x=0;x<width;x++) {
      *q++ = RGB_I1(*p, *(p+1), *(p+2));
      *q++ = RGB_I2(*p, *(p+1), *(p+2));
      *q++ = RGB_I3(*p, *(p+1), *(p+2));
      p += 3;
    }
  }

  MatrixWriteImage(ohta_image, stdout);
  exit(0);
}
