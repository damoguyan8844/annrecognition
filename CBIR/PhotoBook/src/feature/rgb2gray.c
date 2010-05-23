/******************************************************************************
Copyright 1994 by the Massachusetts Institute of Technology.  All
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

void main(int argc, char *argv[])
{
  Matrix rgb_image, gray_image;
  int width, height, x,y;
  double *p, *q;

  if(argc != 3) {
    printf("Usage: %s <width> <height> < infile > outfile\n", argv[0]);
    printf("To convert a 3-channel RGB image to a single channel NTSC gray image\n");
    exit(0);
  }

  width = atoi(argv[1]);
  height = atoi(argv[2]);

  rgb_image = MatrixCreate(height,width*3);
  MatrixReadImage(rgb_image, stdin);
  gray_image = MatrixCreate(height,width);

  p = rgb_image->data[0];
  q = gray_image->data[0];
  for(y=0;y<height;y++) {
    for(x=0;x<width;x++) {
      *q++ = 0.30 * *p + 0.59 * *(p+1) + 0.11 * *(p+2);
      p += 3;
    }
  }

  MatrixWriteImage(gray_image, stdout);
}
