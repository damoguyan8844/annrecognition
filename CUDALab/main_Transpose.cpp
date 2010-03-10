#include <stdio.h>
#include "main_select.h"

#ifdef Main_Transpose

void Transpose_CPU( unsigned char* sImg, unsigned char *tImg, int w, int h )
{
	int x, y, idx1, idx2;
	for( y = 0; y < h; ++ y )
		for( x = 0; x < w; ++ x )
		{
			idx1 = y * w + x;
			idx2 = x * h + y;
			tImg[idx2] = sImg[idx1];
		}
}

void main( int argc, char** argv )
{
	int w	= 1920,
		h	= 1200;

	// Setup test data
	unsigned char	*aSrc = new unsigned char[ w * h ],
		*aRS1 = new unsigned char[ w * h ],
		*aRS2 = new unsigned char[ w * h ];
	for( int i = 0; i < w * h ; ++ i )
		aSrc[i] = i % 256;

	// CPU code
	Transpose_CPU( aSrc, aRS1, w, h );

	// GPU Code
	Transpose_GPU( aSrc, aRS2, w, h );

	// check
	for( int i = 0; i < w * h; ++ i )
		if( aRS1[i] != aRS2[i] )
		{
			printf( "Error!!!!\n" );
			break;
		}
	system("pause");
}

#endif