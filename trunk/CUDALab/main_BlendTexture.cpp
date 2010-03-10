#include <stdio.h>
#include "main_select.h"

#ifdef Main_BlendTexture

void Blend_CPU( unsigned char* aImg1, unsigned char* aImg2,
			   unsigned char* aRS,
			   int width, int height, int channel )
{
	for( int i = 0; i < width * height * channel; ++ i )
		aRS[i] = (unsigned char)( 0.5 * aImg1[i] + 0.5 * aImg2[i] );
}

void main( int argc, char** argv )
{
	int width   = 1920,
		height  = 1200,
		channel = 3;

	// Setup test data
	unsigned char	*aImg1 = new unsigned char[ width*height*channel ],
		*aImg2 = new unsigned char[ width*height*channel ],
		*aRS1  = new unsigned char[ width*height*channel ],
		*aRS2  = new unsigned char[ width*height*channel ];
	for( int i = 0; i < width * height * channel; ++ i )
	{
		aImg1[i] = 0;
		aImg2[i] = 200;
	}

	// CPU code
	Blend_CPU( aImg1, aImg2, aRS1, width, height, channel );

	// GPU Code
	Blend_GPU_2( aImg1, aImg2, aRS2, width, height, channel );

	// check
	for( int i = 0; i < width * height * channel; ++ i )
		if( aRS1[i] != aRS2[i] )
		{
			printf( "Error!!!!\n" );
			break;
		}
}

#endif