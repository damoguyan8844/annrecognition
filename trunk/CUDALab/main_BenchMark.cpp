#include <stdio.h>
#include "main_select.h"

#ifdef Main_BenchMark

void Blend_CPU( unsigned char* aImg1, unsigned char* aImg2, unsigned char* aRS, int width, int height )
{
	for( int i = 0; i < width * height; ++ i )
		aRS[i] = (unsigned char)( 0.5 * aImg1[i] + 0.5 * aImg2[i] );
}

bool CompareResult( unsigned char* aImg1, unsigned char* aImg2, int size )
{
	for( int i = 0; i < size; ++ i )
		if( aImg1[i] != aImg2[i] )
			return false;
	return true;
}

int main( int arc, char** argv )
{
	int numIterations = 300;

	// cutil timer
	unsigned int	timer;
	cutCreateTimer( &timer );

	// define test data
	int width   = 1600,
		height  = 1200;
	int size = height * width;
	int data_size = size * sizeof( unsigned char );

	// Setup test data
	unsigned char	*dev_A,	*dev_B,	*dev_C;
	unsigned char	*aImg1 = new unsigned char[ width*height ],
					*aImg2 = new unsigned char[ width*height ],
					*aRS1  = new unsigned char[ width*height ],
					*aRS2  = new unsigned char[ width*height ];
	for( int i = 0; i < width * height; ++ i )
	{
		aImg1[i] = 0;
		aImg2[i] = 200;
	}

	// CPU Test
	{
		cutStartTimer(timer);
		for (int i = 0; i < numIterations; ++i)
			Blend_CPU( aImg1, aImg2, aRS1, width, height );
		cutStopTimer(timer);
		printf( "CPU                         : %f\n", cutGetTimerValue(timer) / numIterations );
		cutResetTimer(timer);
	}

	// GPU test
	{
		cutStartTimer(timer);
		for (int i = 0; i < numIterations; ++i)
		{
			Blend_GPU( aImg1, aImg2, aRS2, width, height );
			cudaThreadSynchronize();
		}
		cutStopTimer(timer);
		printf( "GPU                         : %f\n", cutGetTimerValue(timer) / numIterations );
		cutResetTimer(timer);

		if( !CompareResult( aRS1, aRS2, size ) )
		{
			printf( "ERROR!!\n" );
			return 1;
		}
	}

	// GPU without memory transfer
	{
		cudaMalloc( (void**)&dev_A, data_size );
		cudaMalloc( (void**)&dev_B, data_size );
		cudaMalloc( (void**)&dev_C, data_size );
		cudaMemcpy( dev_A, aImg1, data_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_B, aImg2, data_size, cudaMemcpyHostToDevice );

		cutStartTimer(timer);
		for (int i = 0; i < numIterations; ++i)
		{
			Blend_GPU_kernel_only( dev_A, dev_B, dev_C, size );
			cudaThreadSynchronize();
		}
		cutStopTimer(timer);
		printf( "GPU without transfer        : %f\n", cutGetTimerValue(timer) / numIterations );
		cutResetTimer(timer);

		cudaMemcpy( aRS2, dev_C, data_size, cudaMemcpyDeviceToHost );
		cudaFree(dev_A);
		cudaFree(dev_B);
		cudaFree(dev_C);

		if( !CompareResult( aRS1, aRS2, size ) )
		{
			printf( "ERROR!!\n" );
			return 1;
		}
	}

	// Texture test
	{
		cutStartTimer(timer);
		for (int i = 0; i < numIterations; ++i)
		{
			Blend_GPU_Texture( aImg1, aImg2, aRS1, width, height );
			cudaThreadSynchronize();
		}
		cutStopTimer(timer);
		printf( "GPU Texture                 : %f\n", cutGetTimerValue(timer) / numIterations );
		cutResetTimer(timer);

		if( !CompareResult( aRS1, aRS2, size ) )
		{
			printf( "ERROR!!\n" );
			return 1;
		}
	}

	// Texture without memory transfer
	{
		cudaMalloc( (void**)&dev_A, data_size );
		cudaMalloc( (void**)&dev_B, data_size );
		cudaMalloc( (void**)&dev_C, data_size );
		cudaMemcpy( dev_A, aImg1, data_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_B, aImg2, data_size, cudaMemcpyHostToDevice );
		BindTexture( dev_A, dev_B );

		cutStartTimer(timer);
		for (int i = 0; i < numIterations; ++i)
		{
			Blend_GPU_Texture_kernel_only( dev_C, size );
			cudaThreadSynchronize();
		}
		cutStopTimer(timer);
		printf( "GPU Texture without transfer: %f\n", cutGetTimerValue(timer) / numIterations );
		cutResetTimer(timer);

		cudaMemcpy( aRS2, dev_C, data_size, cudaMemcpyDeviceToHost );
		UnbindTexture();
		cudaFree(dev_A);
		cudaFree(dev_B);
		cudaFree(dev_C);

		if( !CompareResult( aRS1, aRS2, size ) )
		{
			printf( "ERROR!!\n" );
			return 1;
		}
	}
	return 0;
}
#endif