#include <stdio.h>
#include <stdlib.h>

#include "main_select.h"

#ifdef Main_VectorAdd

void add_vector_cpu( float* a, float* b, float *c, int size )
{
	for( int i = 0; i < size; ++ i )
		c[i] = a[i] + b[i];
}



void main( int argc, char** argv) 
{
	if(!InitCUDA()) {
		printf( "Init CUDA failure ! \n");
		system("pause");
		return ;
	}

	// initial data
	int	data_size = 50;
	float	*dataA = new float[data_size],
			*dataB = new float[data_size],
			*dataC = new float[data_size],
			*dataD = new float[data_size];

	for( int i = 0; i < data_size; ++ i )
	{
		dataA[i] = i;
		dataB[i] = -1 * i;
	}

	// run CPU program
	add_vector_cpu( dataA, dataB, dataC, data_size );

	// run GPU program
	cuda_add_vector( dataA, dataB, dataD, data_size );

	// compare the result
	for( int i = 0; i < data_size; ++ i )
	{
	//	if( dataC[i] != dataD[i] )
			printf( "Error!! (%f & %f)\n", dataC[i], dataD[i] );
	}

	system("pause");
}

#endif