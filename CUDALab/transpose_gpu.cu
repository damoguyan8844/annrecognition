#define BLOCK_DIM 16

texture<unsigned char, 2, cudaReadModeElementType> rT;

extern "C" void Transpose_GPU( unsigned char* sImg, unsigned char *tImg, int w, int h );


__global__ void Transpose_Texture( unsigned char* aRS, int w, int h )
{
	int	idxX = blockIdx.x * blockDim.x + threadIdx.x,
		idxY = blockIdx.y * blockDim.y + threadIdx.y;
	if( idxX < w && idxY < h )
		aRS[ idxX * h + idxY ] = tex2D( rT, idxX, idxY );
}

void Transpose_GPU( unsigned char* sImg, unsigned char *tImg, int w, int h )
{
	// compute the size of data
	int	data_size = sizeof(unsigned char) * w * h;

	// part1a. prepare the result data
	unsigned char *dImg;
	cudaMalloc( (void**)&dImg, data_size );

	// part1b. prepare the source data
	cudaChannelFormatDesc chDesc = cudaCreateChannelDesc<unsigned char>();
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &chDesc, w, h);
	cudaMemcpyToArray( cuArray, 0, 0, sImg, data_size, cudaMemcpyHostToDevice );
	cudaBindTextureToArray( rT, cuArray );

	// part2. run kernel
	dim3	block( BLOCK_DIM, BLOCK_DIM ),
			grid( ceil( (float)w / BLOCK_DIM), ceil( (float)h / BLOCK_DIM) );
	Transpose_Texture<<< grid, block>>>( dImg, w, h );

	// part3. copy the data from device
	cudaMemcpy( tImg, dImg, data_size, cudaMemcpyDeviceToHost );

	// par4. release data
	cudaUnbindTexture( rT );
	cudaFreeArray( cuArray );
	cudaFree( dImg );
}
