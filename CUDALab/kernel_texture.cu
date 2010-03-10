#define BLOCK_DIM 512
texture<unsigned char, 1, cudaReadModeElementType> rT1;
texture<unsigned char, 1, cudaReadModeElementType> rT2;

extern "C" void Blend_GPU_Texture( unsigned char* aImg1, unsigned char* aImg2, unsigned char* aRS, int width, int height );
extern "C" void Blend_GPU_Texture_kernel_only( unsigned char* aRS, int size );
extern "C" void BindTexture( unsigned char* aImg1, unsigned char* aImg2 );
extern "C" void UnbindTexture();

__global__ void Blending_Texture_Kernel( unsigned char* aRS, int size )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if( index < size )
        aRS[index]  = 0.5 * tex1Dfetch( rT1, index ) + 0.5 * tex1Dfetch( rT2, index );
}

void Blend_GPU_Texture( unsigned char* aImg1, unsigned char* aImg2, unsigned char* aRS, int width, int height )
{
    int size = height * width;
    int data_size = size * sizeof( unsigned char );

    // part1, allocate data on device
    unsigned char	*dev_A,	*dev_B,	*dev_C;
    cudaMalloc( (void**)&dev_A, data_size );
    cudaMalloc( (void**)&dev_B, data_size );
    cudaMalloc( (void**)&dev_C, data_size );

    // part2, copy memory to device
    cudaMemcpy( dev_A, aImg1, data_size, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_B, aImg2, data_size, cudaMemcpyHostToDevice );

    // part2a, bind texture
    cudaBindTexture(0, rT1, dev_A );
    cudaBindTexture(0, rT2, dev_B );

    // part3, run kernel
    Blending_Texture_Kernel<<< ceil( (float)size / BLOCK_DIM ), BLOCK_DIM >>>( dev_C, size );

    // part4, copy data from device
    cudaMemcpy( aRS, dev_C, data_size, cudaMemcpyDeviceToHost );

    // part5, release data
    cudaUnbindTexture(rT1);
    cudaUnbindTexture(rT2);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}

void BindTexture( unsigned char* aImg1, unsigned char* aImg2 )
{
    cudaBindTexture(0, rT1, aImg1 );
    cudaBindTexture(0, rT2, aImg2 );
}

void UnbindTexture()
{
    cudaUnbindTexture(rT1);
    cudaUnbindTexture(rT2);
}

void Blend_GPU_Texture_kernel_only( unsigned char* aRS, int size )
{
	Blending_Texture_Kernel<<< ceil( (float)size / BLOCK_DIM ), BLOCK_DIM >>>( aRS, size );
}
