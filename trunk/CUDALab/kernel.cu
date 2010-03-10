#define BLOCK_DIM 512

extern "C" void Blend_GPU( unsigned char* aImg1, unsigned char* aImg2, unsigned char* aImg3, int width, int height );
extern "C" void Blend_GPU_kernel_only( unsigned char* aImg1, unsigned char* aImg2, unsigned char* aRS, int size );

__global__ void Blending_Kernel( unsigned char* aR1, unsigned char* aR2, unsigned char* aRS, int size )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if( index < size )
        aRS[index]  = 0.5 * aR1[index] + 0.5 * aR2[index];
}

void Blend_GPU( unsigned char* aImg1, unsigned char* aImg2, unsigned char* aRS, int width, int height )
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

    // part3, run kernel
    Blending_Kernel<<< ceil( (float)size / BLOCK_DIM ), BLOCK_DIM >>>( dev_A, dev_B, dev_C, size );

    // part4, copy data from device
    cudaMemcpy( aRS, dev_C, data_size, cudaMemcpyDeviceToHost );

    // part5, release data
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}

void Blend_GPU_kernel_only( unsigned char* aImg1, unsigned char* aImg2, unsigned char* aRS, int size )
{
	Blending_Kernel<<< ceil( (float)size / BLOCK_DIM ), BLOCK_DIM >>>( aImg1, aImg2, aRS, size );
}
