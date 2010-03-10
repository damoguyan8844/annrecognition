#ifndef lab_cuda_functions_hpp 
#define lab_cuda_functions_hpp 

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

extern "C" void cuda_init();
extern "C" void cuda_hello( char * host_result,clock_t * time_used);

extern "C" void cuda_add_vector( float* a, float* b, float *c, int size );

extern "C" void Blend_GPU( unsigned char* aImg1, unsigned char* aImg2, unsigned char* aImg3, int width, int height );
extern "C" void Blend_GPU_2( unsigned char* aImg1, unsigned char* aImg2, unsigned char* aImg3, int width, int height, int channel );
extern "C" void Blend_GPU_kernel_only( unsigned char* aImg1, unsigned char* aImg2, unsigned char* aRS, int size );
extern "C" void Blend_GPU_Texture( unsigned char* aImg1, unsigned char* aImg2, unsigned char* aRS, int width, int height );
extern "C" void Blend_GPU_Texture_kernel_only( unsigned char* aRS, int size );
extern "C" void BindTexture( unsigned char* aImg1, unsigned char* aImg2 );
extern "C" void UnbindTexture();
extern "C" void Transpose_GPU( unsigned char* sImg, unsigned char *tImg, int w, int h );

#endif
