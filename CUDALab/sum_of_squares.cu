// First CUDA program


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DATA_SIZE	1048576
#define BLOCK_NUM	32
#define THREAD_NUM	256


extern "C" __global__  void sumOfSquares(int *num, int* result, clock_t* time);
 
__global__  void sumOfSquares(int *num, int* result, clock_t* time)
{
	extern __shared__ int shared[];
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
    int i;

	if(tid == 0) time[bid] = clock();
	shared[tid] = 0;

	for(i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
       shared[tid] += num[i] * num[i];
    }

    __syncthreads();

	if(tid < 128) { shared[tid] += shared[tid + 128]; } __syncthreads();
	if(tid < 64) { shared[tid] += shared[tid + 64]; } __syncthreads();
	if(tid < 32) { shared[tid] += shared[tid + 32]; } __syncthreads();
	if(tid < 16) { shared[tid] += shared[tid + 16]; } __syncthreads();
	if(tid < 8) { shared[tid] += shared[tid + 8]; } __syncthreads();
	if(tid < 4) { shared[tid] += shared[tid + 4]; } __syncthreads();
	if(tid < 2) { shared[tid] += shared[tid + 2]; } __syncthreads();
	if(tid < 1) { shared[tid] += shared[tid + 1]; } __syncthreads();

	if(tid == 0) {
		result[bid] = shared[0];
		time[bid + BLOCK_NUM] = clock();
	}
}



