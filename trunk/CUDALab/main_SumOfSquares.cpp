#include <stdio.h>
#include "main_select.h"

#ifdef Main_SumOfSquares

bool InitCUDA()
{
	int count;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;
	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}

	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}


void GenerateNumbers(int *number, int size)
{
	for(int i = 0; i < size; i++) {
		number[i] = rand() % 10;
	}
}


int main()
{
	if(!InitCUDA()) {
		return 0;
	}

	printf("CUDA initialized.\n");

	GenerateNumbers(data, DATA_SIZE);

	int* gpudata, *result;
	clock_t* time;
	cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
	cudaMalloc((void**) &result, sizeof(int) * BLOCK_NUM);
	cudaMalloc((void**) &time, sizeof(clock_t) * BLOCK_NUM * 2);
	cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

	sumOfSquares<<<BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int)>>>(gpudata, result, time);

	int sum[BLOCK_NUM];
	clock_t time_used[BLOCK_NUM * 2];
	cudaMemcpy(&sum, result, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_used, time, sizeof(clock_t) * BLOCK_NUM * 2, cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);
	cudaFree(time);

	int final_sum = 0;
	for(int i = 0; i < BLOCK_NUM; i++) {
		final_sum += sum[i];
	}

	clock_t min_start, max_end;
	min_start = time_used[0];
	max_end = time_used[BLOCK_NUM];
	for(int i = 1; i < BLOCK_NUM; i++) {
		if(min_start > time_used[i]) min_start = time_used[i];
		if(max_end < time_used[i + BLOCK_NUM]) max_end = time_used[i + BLOCK_NUM];
	}

	printf("sum: %d  time: %d\n", final_sum, max_end - min_start);

	final_sum = 0;
	for(int i = 0; i < DATA_SIZE; i++) {
		final_sum += data[i] * data[i];
	}
	printf("sum (CPU): %d\n", final_sum);

	return 0;
}

#endif