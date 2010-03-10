
extern "C" void cuda_hello( char * host_result,clock_t * time_used);

__global__ static void HelloCUDA(char* result, int num, clock_t* time)
{
	int i = 0;
	char p_HelloCUDA[] = "Hello CUDA!";
	clock_t start = clock();
	for(i = 0; i < num; i++) {
		result[i] = p_HelloCUDA[i];
	}
	*time = clock() - start;
}

void cuda_hello( char* host_result,clock_t * time_used)
{
	char		*device_result	= 0;
	clock_t		*time			= 0;	

	cudaMalloc((void**) &device_result, sizeof(char) * 12);
	cudaMalloc((void**) &time, sizeof(clock_t));

	HelloCUDA<<<1, 1, 0>>>(device_result, 12 , time);

	cudaMemcpy(host_result, device_result, sizeof(char) * 12, cudaMemcpyDeviceToHost);
	cudaMemcpy(time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
	cudaFree(device_result);
	cudaFree(time);

}