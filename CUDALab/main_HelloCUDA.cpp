#include <stdio.h>
#include <stdlib.h>

#include "main_select.h"

#ifdef Main_HelloCUDA

int main(int argc, char** argv)
{

	if(!InitCUDA()) {
		return 0;
	}

	printf("CUDA initialized.\n");

	char	*host_result = new char[12];host_result[0]=0;
	clock_t	time_used	 = 0;

	cuda_hello(host_result,&time_used);

	printf("%s,%d\n", host_result, time_used);

	system("pause");
	return 0;
}

#endif