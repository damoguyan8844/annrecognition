

#include <string>
#include <fstream>
#include "GocrDll.h"


using namespace std;

int main(int argc, char **argv) {
	char veryBig[1024];
	const char* pFile="ocr-a.pnm";

	int out=GocrDll_Eng(pFile,veryBig,1023);
	
	fprintf(stdout,"%s",veryBig);

	return 0;
}