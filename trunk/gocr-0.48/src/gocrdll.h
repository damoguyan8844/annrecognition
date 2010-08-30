
// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the GOCRDLL_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// GOCRDLL_API functions as being imported from a DLL, wheras this DLL sees symbols
// defined with this macro as being exported.
#ifdef GOCRDLL_EXPORTS
#define GOCRDLL_API __declspec(dllexport)
#else
#define GOCRDLL_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C"
{
#endif
	

GOCRDLL_API int __cdecl GocrDll_Eng(const char * imageFile,char * output,int length);


#ifdef __cplusplus
}
#endif
