// ANNRecognition.cpp : Defines the entry point for the DLL application.
//

#include "stdafx.h"
#include "ANNRecognition.h"

#define THIS_VERSION  9010101

BOOL APIENTRY DllMain( HANDLE hModule, 
                       DWORD  ul_reason_for_call, 
                       LPVOID lpReserved
					 )
{
    switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
			break;
		case DLL_THREAD_ATTACH:
			break;
		case DLL_THREAD_DETACH:
			break;
		case DLL_PROCESS_DETACH:
			break;
    }
    return TRUE;
}

//Version
ANNRECOGNITION_API int ANNRecognitionVersion(void)
{
	return THIS_VERSION;
}

fun_Logger g_logger=0;

ANNRECOGNITION_API void ANNRecognitionLog( LPSTR message,int logType )
{
	if(g_logger)
		g_logger(logType,message);		
	return ;	
}
ANNRECOGNITION_API void SetLogHandler( fun_Logger logger )
{
	g_logger=logger;
}
