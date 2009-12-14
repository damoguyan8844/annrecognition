
#ifndef ANNRECOGNITION_LOCK
#define ANNRECOGNITION_LOCK 

struct Lock
{
	Lock(CRITICAL_SECTION * cs) :_cs(cs)
	{	::EnterCriticalSection(_cs);	}
	~Lock()
	{	::LeaveCriticalSection(_cs);	}
	CRITICAL_SECTION * _cs;
}  ;

extern CRITICAL_SECTION _cs;
extern CRITICAL_SECTION _csOCR;

#endif