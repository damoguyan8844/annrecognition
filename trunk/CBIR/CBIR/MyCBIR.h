// MyCBIR.h : PROJECT_NAME Ӧ�ó������ͷ�ļ�
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"		// ������


// CMyCBIRApp:
// �йش����ʵ�֣������ MyCBIR.cpp
//

class CMyCBIRApp : public CWinApp
{
public:
	CMyCBIRApp();

// ��д
	public:
	virtual BOOL InitInstance();
// GDI+��Դ
	ULONG_PTR m_gdiplusToken;

// ʵ��

	DECLARE_MESSAGE_MAP()
public:
	virtual int ExitInstance();
};

extern CMyCBIRApp theApp;