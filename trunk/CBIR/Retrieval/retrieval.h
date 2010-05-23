// retrieval.h : main header file for the RETRIEVAL application
//

#if !defined(AFX_RETRIEVAL_H__E4826055_CF5B_4A92_9248_01C29F6ADB92__INCLUDED_)
#define AFX_RETRIEVAL_H__E4826055_CF5B_4A92_9248_01C29F6ADB92__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#ifndef __AFXWIN_H__
	#error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"       // main symbols

/////////////////////////////////////////////////////////////////////////////
// CRetrievalApp:
// See retrieval.cpp for the implementation of this class
//

class CRetrievalApp : public CWinApp
{
public:
	CRetrievalApp();

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CRetrievalApp)
	public:
	virtual BOOL InitInstance();
	//}}AFX_VIRTUAL

// Implementation
	//{{AFX_MSG(CRetrievalApp)
	afx_msg void OnAppAbout();
		// NOTE - the ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif 
// !defined(AFX_RETRIEVAL_H__E4826055_CF5B_4A92_9248_01C29F6ADB92__INCLUDED_)
