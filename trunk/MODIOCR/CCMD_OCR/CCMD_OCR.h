// CCMD_OCR.h : main header file for the CCMD_OCR application
//

#if !defined(AFX_CCMD_OCR_H__3C82C5FB_F96B_4A5E_92E1_1DAA4B3C1BEB__INCLUDED_)
#define AFX_CCMD_OCR_H__3C82C5FB_F96B_4A5E_92E1_1DAA4B3C1BEB__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#ifndef __AFXWIN_H__
	#error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"       // main symbols

/////////////////////////////////////////////////////////////////////////////
// CCCMD_OCRApp:
// See CCMD_OCR.cpp for the implementation of this class
//

class CCCMD_OCRApp : public CWinApp
{
public:
	CCCMD_OCRApp();

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CCCMD_OCRApp)
	public:
	virtual BOOL InitInstance();
	//}}AFX_VIRTUAL

// Implementation
	//{{AFX_MSG(CCCMD_OCRApp)
	afx_msg void OnAppAbout();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_CCMD_OCR_H__3C82C5FB_F96B_4A5E_92E1_1DAA4B3C1BEB__INCLUDED_)
