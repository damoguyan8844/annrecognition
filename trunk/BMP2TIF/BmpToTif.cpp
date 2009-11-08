// BmpToTif.cpp : Defines the class behaviors for the application.
//
// this is your duty to fill value according to u...
// any problem? let me know...I may help u...
// w w w . p e c i n t . c o m  (remove space & place in internet & get my con tact)
// sumit(under-score)kapoor1980(at)hot mail(dot) com
// sumit (under-score) kapoor1980(at)ya hoo(dot) com
// sumit (under-score) kapoor1980(at)red iff mail(dot) com

#include "stdafx.h"
#include "BmpToTif.h"
#include "BmpToTifDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CBmpToTifApp

BEGIN_MESSAGE_MAP(CBmpToTifApp, CWinApp)
	//{{AFX_MSG_MAP(CBmpToTifApp)
		// NOTE - the ClassWizard will add and remove mapping macros here.
		//    DO NOT EDIT what you see in these blocks of generated code!
	//}}AFX_MSG
	ON_COMMAND(ID_HELP, CWinApp::OnHelp)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CBmpToTifApp construction

CBmpToTifApp::CBmpToTifApp()
{
	// TODO: add construction code here,
	// Place all significant initialization in InitInstance
}

/////////////////////////////////////////////////////////////////////////////
// The one and only CBmpToTifApp object

CBmpToTifApp theApp;

/////////////////////////////////////////////////////////////////////////////
// CBmpToTifApp initialization

BOOL CBmpToTifApp::InitInstance()
{
	AfxEnableControlContainer();

	// Standard initialization
	// If you are not using these features and wish to reduce the size
	//  of your final executable, you should remove from the following
	//  the specific initialization routines you do not need.

#ifdef _AFXDLL
	Enable3dControls();			// Call this when using MFC in a shared DLL
#else
	Enable3dControlsStatic();	// Call this when linking to MFC statically
#endif

	CBmpToTifDlg dlg;
	m_pMainWnd = &dlg;
	int nResponse = dlg.DoModal();
	if (nResponse == IDOK)
	{
		// TODO: Place code here to handle when the dialog is
		//  dismissed with OK
	}
	else if (nResponse == IDCANCEL)
	{
		// TODO: Place code here to handle when the dialog is
		//  dismissed with Cancel
	}

	// Since the dialog has been closed, return FALSE so that we exit the
	//  application, rather than start the application's message pump.
	return FALSE;
}
