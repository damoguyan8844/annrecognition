// BmpToTifDlg.h : header file
//
// this is your duty to fill value according to u...
// any problem? let me know...I may help u...
// w w w . p e c i n t . c o m  (remove space & place in internet & get my con tact)
// sumit(under-score)kapoor1980(at)hot mail(dot) com
// sumit (under-score) kapoor1980(at)ya hoo(dot) com
// sumit (under-score) kapoor1980(at)red iff mail(dot) com

#if !defined(AFX_BMPTOTIFDLG_H__46CA0AC7_3A15_11D8_A667_0050BA8B6949__INCLUDED_)
#define AFX_BMPTOTIFDLG_H__46CA0AC7_3A15_11D8_A667_0050BA8B6949__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

/////////////////////////////////////////////////////////////////////////////
// CBmpToTifDlg dialog

class CBmpToTifDlg : public CDialog
{
// Construction
// this is your duty to fill value according to u...
// any problem? let me know...I may help u...
// w w w . p e c i n t . c o m  (remove space & place in internet & get my con-tact)
// sumit(under-score)kapoor1980(at)hot mail(dot) com
// sumit (under-score) kapoor1980(at)ya hoo(dot) com
// sumit (under-score) kapoor1980(at)red iff mail(dot) com




public:
	CBmpToTifDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	//{{AFX_DATA(CBmpToTifDlg)
	enum { IDD = IDD_BMPTOTIF_DIALOG };
		// NOTE: the ClassWizard will add data members here
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CBmpToTifDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CBmpToTifDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnConvert();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_BMPTOTIFDLG_H__46CA0AC7_3A15_11D8_A667_0050BA8B6949__INCLUDED_)
