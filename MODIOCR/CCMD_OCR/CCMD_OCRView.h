// CCMD_OCRView.h : interface of the CCCMD_OCRView class
//
/////////////////////////////////////////////////////////////////////////////
//{{AFX_INCLUDES()

//}}AFX_INCLUDES

#if !defined(AFX_CCMD_OCRVIEW_H__5CB64813_DF87_441A_A628_7FF3C4E25363__INCLUDED_)
#define AFX_CCMD_OCRVIEW_H__5CB64813_DF87_441A_A628_7FF3C4E25363__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "../class/imginclude/ximage.h"//../class/
#ifdef _DEBUG
	#pragma comment( lib, "../class/imglibd/CxImage.lib")
	#pragma comment( lib, "../class/imglibd/j2k.lib")
	#pragma comment( lib, "../class/imglibd/jbig.lib")
	#pragma comment( lib, "../class/imglibd/Jpeg.lib")
	#pragma comment( lib, "../class/imglibd/png.lib")
	#pragma comment( lib, "../class/imglibd/Tiff.lib")
	#pragma comment( lib, "../class/imglibd/zlib.lib")
	#pragma comment( lib, "../class/imglibd/jasper.lib")
#else
	#pragma comment( lib, "../class/imglib/CxImage.lib")
	#pragma comment( lib, "../class/imglib/j2k.lib")
	#pragma comment( lib, "../class/imglib/jbig.lib")
	#pragma comment( lib, "../class/imglib/Jpeg.lib")
	#pragma comment( lib, "../class/imglib/png.lib")
	#pragma comment( lib, "../class/imglib/Tiff.lib")
	#pragma comment( lib, "../class/imglib/zlib.lib")
	#pragma comment( lib, "../class/imglib/jasper.lib")
#endif
class CCCMD_OCRView : public CFormView
{
protected: // create from serialization only
	CCCMD_OCRView();
	DECLARE_DYNCREATE(CCCMD_OCRView)

public:
	//{{AFX_DATA(CCCMD_OCRView)
	enum { IDD = IDD_CCMD_OCR_FORM };
	CListBox	m_list;
	//}}AFX_DATA

// Attributes
public:
	CCCMD_OCRDoc* GetDocument();
//定义变量
	CxImage  *image;
	HWND	g_hWndview;		//主窗口句柄	
	BYTE*	pBuffer;
	float	hBitW;
	float	hBitH;
	HDC		hScrDC,hDC;	
	CRect	rect;
	CString FileDir;
	CString cn;
	DWORD   type;
public:
	BOOL OCRImageFile	(CString Name);
	void FindFile		(CString DirName);	//查找文件
	void ShowJpg		(CString Name);		//显示图形
	int  ReadJpg		(CString Name);		//读图形文件
// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CCCMD_OCRView)
	public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	virtual void OnInitialUpdate(); // called first time after construct
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CCCMD_OCRView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	//{{AFX_MSG(CCCMD_OCRView)
	afx_msg void OnPaint();
	afx_msg void OnFileOpen();
	afx_msg void OnSelchangeList1();
	afx_msg void OnFileNew();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

#ifndef _DEBUG  // debug version in CCMD_OCRView.cpp
inline CCCMD_OCRDoc* CCCMD_OCRView::GetDocument()
   { return (CCCMD_OCRDoc*)m_pDocument; }
#endif

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_CCMD_OCRVIEW_H__5CB64813_DF87_441A_A628_7FF3C4E25363__INCLUDED_)
