// retrievalDoc.h : interface of the CRetrievalDoc class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_RETRIEVALDOC_H__DEFC54C6_1757_4109_BB46_C85F4C040DA2__INCLUDED_)
#define AFX_RETRIEVALDOC_H__DEFC54C6_1757_4109_BB46_C85F4C040DA2__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
#include"Dib.h"
#include"Jpeg.h"



class CRetrievalDoc : public CDocument
{
protected: // create from serialization only
	CRetrievalDoc();
	DECLARE_DYNCREATE(CRetrievalDoc)

// Attributes
public:
	 void load_telib(int);
     bool onoff[3];
	 bool ifSelectlib;
     CString m_filenameExam[3];
	 CDib m_Exam[3];
     int m_graphExam[3];
	 CDib m_pDibResu[10];
	 bool onresult;
	 CRect rect[3];
	
	
     struct image{
		 char filename[40];
		 float te[3][12];
	 };
	 struct image imagelib[40];
	 int libnumber;


// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CRetrievalDoc)
	public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CRetrievalDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	//{{AFX_MSG(CRetrievalDoc)
	afx_msg void OnWinter();
	afx_msg void OnFlag();
	afx_msg void OnFlower();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_RETRIEVALDOC_H__DEFC54C6_1757_4109_BB46_C85F4C040DA2__INCLUDED_)
