// CCMD_OCRDoc.h : interface of the CCCMD_OCRDoc class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_CCMD_OCRDOC_H__DD3520A7_F8B0_4C7D_AC1B_7663E6444E12__INCLUDED_)
#define AFX_CCMD_OCRDOC_H__DD3520A7_F8B0_4C7D_AC1B_7663E6444E12__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000


class CCCMD_OCRDoc : public CDocument
{
protected: // create from serialization only
	CCCMD_OCRDoc();
	DECLARE_DYNCREATE(CCCMD_OCRDoc)

// Attributes
public:

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CCCMD_OCRDoc)
	public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CCCMD_OCRDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	//{{AFX_MSG(CCCMD_OCRDoc)
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_CCMD_OCRDOC_H__DD3520A7_F8B0_4C7D_AC1B_7663E6444E12__INCLUDED_)
