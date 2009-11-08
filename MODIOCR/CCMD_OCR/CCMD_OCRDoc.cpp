// CCMD_OCRDoc.cpp : implementation of the CCCMD_OCRDoc class
//

#include "stdafx.h"
#include "CCMD_OCR.h"

#include "CCMD_OCRDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CCCMD_OCRDoc

IMPLEMENT_DYNCREATE(CCCMD_OCRDoc, CDocument)

BEGIN_MESSAGE_MAP(CCCMD_OCRDoc, CDocument)
	//{{AFX_MSG_MAP(CCCMD_OCRDoc)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CCCMD_OCRDoc construction/destruction

CCCMD_OCRDoc::CCCMD_OCRDoc()
{
}

CCCMD_OCRDoc::~CCCMD_OCRDoc()
{
}

BOOL CCCMD_OCRDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	return TRUE;
}



/////////////////////////////////////////////////////////////////////////////
// CCCMD_OCRDoc serialization

void CCCMD_OCRDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
	}
	else
	{
	}
}

/////////////////////////////////////////////////////////////////////////////
// CCCMD_OCRDoc diagnostics

#ifdef _DEBUG
void CCCMD_OCRDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CCCMD_OCRDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CCCMD_OCRDoc commands
