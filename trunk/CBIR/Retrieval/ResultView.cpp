// ResultView.cpp : implementation file
//

#include "stdafx.h"
#include "retrieval.h"
#include "ResultView.h"
#include "retrievalDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CResultView

IMPLEMENT_DYNCREATE(CResultView, CScrollView)

CResultView::CResultView()
{
}

CResultView::~CResultView()
{
}


BEGIN_MESSAGE_MAP(CResultView, CScrollView)
	//{{AFX_MSG_MAP(CResultView)
		// NOTE - the ClassWizard will add and remove mapping macros here.
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CResultView drawing

void CResultView::OnInitialUpdate()
{
	CScrollView::OnInitialUpdate();

	CSize sizeTotal;
	// TODO: calculate the total size of this view
	sizeTotal.cx =1300; 
	sizeTotal.cy = 400;
	SetScrollSizes(MM_TEXT, sizeTotal);
}

void CResultView::OnDraw(CDC* pDC)
{
	CRetrievalDoc* pDoc = GetDocument();
	char temp[20];
	CString stemp;
	int xDest1=27,xDest2=27;
	int yDest=40;
   
	// TODO: add draw code for native data here
	if(pDoc->onresult)
	{
		 pDC->TextOut(10,10,_T("查询结果如下:"));
	for(int i=0;i<10;i++)
	{
		if(i/5 == 0)
		{
	       if (!pDoc->m_pDibResu[i].IsEmpty())
		   {
		      pDoc->m_pDibResu[i].Display(pDC,xDest1,yDest,128,96,0,0,pDoc->m_pDibResu[i].GetWidth(),
			       pDoc->m_pDibResu[i].GetHeight());
		      xDest1+=150;
		   }
		}
		if(i/5 == 1)
		{
          if (!pDoc->m_pDibResu[i].IsEmpty())
		  {
		     pDoc->m_pDibResu[i].Display(pDC,xDest2,yDest+120,128,96,0,0,pDoc->m_pDibResu[i].GetWidth(),
			      pDoc->m_pDibResu[i].GetHeight());
		     xDest2+=150;
		  }
		}
	}
	}
	// TODO: add draw code here
}

/////////////////////////////////////////////////////////////////////////////
// CResultView diagnostics

#ifdef _DEBUG
void CResultView::AssertValid() const
{
	CScrollView::AssertValid();
}

void CResultView::Dump(CDumpContext& dc) const
{
	CScrollView::Dump(dc);
}
CRetrievalDoc* CResultView::GetDocument() // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CRetrievalDoc)));
	return (CRetrievalDoc*)m_pDocument;
}

#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CResultView message handlers
