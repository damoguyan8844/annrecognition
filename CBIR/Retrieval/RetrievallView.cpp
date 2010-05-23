// RetrievallView.cpp : implementation file
//

#include "stdafx.h"
#include "retrieval.h"
#include "mainFrm.h"
#include "stdio.h"
#include "math.h"

#include "retrievalDoc.h"
#include "retrievallView.h"
#include "Dib.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CRetrievallView

IMPLEMENT_DYNCREATE(CRetrievallView, CScrollView)

CRetrievallView::CRetrievallView()
{
	for(int i=0;i<3;i++)
		for(int j=0;j<12;j++)
			tezheng[i][j] = 0.0;
	
	for(i=0;i<40;i++)
	{	
		result_te[i]=0;
        cixu[i]=i;
	}
	if_lbd = false;
	ifSelect=false;

	

}

CRetrievallView::~CRetrievallView()
{
}


BEGIN_MESSAGE_MAP(CRetrievallView, CScrollView)
	//{{AFX_MSG_MAP(CRetrievallView)
	ON_WM_LBUTTONDOWN()
	ON_WM_CONTEXTMENU()
	ON_COMMAND(ID_MENUITEM32776, OnHSV_GEN_OU)
	ON_COMMAND(ID_MENUITEM32777, OnHSV_GEN_QUAN)
	ON_COMMAND(ID_MENUITEM32778, OnHSV_GEN_JIAO)
	ON_COMMAND(ID_MENUITEM32779, OnHSV_SUC_OU)
	ON_COMMAND(ID_MENUITEM32780, OnHSV_CEN_OU)
	ON_COMMAND(ID_MTM1, OnMTM_GEN_OU)
	ON_COMMAND(ID_MTM2, OnMTM_GEN_QUAN)
	ON_COMMAND(ID_MTM3, OnMTM_GEN_JIAO)
	ON_COMMAND(ID_MTM4, OnMTM_SUC_OU)
	ON_COMMAND(ID_MTM5, OnMTM_CEN_OU)
	ON_COMMAND(ID_HSV1_2, OnHSV1_2)
	ON_COMMAND(ID_HSV2_2, OnHSV2_2)
	ON_COMMAND(ID_HSV3_2, OnHSV3_2)
	ON_COMMAND(ID_HSV4_2, OnHSV4_2)
	ON_COMMAND(ID_HSV4_3, OnHSV4_3)
	ON_COMMAND(ID_HSV5_2, OnHSV5_2)
	ON_COMMAND(ID_Rotate, OnRotate)
	ON_COMMAND(ID_big, OnBig)
	ON_COMMAND(ID_small, OnSmall)
	ON_COMMAND(ID_Move, OnMove)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CRetrievallView drawing

void CRetrievallView::OnInitialUpdate()
{
	CScrollView::OnInitialUpdate();

	CSize sizeTotal;
	// TODO: calculate the total size of this view
	sizeTotal.cx =1200; 
	sizeTotal.cy =200;
	SetScrollSizes(MM_TEXT, sizeTotal);
}

void CRetrievallView::OnDraw(CDC* pDC)
{
    CRetrievalDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	
    if(pDoc->ifSelectlib)
	{
	   pDC->TextOut(10,30,_T("图像库选择后,请单击感兴趣的查询图像:"));
	

	   for(int i=0;i<3;i++)
	   {
		if (!pDoc->m_Exam[i].IsEmpty())
		{
		 pDoc->m_Exam[i].Display(pDC,pDoc->rect[i].left,pDoc->rect[i].top,pDoc->rect[i].right-pDoc->rect[i].left,
			 pDoc->rect[i].bottom-pDoc->rect[i].top,0,0,pDoc->m_Exam[i].GetWidth(),
			pDoc->m_Exam[i].GetHeight());

		}
    	  if( if_lbd &&(m_exam==i))
		  {
			CPen pen(PS_DASH,1,RGB(0,0,0));
	        CPen * pOldPen = pDC->SelectObject(&pen);

            pDC->MoveTo(pDoc->rect[i].left-2,pDoc->rect[i].top-2);
			pDC->LineTo(pDoc->rect[i].right+2,pDoc->rect[i].top-2);
			pDC->LineTo(pDoc->rect[i].right+2,pDoc->rect[i].bottom+2);
			pDC->LineTo(pDoc->rect[i].left-2,pDoc->rect[i].bottom+2);
			pDC->LineTo(pDoc->rect[i].left-2,pDoc->rect[i].top-2);
           
			if_lbd = false;
		}
       
	}
	

	}
	// TODO: add draw code here
}

/////////////////////////////////////////////////////////////////////////////
// CRetrievallView diagnostics

#ifdef _DEBUG
void CRetrievallView::AssertValid() const
{
	CScrollView::AssertValid();
}

void CRetrievallView::Dump(CDumpContext& dc) const
{
	CScrollView::Dump(dc);
}

CRetrievalDoc* CRetrievallView::GetDocument() // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CRetrievalDoc)));
	return (CRetrievalDoc*)m_pDocument;
}

#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CRetrievallView message handlers

int CRetrievallView::mymax(int a,int b,int c)
{

	int m;
	if(a>b)
		m=a;
	else
		m=b;
	if(m<c)
		m=c;
	return m;
}

int CRetrievallView::mymin(int a,int b,int c)
{

	int m;
	if(a<b)
		m=a;
	else 
		m=b;
	if(m>c)
		m=c;
	return m;
}

double CRetrievallView::mtm_fun(double x)
{
	double y;
	y = 11.6 * pow( x, 1.0/3.0 ) - 1.6;
	return y;
}

void CRetrievallView::RGBToHSV(int r,int g,int b,double *h,double *s,double *v)
{
	*h=acos((r-g+r-b)/(2.0*sqrtf((float)(r-g)*(r-g)+(float)(r-b)*(g-b))));
	if(b>g)
		*h=2*PI-*h;
    *s=(mymax(r,g,b)-mymin(r,g,b))/(float)mymax(r,g,b);
	*v=mymax(r,g,b)/255.0;
}

void CRetrievallView::RGBToMTM(int r,int g,int b,double *h,double *v,double *c)
{
	double x,y,z;
	x = 0.620 * r + 0.178 * g + 0.204 * b;
	y = 0.299 * r + 0.587 * g + 0.144 * b;
	z = 0.056 * g + 0.942 * b;

	double p,q,sita,s,t;
	p = mtm_fun(x) - mtm_fun(y);
	q = 0.4 *(mtm_fun(z) - mtm_fun(y));
    sita = atan(p/q);
	s = ( 8.880 + 0.966 * cos(sita)) * p;
	t = ( 8.025 + 2.558 * sin(sita)) * q;
	*h = atan(s/t);
	*v = mtm_fun(y);
	*c = sqrt( s * s + t * t );
}

void CRetrievallView::hsv_general(int gap)
{
	//gap  1: 12:12:12  2: 8:3:3   3: 6:6:6
    CRetrievalDoc* pDoc = GetDocument();

	COLORREF color;
	double h=0,s=0,v=0;
	long Height=pDoc->m_Exam[m_exam].GetHeight();
	long Width=pDoc->m_Exam[m_exam].GetWidth();
	long totalnum= Height * Width;
	long m_graph[3][12];
    
	for(int i=0;i<3;i++)
	{	
		for(int j=0;j<12;j++)
		{
		   m_graph[i][j]=0;
		   tezheng[i][j]=0.0;
		}
	}
	if(gap==1)
	{
	for(long cy=0;cy<Height;cy++)
	{	
		for(long cx=0;cx<Width;cx++)
		{			
			color=pDoc->m_Exam[m_exam].GetPixel(cx,cy);
			RGBToHSV(GetRValue(color),GetGValue(color),GetBValue(color),&h,&s,&v);
		    int result_h=(int)(6*h/PI);
			if( result_h ==12)
				m_graph[0][11]++;
			else
        	    m_graph[0][result_h]++;

            int result_s=(int)(s*12);
			if( result_s ==12)
				m_graph[1][11]++;
			else
			    m_graph[1][result_s]++;

            int result_v=(int)(v*12);
			if( result_v ==12)
				m_graph[2][11]++;
			else
			    m_graph[2][result_v]++;
		}
	}	

	for(i=0;i<3;i++)
	{
    for(int j=0;j<12;j++)
	{
		tezheng[i][j]=((float)m_graph[i][j])/((float)totalnum);
	}
	}
	}
	if(gap==2)
	{
    for(long cy=0;cy<Height;cy++)
	{	
		for(long cx=0;cx<Width;cx++)
		{			
			color=pDoc->m_Exam[m_exam].GetPixel(cx,cy);
			RGBToHSV(GetRValue(color),GetGValue(color),GetBValue(color),&h,&s,&v);
		    int result_h=(int)(4*h/PI);
			if( result_h ==8)
				m_graph[0][7]++;
			else
        	    m_graph[0][result_h]++;

            int result_s=(int)(s*3);
			if( result_s ==3)
				m_graph[1][2]++;
			else
			    m_graph[1][result_s]++;

            int result_v=(int)(v*3);
			if( result_v ==2)
				m_graph[2][2]++;
			else
			    m_graph[2][result_v]++;
		}
	}	
	
	for(int j=0;j<8;j++)
	{
		tezheng[0][j]=((float)m_graph[0][j])/((float)totalnum);
	}
	for(i=1;i<3;i++)
	{
    for(int j=0;j<3;j++)
	{
		tezheng[i][j]=((float)m_graph[i][j])/((float)totalnum);
	}
	}
	}
}

void CRetrievallView::hsv_succession(int gap)
{
    CRetrievalDoc* pDoc = GetDocument();

	COLORREF color;
	double h=0,s=0,v=0;
		long Height=pDoc->m_Exam[m_exam].GetHeight();
	long Width=pDoc->m_Exam[m_exam].GetWidth();
	long totalnum= Height * Width;
	long m_graph[3][12];
    
	for(int i=0;i<3;i++)
	{	
		for(int j=0;j<12;j++)
		{
		   m_graph[i][j]=0;
		   tezheng[i][j]=0.0;
		}
	}
	if(gap==1)
	{
	for(long cy=0;cy<Height;cy++)
	{	
		for(long cx=0;cx<Width;cx++)
		{			
			color=pDoc->m_Exam[m_exam].GetPixel(cx,cy);
			RGBToHSV(GetRValue(color),GetGValue(color),GetBValue(color),&h,&s,&v);
		    int result_h=(int)(6*h/PI);
			if( result_h ==12)
				m_graph[0][11]++;
			else
        	    m_graph[0][result_h]++;

            int result_s=(int)(s*12);
			if( result_s ==12)
				m_graph[1][11]++;
			else
			    m_graph[1][result_s]++;

            int result_v=(int)(v*12);
			if( result_v ==12)
				m_graph[2][11]++;
			else
			    m_graph[2][result_v]++;
		}
	}
	long temp[3][12];
	for( i=0;i<3;i++)
	{
		for(int j=0;j<12;j++)
		{
		   temp[i][j]=m_graph[i][j];
		}
	}
	for(i=0;i<3;i++)
	{
       for(int j=0;j<12;j++)
	   {
	      for(int k=0;k<j;k++)
		  {
		     m_graph[i][j]+=temp[i][k];
		  }
	   }
	}

	for(i=0;i<3;i++)
	{
    for(int j=0;j<12;j++)
	{
		tezheng[i][j]=((float)m_graph[i][j])/((float)totalnum);
	}
	}
	}
	if(gap==2)
	{
    for(long cy=0;cy<Height;cy++)
	{	
		for(long cx=0;cx<Width;cx++)
		{			
			color=pDoc->m_Exam[m_exam].GetPixel(cx,cy);
			RGBToHSV(GetRValue(color),GetGValue(color),GetBValue(color),&h,&s,&v);
		    int result_h=(int)(4*h/PI);
			if( result_h ==8)
				m_graph[0][7]++;
			else
        	    m_graph[0][result_h]++;

            int result_s=(int)(s*3);
			if( result_s ==3)
				m_graph[1][2]++;
			else
			    m_graph[1][result_s]++;

            int result_v=(int)(v*3);
			if( result_v ==2)
				m_graph[2][2]++;
			else
			    m_graph[2][result_v]++;
		}
	}
	for(i=1;i<3;i++)
		for(int j=0;j<8;j++)
			m_graph[i][j]=0;

	long temp[3][8];
	for( i=0;i<3;i++)
	{
		for(int j=0;j<8;j++)
		{
		   temp[i][j]=m_graph[i][j];
		}
	}
	for(i=0;i<3;i++)
	{
       for(int j=0;j<8;j++)
	   {
	      for(int k=0;k<j;k++)
		  {
		     m_graph[i][j]+=temp[i][k];
		  }
	   }
	}

	for(int j=0;j<8;j++)
	{
		tezheng[0][j]=((float)m_graph[0][j])/((float)totalnum);
	}
	for(i=1;i<3;i++)
	{
    for(int j=0;j<3;j++)
	{
		tezheng[i][j]=((float)m_graph[i][j])/((float)totalnum);
	}
	}
	}
	if(gap==3)
	{
    for(long cy=0;cy<Height;cy++)
	{	
		for(long cx=0;cx<Width;cx++)
		{			
			color=pDoc->m_Exam[m_exam].GetPixel(cx,cy);
			RGBToHSV(GetRValue(color),GetGValue(color),GetBValue(color),&h,&s,&v);
		    int result_h=(int)(3*h/PI);
			if( result_h ==6)
				m_graph[0][5]++;
			else
        	    m_graph[0][result_h]++;

            int result_s=(int)(s*6);
			if( result_s ==5)
				m_graph[1][5]++;
			else
			    m_graph[1][result_s]++;

            int result_v=(int)(v*6);
			if( result_v ==6)
				m_graph[2][5]++;
			else
			    m_graph[2][result_v]++;
		}
	}
	long temp[3][6];
	for( i=0;i<3;i++)
	{
		for(int j=0;j<6;j++)
		{
		   temp[i][j]=m_graph[i][j];
		}
	}
	for(i=0;i<3;i++)
	{
       for(int j=0;j<6;j++)
	   {
	      for(int k=0;k<j;k++)
		  {
		     m_graph[i][j]+=temp[i][k];
		  }
	   }
	}

	for(i=0;i<3;i++)
	{
    for(int j=0;j<6;j++)
	{
		tezheng[i][j]=((float)m_graph[i][j])/((float)totalnum);
	}
	}
	}
}
void CRetrievallView::hsv_centerM(int gap)
{
    CRetrievalDoc* pDoc = GetDocument();

    COLORREF color;
	double h=0,s=0,v=0;
		long Height=pDoc->m_Exam[m_exam].GetHeight();
	long Width=pDoc->m_Exam[m_exam].GetWidth();
	long totalnum= Height * Width;
	long m_graph[3][12];
	float m_graphf[3][12];
    
	for(int i=0;i<3;i++)
	{	
		for(int j=0;j<12;j++)
		{
		   m_graph[i][j]=0;
		   tezheng[i][j]=0.0;
		   m_graphf[i][j]=0.0;
		}
	}
	if(gap==1)
	{
    for(long cy=0;cy<Height;cy++)
	{	
		for(long cx=0;cx<Width;cx++)
		{			
			color=pDoc->m_Exam[m_exam].GetPixel(cx,cy);
			RGBToHSV(GetRValue(color),GetGValue(color),GetBValue(color),&h,&s,&v);
		    int result_h=(int)(6*h/PI);
			if( result_h ==12)
				m_graph[0][11]++;
			else
        	    m_graph[0][result_h]++;

            int result_s=(int)(s*12);
			if( result_s ==12)
				m_graph[1][11]++;
			else
			    m_graph[1][result_s]++;

            int result_v=(int)(v*12);
			if( result_v ==12)
				m_graph[2][11]++;
			else
			    m_graph[2][result_v]++;
		}
	}
    for( i=0;i<3;i++)
	{
		for(int j=0;j<12;j++)
		{
		   m_graphf[i][j]=((float)m_graph[i][j])/((float)totalnum);
		}
	}
    float m1[3],m2[3],m3[3];
	for( i=0;i<3;i++)
	{
		m1[i] = 0.0;
		m2[i] = 0.0;
		m3[i] = 0.0;
	}
	for(i=0;i<3;i++)
	{
	for(int j=0;j<12;j++)
		m1[i] +=m_graphf[i][j]/12;
	}

	for(i=0;i<3;i++)
	{
	for(int j=0;j<12;j++)
	{
		m2[i] +=((m_graphf[i][j] - m1[i]) * (m_graphf[i][j] - m1[i]))/12;
		m3[i] +=((m_graphf[i][j] - m1[i]) * (m_graphf[i][j] - m1[i])
				* (m_graphf[i][j] - m1[i]))/12;
	}
    }

    for( i=0;i<3;i++)
	{
		m2[i] = sqrtf(m2[i]);
		m3[i] = (float)pow( m3[i], 1.0/3.0 );
	}
    tezheng[0][0]=m1[0]; tezheng[0][1]=m2[0]; tezheng[0][2]=m3[0];
	tezheng[1][0]=m1[1]; tezheng[1][1]=m2[1]; tezheng[1][2]=m3[1];
	tezheng[2][0]=m1[2]; tezheng[2][1]=m2[2]; tezheng[2][2]=m3[2];
	}

	if(gap==2)
	{
    for(long cy=0;cy<Height;cy++)
	{	
		for(long cx=0;cx<Width;cx++)
		{			
			color=pDoc->m_Exam[m_exam].GetPixel(cx,cy);
			RGBToHSV(GetRValue(color),GetGValue(color),GetBValue(color),&h,&s,&v);
		    int result_h=(int)(4*h/PI);
			if( result_h ==8)
				m_graph[0][7]++;
			else
        	    m_graph[0][result_h]++;

            int result_s=(int)(s*3);
			if( result_s ==3)
				m_graph[1][2]++;
			else
			    m_graph[1][result_s]++;

            int result_v=(int)(v*3);
			if( result_v ==3)
				m_graph[2][2]++;
			else
			    m_graph[2][result_v]++;
		}
	}
    for( i=0;i<3;i++)
	{
		for(int j=0;j<8;j++)
		{
		   m_graphf[i][j]=((float)m_graph[i][j])/((float)totalnum);
		}
	}
    float m1[3],m2[3],m3[3];
	for( i=0;i<3;i++)
	{
		m1[i] = 0.0;
		m2[i] = 0.0;
		m3[i] = 0.0;
	}

	for(int j=0;j<8;j++)
		m1[0] +=m_graphf[0][j]/8;

	for(i=1;i<3;i++)
	{
	for(int j=0;j<3;j++)
		m1[i] +=m_graphf[i][j]/3;
	}
   
    for(j=0;j<8;j++)
	{
		m2[0] +=((m_graphf[0][j] - m1[0]) * (m_graphf[0][j] - m1[0]))/8;
		m3[0] +=((m_graphf[0][j] - m1[0]) * (m_graphf[0][j] - m1[0])
				* (m_graphf[0][j] - m1[0]))/8;
	}
	for(i=1;i<3;i++)
	{
	for(int j=0;j<3;j++)
	{
		m2[i] +=((m_graphf[i][j] - m1[i]) * (m_graphf[i][j] - m1[i]))/3;
		m3[i] +=((m_graphf[i][j] - m1[i]) * (m_graphf[i][j] - m1[i])
				* (m_graphf[i][j] - m1[i]))/3;
	}
    }

    for( i=0;i<3;i++)
	{
		m2[i] = sqrtf(m2[i]);
		m3[i] = (float)pow( m3[i], 1.0/3.0 );
	}
    tezheng[0][0]=m1[0]; tezheng[0][1]=m2[0]; tezheng[0][2]=m3[0];
	tezheng[1][0]=m1[1]; tezheng[1][1]=m2[1]; tezheng[1][2]=m3[1];
	tezheng[2][0]=m1[2]; tezheng[2][1]=m2[2]; tezheng[2][2]=m3[2];
	}
}
void CRetrievallView::mtm_general(int gap)
{
    CRetrievalDoc* pDoc = GetDocument();

	COLORREF color;
	double h=0,v=0,c=0;
		long Height=pDoc->m_Exam[m_exam].GetHeight();
	long Width=pDoc->m_Exam[m_exam].GetWidth();
	long totalnum= Height * Width;
	long m_graph[3][12];
    
	for(int i=0;i<3;i++)
	{	
		for(int j=0;j<12;j++)
		{
		   m_graph[i][j]=0;
		   tezheng[i][j]=0.0;
		}
	}
	if(gap==1)
	{
	for(long cy=0;cy<Height;cy++)
	{	
		for(long cx=0;cx<Width;cx++)
		{			
			color=pDoc->m_Exam[m_exam].GetPixel(cx,cy);
			RGBToMTM(GetRValue(color),GetGValue(color),GetBValue(color),&h,&v,&c);
		    int result_h=(int)(6*(h+PI/2)/PI);
			if( result_h ==12)
				m_graph[0][11]++;
			else
        	    m_graph[0][result_h]++;

            int result_v=(int)((v-4.4801)/5.627);
			if( result_v ==12)
				m_graph[1][11]++;
			else
			    m_graph[1][result_v]++;

            int result_c=(int)((c-0.017268)/19.483311);
			if( result_c ==12)
				m_graph[2][11]++;
			else
			    m_graph[2][result_c]++;
		}
	}	

	for(i=0;i<3;i++)
	{
    for(int j=0;j<12;j++)
	{
		tezheng[i][j]=((float)m_graph[i][j])/((float)totalnum);
	}
	}
	}
}
void CRetrievallView::mtm_succession(int gap)
{
    CRetrievalDoc* pDoc = GetDocument();

	COLORREF color;
	double h=0,v=0,c=0;
	long Height=pDoc->m_Exam[m_exam].GetHeight();
	long Width=pDoc->m_Exam[m_exam].GetWidth();
	long totalnum= Height * Width;
	long m_graph[3][12];
    
	for(int i=0;i<3;i++)
	{	
		for(int j=0;j<12;j++)
		{
		   m_graph[i][j]=0;
		   tezheng[i][j]=0.0;
		}
	}
	if(gap==1)
	{
	for(long cy=0;cy<Height;cy++)
	{	
		for(long cx=0;cx<Width;cx++)
		{			
			color=pDoc->m_Exam[m_exam].GetPixel(cx,cy);
			RGBToMTM(GetRValue(color),GetGValue(color),GetBValue(color),&h,&v,&c);
		     int result_h=(int)(6*(h+PI/2)/PI);
			if( result_h ==12)
				m_graph[0][11]++;
			else
        	    m_graph[0][result_h]++;

            int result_v=(int)((v-4.4801)/5.627);
			if( result_v ==12)
				m_graph[1][11]++;
			else
			    m_graph[1][result_v]++;

            int result_c=(int)((c-0.017268)/19.483311);
			if( result_c ==12)
				m_graph[2][11]++;
			else
			    m_graph[2][result_c]++;
		}
	}	
    long temp[3][12];
	for( i=0;i<3;i++)
	{
		for(int j=0;j<12;j++)
		{
		   temp[i][j]=m_graph[i][j];
		}
	}
	for(i=0;i<3;i++)
	{
       for(int j=0;j<12;j++)
	   {
	      for(int k=0;k<j;k++)
		  {
		     m_graph[i][j]+=temp[i][k];
		  }
	   }
	}
    for(i=0;i<3;i++)
	{
    for(int j=0;j<12;j++)
	{
		tezheng[i][j]=((float)m_graph[i][j])/((float)totalnum);
	}
	}
	}
}

void CRetrievallView::mtm_centerM(int gap)
{
    CRetrievalDoc* pDoc = GetDocument();

	COLORREF color;
	double h=0,v=0,c=0;
		long Height=pDoc->m_Exam[m_exam].GetHeight();
	long Width=pDoc->m_Exam[m_exam].GetWidth();
	long totalnum= Height * Width;
	long m_graph[3][12];
	float m_graphf[3][12];
    
	for(int i=0;i<3;i++)
	{	
		for(int j=0;j<12;j++)
		{
		   m_graph[i][j]=0;
		   tezheng[i][j]=0.0;
		   m_graphf[i][j]=0.0;
		}
	}
	if(gap==1)
	{
	for(long cy=0;cy<Height;cy++)
	{	
		for(long cx=0;cx<Width;cx++)
		{			
			color=pDoc->m_Exam[m_exam].GetPixel(cx,cy);
			RGBToMTM(GetRValue(color),GetGValue(color),GetBValue(color),&h,&v,&c);
		     int result_h=(int)(6*(h+PI/2)/PI);
			if( result_h ==12)
			        m_graph[0][11]++;
			else
        	    m_graph[0][result_h]++;

            int result_v=(int)((v-4.4801)/5.627);
			if( result_v ==12)
				m_graph[1][11]++;
			else
			    m_graph[1][result_v]++;

            int result_c=(int)((c-0.017268)/19.483311);
			if( result_c ==12)
				m_graph[2][11]++;
			else
			    m_graph[2][result_c]++;
		}
	}
	for( i=0;i<3;i++)
	{
		for(int j=0;j<12;j++)
		{
		   m_graphf[i][j]=((float)m_graph[i][j])/((float)totalnum);
		}
	}
    float m1[3],m2[3],m3[3];
	for( i=0;i<3;i++)
	{
		m1[i] = 0.0;
		m2[i] = 0.0;
		m3[i] = 0.0;
	}
	for(i=0;i<3;i++)
	{
	for(int j=0;j<12;j++)
		m1[i] +=m_graphf[i][j]/12;
	}

	for(i=0;i<3;i++)
	{
	for(int j=0;j<12;j++)
	{
		m2[i] +=((m_graphf[i][j] - m1[i]) * (m_graphf[i][j] - m1[i]))/12;
		m3[i] +=((m_graphf[i][j] - m1[i]) * (m_graphf[i][j] - m1[i])
				* (m_graphf[i][j] - m1[i]))/12;
	}
    }

    for( i=0;i<3;i++)
	{
		m2[i] = sqrtf(m2[i]);
		m3[i] = (float)pow( m3[i], 1.0/3.0 );
	}
    tezheng[0][0]=m1[0]; tezheng[0][1]=m2[0]; tezheng[0][2]=m3[0];
	tezheng[1][0]=m1[1]; tezheng[1][1]=m2[1]; tezheng[1][2]=m3[1];
	tezheng[2][0]=m1[2]; tezheng[2][1]=m2[2]; tezheng[2][2]=m3[2];
	}
}
float CRetrievallView::Caculate(int model,int te,int d,int type)    //相交法
{
	//type  1: 12:12:12  2: 8:3:3  3: 6:6:6 
    CRetrievalDoc* pDoc = GetDocument();
	float min[3];
	float min1[3];
	float result;

	for(int i=0;i<3;i++)
	{
		min[i]=0.0;
	    min1[i]=0.0;
	}

if(model==1)
{
    if(type==1)
	{
		switch (te)
		{
          case 1:
		      hsv_general(1);
		      break;
          case 2:
	          hsv_succession(1);
	          break;
          case 3:
	          hsv_centerM(1);
	          break;
          default:
	      break;
		}

	for(i=0;i<3;i++)
	{
    for(int j=0;j<12;j++)
		{
		   min[i]+=mintwo(tezheng[i][j],pDoc->imagelib[d].te[i][j]);
           min1[i]+=tezheng[i][j];
		}
	}
	}

	if(type==2)
	{
		switch (te)
		{
          case 1:
		      hsv_general(2);
		      break;
          case 2:
	          hsv_succession(2);
	          break;
          case 3:
	          hsv_centerM(2);
	          break;
          default:
	      break;
		}
     for(int j=0;j<8;j++)
		{
		   min[0]+=mintwo(tezheng[0][j],pDoc->imagelib[d].te[0][j]);
           min1[0]+=tezheng[0][j];
		}
	for(i=1;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
		   min[i]+=mintwo(tezheng[i][j],pDoc->imagelib[d].te[i][j]);
           min1[i]+=tezheng[i][j];
		}
	}
	}

	if(type==3)
	{
		switch (te)
		{
          case 1:
		      hsv_general(3);
		      break;
          case 2:
	          hsv_succession(3);
	          break;
          case 3:
	          hsv_centerM(3);
	          break;
          default:
	      break;
		}
    for(i=0;i<3;i++)
	{
    for(int j=0;j<6;j++)
		{
		   min[i]+=mintwo(tezheng[i][j],pDoc->imagelib[d].te[i][j]);
           min1[i]+=tezheng[i][j];
		}
	}
	}
}

if(model==2)
{
    if(type==1)
	{
	    switch (te)
		{
          case 1:
		      mtm_general(1);
		      break;
          case 2:
	          mtm_succession(1);
	          break;
          case 3:
	          mtm_centerM(1);
	          break;
          default:
	      break;
		}	
	for(i=0;i<3;i++)
	{
    for(int j=0;j<12;j++)
		{
		   min[i]+=mintwo(tezheng[i][j],pDoc->imagelib[d].te[i][j]);
           min1[i]+=tezheng[i][j];
		}
	}
	}

}

	result =min[0]/min1[0] + min[1]/min1[1] + min[2]/min1[2];
	return result;
}
float CRetrievallView::distance(int model,int te,int d,int type)    //欧式距离
{
	CRetrievalDoc* pDoc = GetDocument();
	float temp[3];
	float result;
	for(int i=0;i<3;i++)
		temp[i] =0.0;

if(model==1)
{
    if(type==1)
	{
        switch (te)
		{
          case 1:
		      hsv_general(1);
		      break;
          case 2:
	          hsv_succession(1);
	          break;
          case 3:
	          hsv_centerM(1);
	          break;
          default:
	      break;
		}
	for(i=0;i<3;i++)
	{
		for(int j=0;j<12;j++)
		{
		   temp[i]+=(tezheng[i][j]-pDoc->imagelib[d].te[i][j])*
		       (tezheng[i][j]-pDoc->imagelib[d].te[i][j]);
		}
	}
	}

	if(type==2)
	{
		switch (te)
		{
          case 1:
		      hsv_general(2);
		      break;
          case 2:
	          hsv_succession(2);
	          break;
          case 3:
	          hsv_centerM(2);
	          break;
          default:
	      break;
		}
    for(int j=0;j<8;j++)
	{
		temp[0]+=(tezheng[0][j]-pDoc->imagelib[d].te[0][j])*
		     (tezheng[0][j]-pDoc->imagelib[d].te[0][j]);
	}
	for(i=1;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
		   temp[i]+=(tezheng[i][j]-pDoc->imagelib[d].te[i][j])*
		       (tezheng[i][j]-pDoc->imagelib[d].te[i][j]);
		}
	}
	}

	if(type==3)
	{
		switch (te)
		{
          case 1:
		      hsv_general(3);
		      break;
          case 2:
	          hsv_succession(3);
	          break;
          case 3:
	          hsv_centerM(3);
	          break;
          default:
	      break;
		}
    for(i=0;i<3;i++)
	{
		for(int j=0;j<6;j++)
		{
		   temp[i]+=(tezheng[i][j]-pDoc->imagelib[d].te[i][j])*
		       (tezheng[i][j]-pDoc->imagelib[d].te[i][j]);
		}
	}
	}
}
if(model==2)
{
    if(type==1)
	{
        switch (te)
		{
          case 1:
		      mtm_general(1);
		      break;
          case 2:
	          mtm_succession(1);
	          break;
          case 3:
	          mtm_centerM(1);
	          break;
          default:
	      break;
		}

	for(i=0;i<3;i++)
	{
		for(int j=0;j<12;j++)
		{
		   temp[i]+=(tezheng[i][j]-pDoc->imagelib[d].te[i][j])*
		       (tezheng[i][j]-pDoc->imagelib[d].te[i][j]);
		}
	}
	}
}

	result = (float)(sqrt((double)(temp[0])) + sqrt((double)temp[1]) +
		             sqrt((double)temp[2]));
    return result;
}

float CRetrievallView::distance_weight(int model,int te,int d,int type)    //加权距离
{
	CRetrievalDoc* pDoc = GetDocument();
	float weight;
	float temp[3];
	float result;
	for(int i=0;i<3;i++)
        temp[i] = 0.0;

if(model==1)
{
	if(type==1)
	{
        switch (te)
		{
          case 1:
		      hsv_general(1);
		      break;
          case 2:
	          hsv_succession(1);
	          break;
          case 3:
	          hsv_centerM(1);
	          break;
          default:
	      break;
		}
	for( i=0;i<3;i++)
	{
		for(int j=0;j<12;j++)
		{
			if((tezheng[i][j]>0)&&(pDoc->imagelib[d].te[i][j]>0))
				weight = tezheng[i][j];
			else
			{
				if((tezheng[i][j]==0)||(pDoc->imagelib[d].te[i][j]==0))
					weight = 1;
			}
			temp[i] += weight * (tezheng[i][j]-pDoc->imagelib[d].te[i][j])
				       * (tezheng[i][j]-pDoc->imagelib[d].te[i][j]);
		}
		temp[i] = sqrtf(temp[i]);
	}
	}

	if(type==2)
	{
        switch (te)
		{
          case 1:
		      hsv_general(2);
		      break;
          case 2:
	          hsv_succession(2);
	          break;
          case 3:
	          hsv_centerM(2);
	          break;
          default:
	      break;
		}
    for(int j=0;j<8;j++)
		{
			if((tezheng[0][j]>0)&&(pDoc->imagelib[d].te[0][j]>0))
				weight = tezheng[0][j];
			else
			{
				if((tezheng[j]==0)||(pDoc->imagelib[d].te[0][j]==0))
					weight = 1;
			}
			temp[i] += weight * (tezheng[0][j]-pDoc->imagelib[d].te[0][j])
				       * (tezheng[0][j]-pDoc->imagelib[d].te[0][j]);
		}
		temp[0] = sqrtf(temp[0]);

	for( i=1;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			if((tezheng[i][j]>0)&&(pDoc->imagelib[d].te[i][j]>0))
				weight =tezheng[i][j];
			else
			{
				if((tezheng[i][j]==0)||(pDoc->imagelib[d].te[i][j]==0))
					weight = 1;
			}
			temp[i] += weight * (tezheng[i][j]-pDoc->imagelib[d].te[i][j])
				       * (tezheng[i][j]-pDoc->imagelib[d].te[i][j]);
		}
		temp[i] = sqrtf(temp[i]);
	}
	}

	if(type==3)
	{
		switch (te)
		{
          case 1:
		      hsv_general(3);
		      break;
          case 2:
	          hsv_succession(3);
	          break;
          case 3:
	          hsv_centerM(3);
	          break;
          default:
	      break;
		}
    for( i=0;i<3;i++)
	{
		for(int j=0;j<6;j++)
		{
			if((tezheng[i][j]>0)&&(pDoc->imagelib[d].te[i][j]>0))
				weight = tezheng[i][j];
			else
			{
				if((tezheng[i][j]==0)||(pDoc->imagelib[d].te[i][j]==0))
					weight = 1;
			}
			temp[i] += weight * (tezheng[i][j]-pDoc->imagelib[d].te[i][j])
				       * (tezheng[i][j]-pDoc->imagelib[d].te[i][j]);
		}
		temp[i] = sqrtf(temp[i]);
	}
	}
}
if(model==2)
{
	if(type==1)
	{
		switch (te)
		{
          case 1:
		      mtm_general(1);
		      break;
          case 2:
	          mtm_succession(1);
	          break;
          case 3:
	          mtm_centerM(1);
	          break;
          default:
	      break;
		}
	for( i=0;i<3;i++)
	{
		for(int j=0;j<12;j++)
		{
			if((tezheng[i][j]>0)&&(pDoc->imagelib[d].te[i][j]>0))
				weight = tezheng[i][j];
			else
			{
				if((tezheng[i][j]==0)||(pDoc->imagelib[d].te[i][j]==0))
					weight = 1;
			}
			temp[i] += weight * (tezheng[i][j]-pDoc->imagelib[d].te[i][j])
				       * (tezheng[i][j]-pDoc->imagelib[d].te[i][j]);
		}
		temp[i] = sqrtf(temp[i]);
	}
	}
}
        result = 15 * temp[0] + temp[1] +temp[2];
		return result;
}

void CRetrievallView::OnLButtonDown(UINT nFlags, CPoint point) 
{
	CRect rtemp[3];
    CRetrievalDoc* pDoc = GetDocument();
	CPoint ptemp=GetScrollPosition();
	for(int i=0;i<3;i++)
	{
		rtemp[i].bottom=pDoc->rect[i].bottom-ptemp.y;
		rtemp[i].top=pDoc->rect[i].top-ptemp.y;
		rtemp[i].left=pDoc->rect[i].left-ptemp.x;
		rtemp[i].right=pDoc->rect[i].right-ptemp.x;
	}
    
	for( i=0;i<3;i++)
	{
		if(rtemp[i].PtInRect(point))
		{
			m_exam = i;
			if_lbd = true;
			ifSelect=true;
			break;
		}
		ifSelect=false;
		if_lbd = false;
	}
	this->Invalidate(true);

	CScrollView::OnLButtonDown(nFlags, point);
}

void CRetrievallView::OnContextMenu(CWnd* pWnd, CPoint point) 
{
	// TODO: Add your message handler code here
	
	CMenu menu;
	menu.LoadMenu(IDR_MENU1);
	menu.GetSubMenu(0)
		->TrackPopupMenu(TPM_LEFTALIGN |TPM_RIGHTBUTTON,
		point.x,point.y,this);

}
void CRetrievallView::sortimage(int i)
{
	CRetrievalDoc* pDoc = GetDocument();
	if(i==1)
	{
		for(int a=0;a<40;a++)
		for(int b=a+1;b<40;b++)
			if(result_te[a]>result_te[b])
			{
			  float temp;
			  int itemp;
		      temp=result_te[a];
			  itemp=cixu[a];
			  result_te[a]=result_te[b];
			  cixu[a]=cixu[b];
			  result_te[b]=temp;
			  cixu[b]=itemp;
			}
        for(i=0;i<10;i++)
		{
			CJpeg jpeg;
	        jpeg.Load(pDoc->imagelib[cixu[i]].filename);
            HDIB hDIB = CopyHandle(jpeg.GetDib()->GetHandle());
            pDoc->m_pDibResu[i].Attach(hDIB);
		}
		ifSelect=false;
		pDoc->SetModifiedFlag();
		pDoc->UpdateAllViews(NULL);
	}
	else if(i==2)
	{
		for(int a=0;a<40;a++)
		for(int b=a+1;b<40;b++)
			if(result_te[a]<result_te[b])
			{
			  float temp;
			  int itemp;
		      temp=result_te[a];
			  itemp=cixu[a];
			  result_te[a]=result_te[b];
			  cixu[a]=cixu[b];
			  result_te[b]=temp;
			  cixu[b]=itemp;
			}
        for(i=0;i<10;i++)
		{
			CJpeg jpeg;
	        jpeg.Load(pDoc->imagelib[cixu[i]].filename);
            HDIB hDIB = CopyHandle(jpeg.GetDib()->GetHandle());
            pDoc->m_pDibResu[i].Attach(hDIB);
		}
		ifSelect=false;
		pDoc->SetModifiedFlag();
		pDoc->UpdateAllViews(NULL);
	}
}



void CRetrievallView::OnHSV_GEN_OU() 
{
	// TODO: Add your command handler code here
	
	CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(1);
	pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=distance(1,1,d,1);
	sortimage(1);
	
	}
		else
		AfxMessageBox("请选择需要查询的图像");

}


void CRetrievallView::OnHSV_GEN_QUAN() 
{
	// TODO: Add your command handler code here
   
    CRetrievalDoc* pDoc = GetDocument();
    if(pDoc->ifSelectlib&&ifSelect)
	{
	pDoc->load_telib(1);
    pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=distance_weight(1,1,d,1);
    sortimage(1);
	}
		else
		AfxMessageBox("请选择需要查询的图像");
	
}

void CRetrievallView::OnHSV_GEN_JIAO() 
{
	// TODO: Add your command handler code here
    
    CRetrievalDoc* pDoc = GetDocument();
    if(pDoc->ifSelectlib&&ifSelect)
	{
	pDoc->load_telib(1);
    pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=Caculate(1,1,d,1);
	sortimage(2);
	}
		else
		AfxMessageBox("请选择需要查询的图像");
}

void CRetrievallView::OnHSV_SUC_OU() 
{
	// TODO: Add your command handler code here
    
    CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(2);
    pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=distance(1,2,d,1);
	sortimage(1);
	}
		else
		AfxMessageBox("请选择需要查询的图像");
}

void CRetrievallView::OnHSV_CEN_OU() 
{
	// TODO: Add your command handler code here
  
    CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(3);
    pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=distance(1,3,d,1);
    sortimage(1);
	}
		else
		AfxMessageBox("请选择需要查询的图像");
	
}

void CRetrievallView::OnMTM_GEN_OU() 
{
	// TODO: Add your command handler code here
	CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(4);
	pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=distance(2,1,d,1);
    sortimage(1);
	}
		else
		AfxMessageBox("请选择需要查询的图像");

}

void CRetrievallView::OnMTM_GEN_QUAN() 
{
	// TODO: Add your command handler code here
    CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(4);
    pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=distance_weight(2,1,d,1);
    sortimage(1);
	}
		else
		AfxMessageBox("请选择需要查询的图像");
		
}

void CRetrievallView::OnMTM_GEN_JIAO() 
{
	// TODO: Add your command handler code here
	CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(4);
    pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=Caculate(2,1,d,1);
    sortimage(2);
	}
		else
		AfxMessageBox("请选择需要查询的图像");
	
}

void CRetrievallView::OnMTM_SUC_OU() 
{
	// TODO: Add your command handler code here
	CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(5);
    pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=distance(2,2,d,1);
	sortimage(1);
	}
		else
		AfxMessageBox("请选择需要查询的图像");
}

void CRetrievallView::OnMTM_CEN_OU() 
{
	// TODO: Add your command handler code here
	CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(6);
    pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=distance(2,3,d,1);
    sortimage(1);
	}
		else
		AfxMessageBox("请选择需要查询的图像");
	
}

void CRetrievallView::OnHSV1_2() 
{
	// TODO: Add your command handler code here
	CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(7);
	pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=distance(1,1,d,2);
    sortimage(1);
	}
		else
		AfxMessageBox("请选择需要查询的图像");

}

void CRetrievallView::OnHSV2_2() 
{
	// TODO: Add your command handler code here
	CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(7);
	pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=distance_weight(1,1,d,2);
    sortimage(1);
	}
		else
		AfxMessageBox("请选择需要查询的图像");

}

void CRetrievallView::OnHSV3_2() 
{
	// TODO: Add your command handler code here
	CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(7);
	pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=Caculate(1,1,d,2);
    sortimage(2);
	}
		else
		AfxMessageBox("请选择需要查询的图像");

}

void CRetrievallView::OnHSV4_2() 
{
	// TODO: Add your command handler code here
	CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(8);
	pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=distance(1,2,d,2);
	sortimage(1);
	}	
	else
		AfxMessageBox("请选择需要查询的图像");

}

void CRetrievallView::OnHSV4_3() 
{
	// TODO: Add your command handler code here
	CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(9);
	pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=distance(1,2,d,3);
	sortimage(1);
	}
	else
		AfxMessageBox("请选择需要查询的图像");

}

void CRetrievallView::OnHSV5_2() 
{
	// TODO: Add your command handler code here
	CRetrievalDoc* pDoc = GetDocument();
	if(pDoc->ifSelectlib&&ifSelect)
	{
    pDoc->load_telib(10);
	pDoc->onresult=true;

	for(int i=0;i<40;i++)
		cixu[i]=i;
	for(int d=0;d<pDoc->libnumber;d++)
		result_te[d]=distance(1,3,d,2);
    sortimage(1);
	}
	else
		AfxMessageBox("请选择需要查询的图像");
}

void CRetrievallView::OnRotate() 
{
	// TODO: Add your command handler code here
	if(ifSelect)
	{
		CRetrievalDoc* pDoc = GetDocument();
        pDoc->m_Exam[m_exam].Rotate180();
		ifSelect=false;
       	pDoc->SetModifiedFlag();
		pDoc->UpdateAllViews(NULL);
	}
	else
		AfxMessageBox("请选择需要旋转的图像");
}

void CRetrievallView::OnBig() 
{
	// TODO: Add your command handler code here
	if(ifSelect)
	{
		CRetrievalDoc* pDoc = GetDocument();

		pDoc->m_Exam[m_exam].ChangeImageSize(pDoc->m_Exam[m_exam].GetWidth()+10,
			                                  pDoc->m_Exam[m_exam].GetHeight()+10);
		pDoc->rect[m_exam].bottom+=10;
	    pDoc->rect[m_exam].right+=10;
	 
		for(int i=m_exam;i<3;i++)
		{
			pDoc->rect[i+1].left=pDoc->rect[i].right+72;
			pDoc->rect[i+1].right+=10;
		}
	    ifSelect=false;
		pDoc->SetModifiedFlag();
		pDoc->UpdateAllViews(NULL);
	}
	else
		AfxMessageBox("请选择需要放大的图像");

}


void CRetrievallView::OnSmall() 
{
	// TODO: Add your command handler code here
		if(ifSelect)
		{
		CRetrievalDoc* pDoc = GetDocument();
			pDoc->m_Exam[m_exam].ChangeImageSize(pDoc->m_Exam[m_exam].GetWidth()-10,
			                                  pDoc->m_Exam[m_exam].GetHeight()-10);

		pDoc->rect[m_exam].bottom-=10;
	    pDoc->rect[m_exam].right-=10;
		for(int i=m_exam;i<3;i++)
			{
			pDoc->rect[i+1].left=pDoc->rect[i].right+72;
			pDoc->rect[i+1].right-=10;
		}
	    ifSelect=false;
		pDoc->SetModifiedFlag();
		pDoc->UpdateAllViews(NULL);
	}
	else
		AfxMessageBox("请选择需要缩小的图像");


}

void CRetrievallView::OnMove() 
{
	// TODO: Add your command handler code here
	if(ifSelect)
		{
	
		CRetrievalDoc* pDoc = GetDocument();

		pDoc->rect[m_exam].bottom+=20;
	    pDoc->rect[m_exam].right+=20;
		 pDoc->rect[m_exam].top+=20;
		  pDoc->rect[m_exam].left+=20;
		for(int i=m_exam;i<3;i++)
			{
			pDoc->rect[i+1].left=pDoc->rect[i].right+72;
			pDoc->rect[i+1].right+=20;
		}
	    ifSelect=false;
		pDoc->SetModifiedFlag();
		pDoc->UpdateAllViews(NULL);
	}
	else
		AfxMessageBox("请选择需要移动的图像");


	
}
