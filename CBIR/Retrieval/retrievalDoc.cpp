// retrievalDoc.cpp : implementation of the CRetrievalDoc class
//

#include "stdafx.h"
#include "retrieval.h"

#include "retrievalDoc.h"
#include "iostream.h"
#include "fstream.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CRetrievalDoc

IMPLEMENT_DYNCREATE(CRetrievalDoc, CDocument)

BEGIN_MESSAGE_MAP(CRetrievalDoc, CDocument)
	//{{AFX_MSG_MAP(CRetrievalDoc)
	ON_COMMAND(ID_WINTER, OnWinter)
	ON_COMMAND(ID_FLAG, OnFlag)
	ON_COMMAND(ID_FLOWER, OnFlower)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CRetrievalDoc construction/destruction

CRetrievalDoc::CRetrievalDoc()
{
	// TODO: add one-time construction code here
    for(int i=0;i<3;i++)
		onoff[i]=false;
	onresult=false;
	ifSelectlib=false;

}

CRetrievalDoc::~CRetrievalDoc()
{


}

BOOL CRetrievalDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}



/////////////////////////////////////////////////////////////////////////////
// CRetrievalDoc serialization

void CRetrievalDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
	}
	else
	{
		// TODO: add loading code here
	}
}

/////////////////////////////////////////////////////////////////////////////
// CRetrievalDoc diagnostics

#ifdef _DEBUG
void CRetrievalDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CRetrievalDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CRetrievalDoc commands

void CRetrievalDoc::OnWinter() 
{
	onoff[0]=true;
	onoff[1]=false;
	onoff[2]=false;
	onresult=false;
	// TODO: Add your command handler code here
   		ifSelectlib=true;
	rect[0].left=140; rect[0].top=70; rect[0].right=268; rect[0].bottom=166;
	rect[1].left=340; rect[1].top=70; rect[1].right=468; rect[1].bottom=166;
	rect[2].left=540; rect[2].top=70; rect[2].right=668; rect[2].bottom=166;
	
	m_filenameExam[0]="pic\\winter\\Sr048.jpg";
	m_filenameExam[1]="pic\\winter\\Sr118.jpg";
    m_filenameExam[2]="pic\\winter\\Sr127.jpg";
	m_graphExam[0] = 5;
    m_graphExam[1] = 20;
	m_graphExam[2] = 24;


    for(int i=0;i<3;i++)
	{
	  CJpeg jpeg;
	  jpeg.Load(m_filenameExam[i].GetBuffer(40));
      HDIB hDIB = CopyHandle(jpeg.GetDib()->GetHandle());
      m_Exam[i].Attach(hDIB);
	}
	UpdateAllViews(NULL);

}

void CRetrievalDoc::OnFlag() 
{
	onoff[0]=false;
	onoff[1]=true;
	onoff[2]=false;
	onresult=false;
	// TODO: Add your command handler code here
	 rect[0].left=140; rect[0].top=70; rect[0].right=268; rect[0].bottom=166;
	rect[1].left=340; rect[1].top=70; rect[1].right=468; rect[1].bottom=166;
	rect[2].left=540; rect[2].top=70; rect[2].right=668; rect[2].bottom=166;
	
		ifSelectlib=true;
	m_filenameExam[0]="pic\\flag\\beinin.jpg";
	m_filenameExam[1]="pic\\flag\\ydl.jpg";
    m_filenameExam[2]="pic\\flag\\xila.jpg";
    m_graphExam[0] = 8;
	m_graphExam[1] = 33;
	m_graphExam[2] = 31;

    for(int i=0;i<3;i++)
	{
	  CJpeg jpeg;
	  jpeg.Load(m_filenameExam[i].GetBuffer(40));
      HDIB hDIB = CopyHandle(jpeg.GetDib()->GetHandle());
      m_Exam[i].Attach(hDIB);
	}
	UpdateAllViews(NULL);
	
}

void CRetrievalDoc::OnFlower() 
{
	// TODO: Add your command handler code here
	onoff[0]=false;
	onoff[1]=false;
	onoff[2]=true;
	onresult=false;
	rect[0].left=140; rect[0].top=70; rect[0].right=268; rect[0].bottom=166;
	rect[1].left=340; rect[1].top=70; rect[1].right=468; rect[1].bottom=166;
	rect[2].left=540; rect[2].top=70; rect[2].right=668; rect[2].bottom=166;
	// TODO: Add your command handler code here

		ifSelectlib=true;
	m_filenameExam[0]="pic\\flower\\E1454.jpg";
	m_filenameExam[1]="pic\\flower\\ys067.jpg";
    m_filenameExam[2]="pic\\flower\\tn_0026.jpg";
    m_graphExam[0] = 1;
	m_graphExam[1] = 36;
	m_graphExam[2] = 25;


    for(int i=0;i<3;i++)
	{
	  CJpeg jpeg;
	  jpeg.Load(m_filenameExam[i].GetBuffer(40));
      HDIB hDIB = CopyHandle(jpeg.GetDib()->GetHandle());
      m_Exam[i].Attach(hDIB);
	}
	UpdateAllViews(NULL);
}

void CRetrievalDoc::load_telib(int select)
{ 
	int i;
	ifstream fin1;
	if(onoff[0])
	{
	switch (select)
	{
	case 1:
	   fin1.open(_T("telib_hsv_general_winter12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<12;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	case 2:
	   fin1.open(_T("telib_hsv_succession_winter12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<12;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	case 3:
	   fin1.open(_T("telib_hsv_centerM_winter12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;
		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
			   for(int s=3;s<12;s++)
				   imagelib[i].te[j][s] = 0.0;
		   }
	   }
	   fin1.close();
	   break;

       case 4:
	   fin1.open(_T("telib_mtm_general_winter12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<12;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	case 5:
	   fin1.open(_T("telib_mtm_succession_winter12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<12;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	case 6:
	   fin1.open(_T("telib_mtm_centerM_winter12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;
		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
			   for(int s=3;s<12;s++)
				   imagelib[i].te[j][s] = 0.0;
		   }
	   }
	   fin1.close();
	   break;

    case 7:
	   fin1.open(_T("telib_hsv_general_winter833.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int k=0;k<8;k++)
			   fin1>>imagelib[i].te[0][k];

		   for(int j=1;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

       case 8:
	   fin1.open(_T("telib_hsv_succession_winter833.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		  for(int k=0;k<8;k++)
			   fin1>>imagelib[i].te[0][k];

		   for(int j=1;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

       case 9:
	   fin1.open(_T("telib_hsv_succession_winter6.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<6;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

       case 10:
	   fin1.open(_T("telib_hsv_center_winter833.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		  for(int k=0;k<8;k++)
			   fin1>>imagelib[i].te[0][k];

		   for(int j=1;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	 default:
	   break;
     }
	 }

	if(onoff[1])
	{
	switch (select)
	{
	case 1:
	   fin1.open(_T("telib_hsv_general_flag12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<12;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	case 2:
	   fin1.open(_T("telib_hsv_succession_flag12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<12;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	
	case 3:
	   fin1.open(_T("telib_hsv_centerM_flag12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;
		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
			   for(int s=3;s<12;s++)
				   imagelib[i].te[j][s] = 0.0;
		   }
	   }
	   fin1.close();
	   break;

       case 4:
	   fin1.open(_T("telib_mtm_general_flag12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<12;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	case 5:
	   fin1.open(_T("telib_mtm_succession_flag12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<12;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	
	case 6:
	   fin1.open(_T("telib_mtm_centerM_flag12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;
		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
			   for(int s=3;s<12;s++)
				   imagelib[i].te[j][s] = 0.0;
		   }
	   }
	   fin1.close();
	   break;

	   case 7:
	   fin1.open(_T("telib_hsv_general_flag833.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int k=0;k<8;k++)
			   fin1>>imagelib[i].te[0][k];

		   for(int j=1;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

       case 8:
	   fin1.open(_T("telib_hsv_succession_flag833.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		  for(int k=0;k<8;k++)
			   fin1>>imagelib[i].te[0][k];

		   for(int j=1;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

       case 9:
	   fin1.open(_T("telib_hsv_succession_flag6.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<6;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

       case 10:
	   fin1.open(_T("telib_hsv_center_flag833.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		  for(int k=0;k<8;k++)
			   fin1>>imagelib[i].te[0][k];

		   for(int j=1;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	 default:
	   break;
     }
	 }

    if(onoff[2])
	{
	switch (select)
	{
	case 1:
	   fin1.open(_T("telib_hsv_general_flower12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;
		   
		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<12;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	case 2:
	   fin1.open(_T("telib_hsv_succession_flower12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<12;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	case 3:
	   fin1.open(_T("telib_hsv_centerM_flower12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;
		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
			   for(int s=3;s<12;s++)
				   imagelib[i].te[j][s] = 0.0;
		   }
	   }
	   fin1.close();
	   break;

       case 4:
	   fin1.open(_T("telib_mtm_general_flower12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;
		   
		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<12;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	case 5:
	   fin1.open(_T("telib_mtm_succession_flower12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<12;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	case 6:
	   fin1.open(_T("telib_mtm_centerM_flower12.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;
		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
			   for(int s=3;s<12;s++)
				   imagelib[i].te[j][s] = 0.0;
		   }
	   }
	   fin1.close();
	   break;

	   case 7:
	   fin1.open(_T("telib_hsv_general_flower833.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int k=0;k<8;k++)
			   fin1>>imagelib[i].te[0][k];

		   for(int j=1;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	   case 8:
	   fin1.open(_T("telib_hsv_succession_flower833.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		  for(int k=0;k<8;k++)
			   fin1>>imagelib[i].te[0][k];

		   for(int j=1;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	   case 9:
	   fin1.open(_T("telib_hsv_succession_flower6.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		   for(int j=0;j<3;j++)
		   {
			   for(int k=0;k<6;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;

	   case 10:
	   fin1.open(_T("telib_hsv_center_flower833.txt"));      
	   fin1>>libnumber;

	   for(i=0;i<libnumber;i++)
	   {
		   fin1>>imagelib[i].filename;

		  for(int k=0;k<8;k++)
			   fin1>>imagelib[i].te[0][k];

		   for(int j=1;j<3;j++)
		   {
			   for(int k=0;k<3;k++)
				   fin1>>imagelib[i].te[j][k];
		   }
	   }
	   fin1.close();
	   break;
	   default:
	   break;
     }
	 }
}

