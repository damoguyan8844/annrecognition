// CCMD_OCRView.cpp : implementation of the CCCMD_OCRView class
//
#include "stdafx.h"
#include "CCMD_OCR.h"
#include "CCMD_OCRDoc.h"
#include "CCMD_OCRView.h"
#include "../class/global.h"//
#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif
#include "mdivwctl.h"
enum MiLANGUAGES 
{ miLANG_CHINESE_SIMPLIFIED = 2052,
  miLANG_CHINESE_TRADITIONAL = 1028,
  miLANG_CZECH = 5,
  miLANG_DANISH = 6, 
  miLANG_DUTCH = 19,
  miLANG_ENGLISH = 9,
  miLANG_FINNISH = 11,
  miLANG_FRENCH = 12,
  miLANG_GERMAN = 7,
  miLANG_GREEK = 8, 
  miLANG_HUNGARIAN = 14,
  miLANG_ITALIAN = 16,
  miLANG_JAPANESE = 17,
  miLANG_KOREAN = 18,
  miLANG_NORWEGIAN = 20,
  miLANG_POLISH = 21,
  miLANG_PORTUGUESE = 22,
  miLANG_RUSSIAN = 25,
  miLANG_SPANISH = 10,
  miLANG_SWEDISH = 29,
  miLANG_SYSDEFAULT = 2048,
  miLANG_TURKISH = 31
};
enum MiFILE_FORMAT
{ miFILE_FORMAT_DEFAULTVALUE = -1,
  miFILE_FORMAT_TIFF = 1,
  miFILE_FORMAT_TIFF_LOSSLESS = 2, 
  miFILE_FORMAT_MDI = 4
};
enum MiCOMP_LEVEL 
{ miCOMP_LEVEL_LOW = 0,
  miCOMP_LEVEL_MEDIUM = 1, 
  miCOMP_LEVEL_HIGH = 2
};
/////////////////////////////////////////////////////////////////////////////
// CCCMD_OCRView

IMPLEMENT_DYNCREATE(CCCMD_OCRView, CFormView)

BEGIN_MESSAGE_MAP(CCCMD_OCRView, CFormView)
	//{{AFX_MSG_MAP(CCCMD_OCRView)
	ON_WM_PAINT()
	ON_COMMAND(ID_FILE_OPEN, OnFileOpen)
	ON_LBN_SELCHANGE(IDC_LIST1, OnSelchangeList1)
	ON_COMMAND(ID_FILE_NEW, OnFileNew)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CCCMD_OCRView construction/destruction

CCCMD_OCRView::CCCMD_OCRView()
	: CFormView(CCCMD_OCRView::IDD)
{
	//{{AFX_DATA_INIT(CCCMD_OCRView)
	//}}AFX_DATA_INIT
}

CCCMD_OCRView::~CCCMD_OCRView()
{
}

void CCCMD_OCRView::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CCCMD_OCRView)
	DDX_Control(pDX, IDC_LIST1, m_list);
	//}}AFX_DATA_MAP
}

BOOL CCCMD_OCRView::PreCreateWindow(CREATESTRUCT& cs)
{
	return CFormView::PreCreateWindow(cs);
}
/////////////////////////////////////////////////////////////////////////////
// CCCMD_OCRView diagnostics

#ifdef _DEBUG
void CCCMD_OCRView::AssertValid() const
{
	CFormView::AssertValid();
}

void CCCMD_OCRView::Dump(CDumpContext& dc) const
{
	CFormView::Dump(dc);
}

CCCMD_OCRDoc* CCCMD_OCRView::GetDocument() // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CCCMD_OCRDoc)));
	return (CCCMD_OCRDoc*)m_pDocument;
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CCCMD_OCRView message handlers
void CCCMD_OCRView::OnInitialUpdate()
{	CFormView::OnInitialUpdate();
	GetParentFrame()->RecalcLayout();
	ResizeParentToFit();
	g_hWndview = GetSafeHwnd();
	hDC  = ::GetDC(g_hWndview);
	hScrDC= CreateDC("DISPLAY", NULL, NULL, NULL);
}
BOOL CCCMD_OCRView::OCRImageFile( CString Name)//OCR
{ IDocument *pDoc = new IDocument;
  pDoc->CreateDispatch( "MODI.Document" );
  pDoc->Create(Name);
  pDoc->OCR( miLANG_CHINESE_SIMPLIFIED, 0, 0 );
  IImages images = pDoc->GetImages();
  long	  num =images.GetCount();
  for( int i = 0; i < num; i++ )
  { IImage  image = images.GetItem(i);
    ILayout layout = image.GetLayout();
	SetDlgItemText(IDC_EDIT1, layout.GetText());
  }
  pDoc->Close(0);
  pDoc->ReleaseDispatch();
  delete pDoc;
  return (num > 0) ? TRUE : FALSE;
}
void CCCMD_OCRView::OnPaint() 
{	GetWindowRect(rect);
	CPaintDC dc(this); // device context for painting
	GetDlgItem(IDC_LIST1)->MoveWindow( 0,20,80,rect.Height()-25,TRUE);
	GetDlgItem(IDC_EDIT1)->MoveWindow( 82,20,(rect.Width()-90)/2,rect.Height()-25,TRUE);
}
void CCCMD_OCRView::OnFileOpen() //打开目录
{	g_fSelectFolderDlg(&FileDir,FileDir,false);
	FindFile(FileDir);//查找文件	
}
void CCCMD_OCRView::FindFile(CString DirName)//查找文件
{	WIN32_FIND_DATA	FindFileData;
	HANDLE hFindFile;
	SetCurrentDirectory(DirName);
	hFindFile=FindFirstFile("*.*",&FindFileData);
	m_list.ResetContent( );
	CString tFile;
	if (hFindFile!=INVALID_HANDLE_VALUE)
	{	do {tFile=FindFileData.cFileName;
			if ((tFile==".")||(tFile=="..")) continue;
			if (FindFileData.dwFileAttributes!=FILE_ATTRIBUTE_DIRECTORY)
				if(tFile.Right(3)=="jpg"||tFile.Right(3)=="JPG"||
				   tFile.Right(3)=="TIF"||tFile.Right(3)=="tif"||
				   tFile.Right(3)=="BMP"||tFile.Right(3)=="bmp")
					{m_list.AddString(tFile);
					}
			}
		while (FindNextFile(hFindFile,&FindFileData));
	}
	FindClose(hFindFile);
	pBuffer = new BYTE [2];
}
void CCCMD_OCRView::OnFileNew() //OCR
{	SetCurrentDirectory(FileDir);
	OCRImageFile(cn);
}
void CCCMD_OCRView::OnSelchangeList1() 
{	int p=m_list.GetCurSel();
	m_list.GetText(p,cn);
	SetCurrentDirectory(FileDir);
	if(cn.Right(3)=="tif"||cn.Right(3)=="TIF") type=CXIMAGE_FORMAT_TIF;
	if(cn.Right(3)=="jpg"||cn.Right(3)=="JPG") type=CXIMAGE_FORMAT_JPG;
	if(cn.Right(3)=="bmp"||cn.Right(3)=="BMP") type=CXIMAGE_FORMAT_BMP;
	ShowJpg(cn);		//显示图形
	SetDlgItemText(IDC_EDIT1,"");
}
void CCCMD_OCRView::ShowJpg(CString Name)//显示图形
{	image = new CxImage();//
	cn.Format("%s\\%s",FileDir,Name);
	int lan=ReadJpg(cn);				//读图形文件
	image->Decode(pBuffer, lan, type);	//转成位图
	hBitW=(float)image->GetWidth();		//图形大小
	hBitH=(float)image->GetHeight();	//图形大小
	HDC     hMDC = CreateCompatibleDC(hScrDC);
	HBITMAP hBit = CreateCompatibleBitmap(hScrDC,(int)hBitW,(int)hBitH);
	HBITMAP hold =(HBITMAP)SelectObject(hMDC,hBit);
	image->Draw2(hMDC,0,0,(int)hBitW,(int)hBitH);
	SetStretchBltMode(hDC,STRETCH_HALFTONE);
	StretchBlt(hDC,82+(rect.Width()-90)/2,20,(rect.Width()-90)/2,rect.Height()-25,hMDC,0,0,(int)hBitW,(int)hBitH,SRCCOPY);//g_hDCS
	SelectObject(hMDC,hold);
	DeleteObject(hBit);
	DeleteDC(hMDC);
	image->Clear(0);					//
	image->Destroy();					//
}
int CCCMD_OCRView::ReadJpg(CString Name)//读图形文件
{	CFile fi;
	CFileException e;
	int nSize;
	delete [] pBuffer;				//删掉内存
	if(fi.Open(Name,CFile::modeRead|CFile::typeBinary,&e))//打开了一个图形文件
	  {nSize = fi.GetLength();      //先得到图形文件长度
	   pBuffer = new BYTE [nSize+2];//按文件的大小申请一块内存
	   fi.Read(pBuffer, nSize);		//把图形文件读到pBuffer
	   fi.Close();
	  }
	return nSize;					//返回图形的长度。
}


