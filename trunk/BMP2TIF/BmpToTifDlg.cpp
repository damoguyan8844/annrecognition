// BmpToTifDlg.cpp : implementation file
//

#include "stdafx.h"
#include "BmpToTif.h"
#include "BmpToTifDlg.h"
#include "tiffio.h"
#include "tiffiop.h"
#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CBmpToTifDlg dialog

CBmpToTifDlg::CBmpToTifDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CBmpToTifDlg::IDD, pParent)
{
	//{{AFX_DATA_INIT(CBmpToTifDlg)
		// NOTE: the ClassWizard will add member initialization here
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CBmpToTifDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CBmpToTifDlg)
		// NOTE: the ClassWizard will add DDX and DDV calls here
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CBmpToTifDlg, CDialog)
	//{{AFX_MSG_MAP(CBmpToTifDlg)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, OnConvert)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CBmpToTifDlg message handlers

BOOL CBmpToTifDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon
	
	// TODO: Add extra initialization here
	
	return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CBmpToTifDlg::OnPaint() 
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, (WPARAM) dc.GetSafeHdc(), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// The system calls this to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CBmpToTifDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

void CBmpToTifDlg::OnConvert() 
{
	// this is your duty to fill value according to u...
	// any problem? let me know...I may help u...
	// w w w . p e c i n t . c o m  (remove space & place in internet & get my con tact)
	// sumit(under-score)kapoor1980(at)hot mail(dot) com
	// sumit (under-score) kapoor1980(at)ya hoo(dot) com
	// sumit (under-score) kapoor1980(at)red iff mail(dot) com
	
	HBITMAP hImage = (HBITMAP)LoadImage(NULL, "C:\\Sample.bmp", IMAGE_BITMAP,
	0, 0, LR_LOADFROMFILE|LR_CREATEDIBSECTION|LR_DEFAULTSIZE);

	CBitmap* m_Bitmap = CBitmap::FromHandle(hImage);

	// Sumit: memory allocation is still 1800x1800 in your code..
	BYTE* bmpBuffer=(BYTE*)GlobalAlloc(GPTR, 600*600);//allocate memory


	// Size of bitmap as I draw by using x,y points...
	m_Bitmap->GetBitmapBits(600*600 ,bmpBuffer);


	TIFF *image;

	// Open the TIFF file
	if((image = TIFFOpen("C:\\output.tif", "w")) == NULL)
	{
		printf("Could not open output.tif for writing\n");
	}

	TIFFSetField(image, TIFFTAG_IMAGEWIDTH,600);
	TIFFSetField(image, TIFFTAG_IMAGELENGTH,600);
	TIFFSetField(image, TIFFTAG_BITSPERSAMPLE,8);
	TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL,1);

	uint32 rowsperstrip = TIFFDefaultStripSize(image, -1); 
	//<REC> gives better compression

	TIFFSetField(image, TIFFTAG_ROWSPERSTRIP, rowsperstrip);
  
	//  TIFFSetField(image, TIFFTAG_COMPRESSION, COMPRESSION_CCITTFAX3);
	TIFFSetField(image, TIFFTAG_COMPRESSION, COMPRESSION_PACKBITS); 
	// TIFFSetField(image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);

	// Start CCITTFAX3 setting
  
	uint32 group3options = GROUP3OPT_FILLBITS+GROUP3OPT_2DENCODING;
	TIFFSetField(image, TIFFTAG_GROUP3OPTIONS, group3options);
	TIFFSetField(image, TIFFTAG_FAXMODE, FAXMODE_CLASSF);
	TIFFSetField(image, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
	TIFFSetField(image, TIFFTAG_ROWSPERSTRIP, -1L);


	// End CCITTFAX3 setting

	//if we comment following line then Tiff will not view in Imaging 
	//but view in DC
	TIFFSetField(image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

	TIFFSetField(image, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);
	TIFFSetField(image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

	TIFFSetField(image, TIFFTAG_RESOLUTIONUNIT, RESUNIT_INCH);
	TIFFSetField(image, TIFFTAG_XRESOLUTION, 100.0);
	TIFFSetField(image, TIFFTAG_YRESOLUTION, 100.0);


	char page_number[20];
	sprintf(page_number, "Page %d", 1);

	TIFFSetField(image, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
	TIFFSetField(image, TIFFTAG_PAGENUMBER, 1, 1);
	TIFFSetField(image, TIFFTAG_PAGENAME, page_number);

	// Write the information to the file
	BYTE *bits;
	for (int y = 0; y < 600; y++)
	{
		bits= bmpBuffer + y*600;
		if (TIFFWriteScanline(image,bits, y, 0)==-1) MessageBox("Complete or error");
	}

	// Close the file
	TIFFClose(image);
	
}
