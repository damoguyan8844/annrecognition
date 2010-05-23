// MyCBIRDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "MyCBIR.h"
#include "MyCBIRDlg.h"
#include "LBP.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define MAXINT 2147483647
// CMyCBIRDlg 对话框




CMyCBIRDlg::CMyCBIRDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CMyCBIRDlg::IDD, pParent)
	, m_bTextureBased(TRUE)
	, m_nLBPMethod(0)
	, m_bColorBased(TRUE)
	, m_nColorMethod(0)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_pTargetPic = NULL;
	for(int i=0;i<9;i++) m_pLibPic[i]=NULL;
	for(int i=0;i<200;i++) m_pLBPBuf[i] = m_pColorBuf[i] = NULL;
	m_bLibLoaded = FALSE;
	m_bTargetLoaded = FALSE;
	m_iPicNum = 0;
	m_nLibType = 0;
	m_bSearched = FALSE;
	for(int i=0;i<9;i++) index[i]=i;
}

void CMyCBIRDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Check(pDX, IDC_TEXTURE_BASED, m_bTextureBased);
	DDX_CBIndex(pDX, IDC_COMBO_TEXTURE, m_nLBPMethod);
	DDX_Check(pDX, IDC_COLOR_BASED, m_bColorBased);
	DDX_CBIndex(pDX, IDC_COMBO_COLOR, m_nColorMethod);
}

BEGIN_MESSAGE_MAP(CMyCBIRDlg, CDialog)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(ID_OPEN_PIC, &CMyCBIRDlg::OnBnClickedOpenPic)
	ON_BN_CLICKED(ID_LOAD_LIB, &CMyCBIRDlg::OnBnClickedLoadLib)
	ON_BN_CLICKED(ID_SEARCH, &CMyCBIRDlg::OnBnClickedSearch)
END_MESSAGE_MAP()


// CMyCBIRDlg 消息处理程序

BOOL CMyCBIRDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// 设置此对话框的图标。当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CMyCBIRDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文
		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.m_hDC), 0);

		// 使图标在工作矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CPaintDC dc(this); // 用于绘制的设备上下文
		CDialog::OnPaint();
		Graphics graphics(dc.m_hDC);
		CString  strPrint;
		
		if(m_bLibLoaded)
		{
			for(int i=0;i<9;i++)
			{				
				int nW = m_pLibPic[i]->GetWidth();
				int nH = m_pLibPic[i]->GetHeight();
				Rect rect(340+(int)(i%3)*240-nW/4,120+(int)(i/3)*240-nH/4,nW/2,nH/2);
				graphics.DrawImage(m_pLibPic[i],rect);
				if(m_bSearched)
				{
					strPrint.Format(L"%d:%d\n",i,result[i]);
					CString filename = fileSet.GetAt( index[i] );
					int l = 0, s;
					while( l != -1)
					{
						s = l;
						l = filename.Find('\\',s+1);
					}
					filename = filename.Right( filename.GetLength() - s -1);
					strPrint += filename;

					graphics.DrawString(strPrint, -1, &Font(L"宋体",9,FontStyleRegular),PointF(320+(int)(i%3)*240,125+(int)(i/3)*240+nH/4),&SolidBrush(Color::Black));
				}
			}
			if(m_bSearched)
			{
				strPrint = L"搜索结果按欧氏距离从小到大排列";
				graphics.DrawString(strPrint, -1, &Font(L"宋体",9,FontStyleRegular),PointF(20,300),&SolidBrush(Color::Black));
			}
		}
		if(m_bTargetLoaded)
		{
			int nW = m_pTargetPic->GetWidth();
			int nH = m_pTargetPic->GetHeight();
			Rect rect(120-nW/4,120-nH/4,nW/2,nH/2);
			graphics.DrawImage(m_pTargetPic,rect);
			strPrint = L"目标图片";
			graphics.DrawString(strPrint, -1, &Font(L"宋体",9,FontStyleRegular),PointF(90,125+nH/4),&SolidBrush(Color::Black));
		}
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标显示。
//
HCURSOR CMyCBIRDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CMyCBIRDlg::OnBnClickedOpenPic()
{
	// TODO: 在此添加控件通知处理程序代码
	CFileDialog dlg(TRUE,NULL,L"支持各种图片",OFN_HIDEREADONLY|OFN_OVERWRITEPROMPT,
	   L"JPEG(*.JPG,*.JPE),BMP(*.BMP,*.RLE),TIFF(*.TIF)|*.BMP;*.RLE;*.JPG;JPE;*.TIF;*.TIFF|BMP(*.BMP,*.RLE)|*.BMP;*.RLE|JPEG(*.JPG,*.JPE)|*.JPG;JPE|TIFF(*.TIF)|*.TIF;*.TIFF|*.*|");
	int result = dlg.DoModal();
	if( IDOK == result )
	{
		CString filename = dlg.GetFileName();
		if(m_pTargetPic != NULL)	delete m_pTargetPic;
		m_pTargetPic =new Bitmap(filename);
		m_bTargetLoaded = true;
		m_bSearched = FALSE;
	}

	if(m_bLibLoaded && m_bTargetLoaded)
		(CButton*)GetDlgItem(ID_SEARCH)->EnableWindow();

	Invalidate();
}

void CMyCBIRDlg::OnBnClickedLoadLib()
{
	// TODO: 在此添加控件通知处理程序代码
	// 打开文件夹
	BROWSEINFO *info=new BROWSEINFO; 
	ITEMIDLIST* _list; 
	WCHAR path[100];
	ZeroMemory(info,sizeof(BROWSEINFO)); 
	info->hwndOwner=GetSafeHwnd(); 
	info->iImage=NULL; 
	info->lpszTitle=L"请选择你的文件夹"; 
	info->pidlRoot=NULL; 
	info->lpfn=NULL; 
	info->ulFlags=BIF_EDITBOX|BIF_RETURNONLYFSDIRS ; 
	_list=SHBrowseForFolder(info); 
	if (_list) 
	{ 
		SHGetPathFromIDList(_list,path); 
	} 
	delete info;
	CFileFind find; 
	CString filename; 
	wcscat(path,L"\\*.*");
	BOOL isfind=find.FindFile(path);
	if(!isfind)  
		return;

	while (isfind) 
	{ 
		isfind=find.FindNextFile(); 
		if (find.IsDots() || find.IsHidden()) 
		{ 
			continue; 
		} 

		filename=find.GetFilePath();
		fileSet.Add(filename);
		m_iPicNum++;
	}

	UpdateData();
	CreateLib();

	m_bLibLoaded = true;
	if(m_bLibLoaded && m_bTargetLoaded)
		(CButton*)GetDlgItem(ID_SEARCH)->EnableWindow();

	//  更新右侧显示图片
	UpdateShowPics();
	Invalidate();
}

void CMyCBIRDlg::OnBnClickedSearch()
{
	// TODO: 在此添加控件通知处理程序代码
	
	UpdateData();
	if(!m_bTextureBased && !m_bColorBased)
		return;
	// 检查是否要更新特征库
	CreateLib();

	int *pLBP = new int [256];
	int *pColor = new int [256];
	if(m_bTextureBased)   // 计算目标图片的LBP
		CaculateLBP(m_pTargetPic,pLBP);	
	if(m_bColorBased)     // 计算目标图片的Color直方图
		CaculateColor(m_pTargetPic,pColor);
	// 与图片库的特征集合做比较
	for(int i=0;i<9;i++) result[i] = MAXINT;
	for(int i=0;i<m_iPicNum;i++)
	{
		int diffLBP=0;
		for(int j=0;j<256;j++)
		{
			int d=0,f=0;
			if(m_bTextureBased)
			{
				d = ( pLBP[j]-m_pLBPBuf[i][j] ) /10;
				diffLBP+=d*d;
			}
			if(m_bColorBased)
			{
				f = ( pColor[j]-m_pColorBuf[i][j] ) / 100;
				diffLBP+=f*f;
			}
		}

		for(int k=0;k<9;k++)
		{
			if(diffLBP < result[k])
			{
				for(int p=8;p>k;p--) 
				{
					result[p]=result[p-1];
					index[p]=index[p-1];
				}
				result[k] = diffLBP;
				index[k] = i;
				break;
			}
		}
	}
	delete [] pLBP;
	delete [] pColor;

	// 显示搜索结果前9张图片
	m_bSearched = TRUE;
	UpdateShowPics();
	Invalidate();
}


// 建立基于纹理的图片库特征集合
void CMyCBIRDlg::CreateLib()
{
	// 建立等待进度条
	CComboBox *a;
	CProgressCtrl *myPro = (CProgressCtrl*)GetDlgItem(IDC_PROGRESS_WAIT);
	myPro->SetRange(0,m_iPicNum);

	CDC* pDC = GetDC();
	Graphics graph(pDC->m_hDC);

	// 搜索type计算
	int type = m_nLibType;
	int  nLBPMethod = type & 0xF;
	type >>= 7;
	bool bTextureBased = type & 1;
	type >>= 1;
	int  nColorMethod = type & 0xF;
	type >>= 7;
	bool bColorBased = type & 1;
	m_nLibType = ((int)(bTextureBased || m_bTextureBased))<<7 |  m_nLBPMethod 
			| ((int)(bColorBased || m_bColorBased))<<15 | m_nColorMethod<<8;

	// 建立特征集合
	if((!bTextureBased && m_bTextureBased) || (bTextureBased && nLBPMethod != m_nLBPMethod))   //纹理
	{
		myPro->ShowWindow(SW_SHOW);
		a = (CComboBox*)GetDlgItem(IDC_COMBO_TEXTURE);
		CString	 b;
		a->GetLBText(a->GetCurSel(),b);
		CString  strPrint(L"更新纹理特征库: ");
		strPrint+=b;
		graph.DrawString(strPrint, -1, &Font(L"宋体",9,FontStyleRegular),PointF(20,350),&SolidBrush(Color::Black));
		for(int i=0;i<m_iPicNum;i++)
		{
			if(m_pLBPBuf[i] != NULL) delete [] m_pLBPBuf[i];
			m_pLBPBuf[i] = new int [256];
			Bitmap tempImage(fileSet.GetAt(i));
			CaculateLBP(&tempImage,m_pLBPBuf[i]);
			myPro->SetPos(i);
		}
		myPro->ShowWindow(SW_HIDE);
		CRect rect(10,340,200,380);
		InvalidateRect(&rect);
	}
	if((!bColorBased && m_bColorBased) || (bColorBased && nColorMethod != m_nColorMethod))		//颜色
	{
		myPro->SetPos(0);
		myPro->ShowWindow(SW_SHOW);
		a = (CComboBox*)GetDlgItem(IDC_COMBO_COLOR);
		CString	 b;
		a->GetLBText(a->GetCurSel(),b);
		CString  strPrint(L"更新颜色特征库: ");
		strPrint+=b;
		graph.DrawString(strPrint, -1, &Font(L"宋体",9,FontStyleRegular),PointF(20,350),&SolidBrush(Color::Black));
		for(int i=0;i<m_iPicNum;i++)
		{
			if(m_pColorBuf[i] != NULL) delete [] m_pColorBuf[i];
			m_pColorBuf[i] = new int [256];
			Bitmap tempImage(fileSet.GetAt(i));
			CaculateColor(&tempImage,m_pColorBuf[i]);
			myPro->SetPos(i);
		}
		myPro->ShowWindow(SW_HIDE);
	}
}

// 计算指定图片的LBP值,返回值存放在pLBP[256]
void CMyCBIRDlg::CaculateLBP(Bitmap* tempImage,int* pLBP)
{
		int  nWidth = tempImage->GetWidth();
		int	 nHeight = tempImage->GetHeight();
		int	 *img  =  new int[nWidth * nHeight];

		for(int i=0;i<nHeight;i++)
		{
			for(int j=0;j<nWidth;j++)
			{
				Color col;
				tempImage->GetPixel(j,i,&col);
				img[i*nWidth+j] = col.GetB()+col.GetG()+col.GetR();
			}
		}

		lbp_histogram(img, nHeight, nWidth, pLBP, m_nLBPMethod);
		delete [] img;
}

// 计算指定图片的颜色直方图,返回值存放在pColor[256]
void CMyCBIRDlg::CaculateColor(Bitmap* tempImage,int* pColor)
{
		for(int i=0;i<256;i++) pColor[i]=0;

		int  nWidth = tempImage->GetWidth();
		int	 nHeight = tempImage->GetHeight();
		byte  b,g,r,max,min,r_s,r_v;
		float s,v;
		USHORT  h;

		for(int i=0;i<256;i++) pColor[i]=0;

		for(int i=0;i<nHeight;i++)
		{
			for(int j=0;j<nWidth;j++)
			{
				Color col;
				tempImage->GetPixel(j,i,&col);
				b = col.GetB();	g = col.GetG();	r = col.GetR();

				max = b; min =g;
				if(b < g) { max = g; min =b;}
				if(r > max) max = r;
				else if(r < min) min = r;

				byte delta = max - min;

				v = (0.299*r + 0.587*g + 0.114*b)/255;

				if(max == 0)
					s = 0;
				else
					s = delta/max;

				if(delta == 0)
					h = 0;
				else if(max==r && g>b)
					h = 60 * (g-b)/delta;
				else if(max==r && g<b)
					h = 360 + 60 * (g-b)/delta;
				else if(max == g)
					h = 60*(2 + (b - r)/delta);
				else h = 60 * (4 + (r - g)/delta);

				//量化
				if(s < 0.15)		r_s = 0;
				else if(s <0.4)		r_s = 1;
				else if(s < 0.75)	r_s = 2;
				else				r_s = 3;

				if(v < 0.15)		r_v = 0;
				else if(v <0.4)		r_v = 1;
				else if(v < 0.75)	r_v = 2;
				else				r_v = 3;

				if(h <= 15 || h > 345)	h = 0;
				else if(h <= 25)		h = 1;
				else if(h <= 45)		h = 2;
				else if(h <= 55)		h = 3;
				else if(h <= 80)		h = 4;
				else if(h <= 108)		h = 5;
				else if(h <= 140)		h = 6;
				else if(h <= 165)		h = 7;
				else if(h <= 190)		h = 8;
				else if(h <= 220)		h = 9;
				else if(h <= 255)		h = 10;
				else if(h <= 275)		h = 11;
				else if(h <= 290)		h = 12;
				else if(h <= 316)		h = 13;
				else if(h <= 330)		h = 14;
				else					h = 15;
				pColor[16*h+4*r_s+r_v]++;
			}
		}
}

// 更新右侧显示的9张图片
void CMyCBIRDlg::UpdateShowPics(void)
{
	for(int i=0;i< 9;i++)
	{
		if(m_pLibPic[i] != NULL) delete m_pLibPic[i];
		CString filename;
		filename=fileSet.GetAt(index[i]);
		m_pLibPic[i] = new Bitmap(filename);
	}
}

