// ChildView.cpp : implementation of the CChildView class
//
#include "stdafx.h"
#include "DigitRec.h"
#include "ChildView.h"
#include "INPUT1.h"
#include "ANNRecognition.h"
#include "Bp.h"
#include "DBpParamater.h"
#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif
void ThiningDIB(LPSTR lpDIBBits, LONG lWidth, LONG lHeight);
/////////////////////////////////////////////////////////////////////////////
// CChildView

typedef deque<RECT>  CRectLink;
typedef deque<HDIB>  HDIBLink;


//声明一些必要的全局变量
int w_sample=8;
int h_sample=16;
bool fileloaded;
bool gyhinfoinput;
bool gyhfinished;
//int digicount;
int m_lianXuShu;
//CRectLink m_charRectCopy;
//CRectLink m_charRect;

LONG m_charRectID;

HDIBLink  m_dibRect;
HDIBLink  m_dibRectCopy;

HDIB m_hDIB;
string strPathName;
string strPathNameSave;

CChildView::CChildView()
{
	fileloaded=false;
	gyhinfoinput=false;
	gyhfinished=false;
	m_hDIB=NULL;
}

CChildView::~CChildView()
{
}

BEGIN_MESSAGE_MAP(CChildView,CWnd )
	//{{AFX_MSG_MAP(CChildView)
	ON_WM_PAINT()
	ON_COMMAND(IDmy_FILE_OPEN_BMP, OnFileOpenBmp)
	ON_COMMAND(IDmy_FILE_SAVE_BMP, OnFileSaveBmp)
	ON_COMMAND(IDmy_IMGPRC_SHRINK_ALIGN, OnImgprcShrinkAlign)
	ON_COMMAND(IDmy_IMGPRC_ALL, OnImgprcAll)
	ON_COMMAND(IDmy_IMGPRC_256ToGray, OnIMGPRC256ToGray)
	ON_COMMAND(IDmy_IMGPRC_DIVIDE, OnImgprcDivide)
	ON_COMMAND(IDmy_IMGPRC_TO_DIB_AND_SAVE, OnImgprcToDibAndSave)
	ON_COMMAND(IDmy_IMGPRC_REMOVE_NOISE, OnImgprcRemoveNoise)
	ON_COMMAND(IDmy_IMGPRC_STANDARIZE, OnImgprcStandarize)
	ON_COMMAND(IDmy_IMGPRC_THINNING, OnImgprcThinning)
	ON_COMMAND(IDmy_IMGPRC_ADJUST_SLOPE, OnImgprcAdjustSlope)
	ON_COMMAND(IDmy_IMGPRC_GrayToWhiteBlack, OnIMGPRCGrayToWhiteBlack)
	ON_COMMAND(IDmy_IMGPRC_SHARP, OnImgprcSharp)
	ON_COMMAND(IDmy_FILE_RE_LOAD_BMP, OnFileReLoadBmp)
	ON_COMMAND(ID_INPUT1, OnInputGuiyihuaInfo)
	ON_COMMAND(IDmy_BPNET_TRAIN, OnBpnetTrain)
	ON_COMMAND(IDmy_BPNET_RECOGNIZE, OnBpnetRecognize)
	ON_COMMAND(ID_aver, Onaver)
	ON_COMMAND(ID_Gass, OnGass)
	ON_COMMAND(ID_Mid, OnMid)
	ON_COMMAND(IDmy_IMGPRC_EQUALIZE, OnImgprcEqualize)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CChildView message handlers

BOOL CChildView::PreCreateWindow(CREATESTRUCT& cs) 
{
	if (!CWnd::PreCreateWindow(cs))
		return FALSE;

	cs.dwExStyle |= WS_EX_CLIENTEDGE;
	cs.style &= ~WS_BORDER;
	cs.lpszClass = AfxRegisterWndClass(CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS, 
		::LoadCursor(NULL, IDC_ARROW), HBRUSH(COLOR_WINDOW+1), NULL);

	return TRUE;
}

void CChildView::OnPaint() 
{
	CPaintDC dc(this); // device context for painting
	OnDraw(&dc);
	// Do not call CWnd::OnPaint() for painting messages
}

//打开256色位图文件
void CChildView::OnFileOpenBmp() 
{
	//创建一个打开文件对话框，并返回完整的文件路径
	static char BASED_CODE szFilter[] = "256色位图文件(*.bmp)|";
	CFileDialog dlg(TRUE,NULL,NULL,OFN_HIDEREADONLY|OFN_OVERWRITEPROMPT,szFilter,NULL);
    if(dlg.DoModal() == IDOK)
	   strPathName = dlg.GetPathName();
	else return;
	//创建一个文件对象
   	//CFile file;
	//以只读模式打开文件
	//file.Open (strPathName.c_str(),CFile::modeRead);
	//读取文件到HDIB句柄中. 注意:此时只是读取位图文件中文件头之后的部分,不含文件头
	m_hDIB=::ReadDIBFile (strPathName.c_str());
	//HDIB句柄: 就是一块存储位图数据的内存区域的地址
	//HDIB句柄包含:位图信息头、调色板(如果有的话)、DIB图像数据
	//关闭文件
	//file.Close ();
	//指向DIB的指针(指向位图信息头)
	BYTE* lpDIB=(BYTE*)::GlobalLock ((HGLOBAL)m_hDIB);
	// 获取DIB中颜色表中的颜色数目
	WORD wNumColors;	
	wNumColors = ::DIBNumColors((char*)lpDIB);	
	// 判断是否是256色位图
	if (wNumColors != 256)
	{
		// 提示用户
		MessageBox("非256色位图！", "系统提示" , MB_ICONINFORMATION | MB_OK);
		// 解除锁定
		::GlobalUnlock((HGLOBAL)m_hDIB);
		// 返回
		return;
	}
	//在屏幕上显示位图
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
	//更改位图文件是否已加载的标志
	fileloaded=true;
    //gyhinfoinput=false;          //2004.4.26修改
	gyhfinished=false;

	::GlobalUnlock ((HGLOBAL)m_hDIB);

}

//取消一切更改，重新加载位图文件
void CChildView::OnFileReLoadBmp() 
{
	//判断位图文件是否已加载。如果尚未加载，则弹出文件打开对话框
	if(fileloaded==false)
	{
		OnFileOpenBmp();
		if(fileloaded==false)
			return;
	}
	//创建一个文件对象
//   	CFile file;
	//以只读模式打开文件
//	file.Open (strPathName,CFile::modeReadWrite);
	m_hDIB=::ReadDIBFile (strPathName.c_str());
	//关闭文件
//	file.Close ();
	//在屏幕上显示位图
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);		
}


void CChildView::OnFileSaveBmp() 
{
	//创建一个保存文件对话框，并返回完整的文件路径
	static char BASED_CODE szFilter[] = "256色位图文件(*.bmp)|";
	CFileDialog dlg(FALSE,NULL,NULL,OFN_HIDEREADONLY|OFN_OVERWRITEPROMPT,szFilter,NULL);
    if(dlg.DoModal() == IDOK)
	   strPathNameSave =dlg.GetPathName();
	else return;

	//在文件名后添加.bmp后缀
	//strPathNameSave+=".bmp";
	//以读写模式打开一个文件。如果文件不存在，则创建之
// 	CFile file(strPathNameSave, CFile::modeReadWrite|CFile::modeCreate);
	
	::SaveDIB (m_hDIB,(char *)strPathNameSave.c_str());
	//关闭文件
//	file.Close ();	
}

//一次性预处理
void CChildView::OnImgprcAll() 
{
	if(fileloaded==false)
	{
		if(::AfxMessageBox ("请先打开一个图像文件再进行此操作！",MB_YESNO|MB_ICONSTOP)==IDNO)
		   return;
	}
	//打开文件
	OnFileReLoadBmp();
	//判断用户是否已输入归一化高度和宽度信息
	if(gyhinfoinput==false) OnInputGuiyihuaInfo();
	//将256色图转换为灰度图
	OnIMGPRC256ToGray();
	//将灰度图二值化
	OnIMGPRCGrayToWhiteBlack();
	//梯度锐化
	//OnImgprcSharp();
	//去除离散杂点噪声
	OnImgprcRemoveNoise();
	//调整数字字符的整体倾斜
	OnImgprcAdjustSlope();
	//分割并画框标识
	OnImgprcDivide();
	//将分割后的数字字符宽、高标准化，以便于下一步与BP网络的输入兼容
	OnImgprcStandarize();
	ConvertGrayToWhiteBlack(m_hDIB);
	//紧缩重排数字字符
	OnImgprcShrinkAlign();
	//分别保存这些已经经过分割、标准后的单个的数字字符到bmp文件，以便后续过程使用
	//OnImgprcToDibAndSave();
	//OnPreprocThin();	
}

//图像预处理第1步：将256色图像转化为灰度图像
void CChildView::OnIMGPRC256ToGray() 
{	
	Convert256toGray(m_hDIB);	
	//在屏幕上显示位图
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
}

//图像预处理第2步：将灰度图二值化
void CChildView::OnIMGPRCGrayToWhiteBlack()
{
	ConvertGrayToWhiteBlack(m_hDIB);
	//在屏幕上显示位图
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
}

//图像预处理第3步：梯度锐化
void CChildView::OnImgprcSharp() 
{
	GradientSharp(m_hDIB);
	//在屏幕上显示位图
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);		
}

//图像预处理第4步：去离散杂点噪声
void CChildView::OnImgprcRemoveNoise() 
{
	RemoveScatterNoise(m_hDIB);
	//在屏幕上显示位图
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
}

//图像预处理第5步：倾斜度调整
void CChildView::OnImgprcAdjustSlope() 
{
    SlopeAdjust(m_hDIB);
	//在屏幕上显示位图
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
}

//图像预处理第6步：分割，并在分割出来的字符外面画框以标识
void CChildView::OnImgprcDivide() 
{
	m_charRectID=CharSegment(m_hDIB);
	//在屏幕上显示位图
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
	DrawFrame(pDC->m_hDC,m_hDIB,m_charRectID,2,RGB(20,60,200));
}

//图像预处理第7步：标准归一化
//将分割出来的各个不同宽、高的数字字符宽、高统一
void CChildView::OnImgprcStandarize() 
{
	StdDIBbyRect(m_hDIB,m_charRectID,w_sample,h_sample);
	//在屏幕上显示位图
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
	DrawFrame(pDC->m_hDC,m_hDIB,m_charRectID,2,RGB(21,255,25));
	gyhfinished=true;
}

//图像预处理第8步：紧缩重排已经分割完毕的数字字符，并形成新的位图句柄
void CChildView::OnImgprcShrinkAlign() 
{
	m_hDIB=AutoAlign(m_hDIB,m_charRectID);
	//在屏幕上显示位图
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
	DrawFrame(pDC->m_hDC,m_hDIB,m_charRectID,1,RGB(252,115,27));
}

//图像预处理第9步：将最终标准化后的字符图像分为单个单个的HDIB保存，并存为.bmp文件
void CChildView::OnImgprcToDibAndSave() 
{
	string strFolder=strPathName.substr(0,strPathName.find_last_of("\\"));

	SaveSegment(m_hDIB,m_charRectID,(char *)strFolder.c_str());
}

void CChildView::OnImgprcThinning() 
{
	 Thinning(m_hDIB);	
	//在屏幕上显示位图
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
}



void CChildView::OnInputGuiyihuaInfo() 
{
	CINPUT1 input;
	input.w =8;
	input.h =16;
	if(input.DoModal ()!=IDOK) return;
	w_sample=input.w;
	h_sample=input.h;
	gyhinfoinput=true;
}

void CChildView::OnBpnetTrain() 
{
	OnImgprcAll();
	//判断是否经过了归一划的处理
	if(gyhfinished==false)
	{
		//如果没有进行提示错误并返回
		::MessageBox(NULL,"没有进行归一划预处理",NULL,MB_ICONSTOP);
		return;
	}
	
	//建立BP网络训练参数对话框
	
	CDBpParamater BpPa;
	
	//初始化变量
	BpPa.m_a=0;
	BpPa.m_eta=0.015;
	BpPa.m_ex=0.001;
	BpPa.m_hn=10;
	
	// 显示对话框
	if(BpPa.DoModal()!=IDOK)
	{
		//返回
		return;
	}
	//用户获得参数信息
	
	//相关系数
	double  momentum=BpPa.m_a; 
	//最小均方误差
	double  min_ex=BpPa.m_ex; 
	//隐层数目
	int  n_hidden=BpPa.m_hn; 
	//训练步长
	double eta=BpPa.m_eta;
	
	int digicount=GetSegmentCount(m_charRectID);
	
	//获得指向DIB的指针
	BYTE *lpDIB=(BYTE*)::GlobalLock((HGLOBAL) m_hDIB);
	
	//获得指向DIB象素的指针，并指向象素的起始位置
	BYTE *lpDIBBits =(BYTE*)::FindDIBBits((char *)lpDIB);
	
	//获得颜色信息
	int numColors=(int) ::DIBNumColors((char *)lpDIB);
	//不是灰度图返回
    if (numColors!=256) 
	{
		::GlobalUnlock((HGLOBAL) m_hDIB);
		::MessageBox(NULL,"只能处理灰度图像",NULL,MB_ICONSTOP);
		return;
	}
	
	//获取图像的宽度
    LONG lWidth = (LONG) ::DIBWidth((char *)lpDIB); 
	
	//获取图像的高度
	LONG lHeight = (LONG) ::DIBHeight((char *)lpDIB);
	
	//计算图像每行的字节数
	LONG lLineByte = (lWidth+3)/4*4; 
	
	//归一化的宽度
	LONG lSwidth = w_sample;
	
	//归一化的高度
	LONG LSheight = h_sample;
	
	//指向输入样本的特征向量的指针  
	double **data_in;
	//从输入的训练样本中提取特征向量
	data_in = code ( lpDIBBits, digicount,  lLineByte, lSwidth, LSheight);
	
	//计算输入层结点的数目
	
	int n_in = LSheight*lSwidth;
	
	double out[][4]={      0.1,0.1,0.1,0.1,
		0.1,0.1,0.1,0.9,
		0.1,0.1,0.9,0.1,
		0.1,0.1,0.9,0.9,
		0.1,0.9,0.1,0.1,
		0.1,0.9,0.1,0.9,
		0.1,0.9,0.9,0.1,
		0.1,0.9,0.9,0.9,
		0.9,0.1,0.1,0.1,
		0.9,0.1,0.1,0.9};
	
	
	
	double **data_out;
	
	data_out = alloc_2d_dbl(digicount,4);
	
	for(int i=0;i<digicount;i++)
	{
		for(int j=0;j<4;j++)
			data_out[i][j]=out[i%10][j];
		
	}
	
	BpTrain( data_in, data_out, n_in,n_hidden,min_ex,momentum,eta,digicount);
	
	::GlobalUnlock(m_hDIB);
	
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);
	
}



#define INPUTDEM      64
#define IMPLICITDEM   8
#define OUTPUTDEM     4 

double  data[INPUTDEM+1];

bool test()
{
	double  W1[IMPLICITDEM][INPUTDEM]={
		{0.0657711998,-1.0408977270,-0.6358233094,-1.0153377056,0.9062056541,5.4504475594,0.0543763526,0.1016228199,0.0345469527,-1.2216038704,-0.7230613828,2.7239272594,2.4320642948,-1.8046431541,-1.0150972605,0.0106605720,0.0227019563,-1.1099717617,-1.2567458153,2.2522141933,-2.5440146923,0.3748340607,1.4435259104,0.0543953814,0.1014969349,2.0947003365,-0.6830629110,-2.2538888454,-3.1687049866,-0.8890119195,-0.1609236896,-0.0247288570,0.0395291001,1.2614732981,-1.2125163078,-1.4397275448,0.0987524912,1.1453298330,1.1698075533,-0.0406300016,0.0678846091,1.7711954117,1.6882512569,-1.2285256386,-1.0323500633,3.5516011715,-1.0446614027,0.0581298396,0.0983916745,0.2308012843,0.1480615288,-1.9022634029,2.7312784195,-1.2308000326,-0.5429793000,0.0869238377,0.0199896246,1.0711858273,-0.3765482903,0.6005362272,0.5691500902,-0.8414043784,0.4517452121,0.0755005330},
		{0.0211989190,0.4827364087,0.1287993938,0.4879863858,0.9721140862,1.2839750051,0.0280045476,0.0930051580,0.0363513604,-0.0149183702,-3.5280985832,2.1412277222,-0.0979295373,1.8925156593,-1.2524473667,0.0268760547,0.0657406822,-1.2222776413,-2.0959596634,0.3556683660,0.6260790825,1.7420262098,-2.6168351173,0.0332354940,0.0551202446,-0.6904306412,-0.5508130193,-0.4266467392,1.9450210333,-0.2292221189,-2.9323678017,0.0404477976,0.1121402606,2.9920332432,0.7581982613,3.0068986416,-0.7193978429,-2.1295166016,-3.6252281666,0.0679905564,0.0859019756,2.3699996471,6.4816179276,-0.6630298495,-0.0622091070,-2.1387367249,-0.5717818737,0.0633199066,0.0372020639,-1.1647202969,-1.9721451998,-1.3267503977,0.3576690555,1.9055876732,0.8641766310,0.0764810294,0.0993720815,-3.2507245541,-1.9577413797,-0.0253037885,-3.0230355263,1.6238859892,0.8612628579,-0.0311680865},
		{0.0339136943,0.1333990544,0.5944855213,1.5579904318,1.6712938547,1.4033030272,0.0642948672,0.0210005492,0.0991965979,-1.7205022573,-1.5068756342,4.9245886803,2.8408503532,0.0853944346,-1.6774814129,0.1005876735,0.0970755666,0.0033422790,-1.3745654821,0.2047235966,-0.4279236495,-4.1755084991,-5.7998008728,0.0764322728,0.0553872809,0.8094511032,-1.7537610531,1.3075741529,0.6493140459,-0.4767634571,-5.2927579880,0.6922936440,0.0045968504,-0.3944042623,1.4944385290,4.8526468277,1.6691625118,-4.5625948906,-6.1335301399,0.7990561724,0.1140552983,-2.4338922501,0.9767515659,1.8496026993,2.4591665268,-0.5412976742,-3.7483665943,0.7651758790,0.0313196220,-2.9123678207,-2.1100964546,0.8034732938,2.1665184498,0.0539541729,-0.5003786683,0.8385582566,0.0496688746,1.4598979950,-1.0422117710,0.7221356630,-0.5475946665,1.1029571295,-0.5753654242,0.2938985825},
		{0.1090159342,-0.8091817498,-2.1233696938,-1.4307942390,-1.8871173859,-1.4621925354,0.0316324346,0.0987693444,0.0412343219,0.2650355995,0.4397537112,0.5505270362,-1.8308060169,-0.8978471160,0.6949614882,-0.0060911127,0.0476088747,-2.9999425411,-2.6974067688,-0.0561672598,0.3926002979,0.5687451363,2.0348138809,-0.0239059478,0.0430616178,-1.0067508221,-0.5323839188,-1.2760251760,2.3126606941,3.2627983093,2.5943460464,-0.2406890988,0.0947447121,-0.0802738369,-1.5817034245,2.3311226368,-2.3485629559,-1.7960362434,0.9214257598,-0.1709846854,0.0133518483,0.6997285485,3.5863673687,2.0381948948,1.8990703821,-0.8996594548,-0.1236706898,-0.1344173700,0.0850779712,1.5272589922,1.2149499655,-2.9770672321,1.2900991440,3.1671802998,-0.4556728601,-0.1853604466,0.0691282377,0.9863396287,3.1271355152,0.9842633009,0.7407570481,2.8153467178,0.4856957495,0.0699187964},
		{0.0187994018,-1.2292592525,-2.8382351398,-1.7299988270,-0.9380283952,0.0638878793,0.1193426326,0.0550668351,0.0742362738,-0.0224502645,0.2472783327,-0.8109086156,1.3995254040,-0.0232292805,-0.7246418595,0.0599259995,0.0227935109,-0.9517186284,-2.4104866982,1.6725218296,2.3453915119,-1.7128175497,0.3150989115,0.0991617963,0.0932264179,2.3964443207,-3.3768141270,-2.0219733715,0.7245206237,-1.1775341034,-1.8507043123,-0.0199586879,0.0771126449,2.0450937748,-0.2374781221,1.0496165752,-0.6196383834,0.9443879724,2.3421278000,0.0424244478,0.0549218729,2.9078598022,4.3644609451,0.6047604084,-3.3235831261,0.2197382897,-0.7988086343,0.0811457261,0.0752204955,0.9127132297,-1.9734805822,0.6750795841,3.1705403328,-1.8906745911,-2.0637538433,0.1014575064,0.0369846188,1.8750017881,0.1336778253,0.7814764977,-0.2893581092,0.4990823269,-0.3993346393,0.1169251725},
		{0.0329676196,-0.0155727640,1.9910529852,1.2743365765,0.5783035755,2.3062982559,0.0475211330,0.1042550430,0.1249122620,0.7219545841,1.6058586836,1.1746187210,2.3206634521,-3.3424274921,-0.1172906905,0.0667779446,0.1213835552,-0.5114148259,2.5515954494,0.9962009192,-1.0146026611,-2.6173300743,-1.3269346952,0.1516765505,0.0093386639,1.4643285275,0.4341837466,-0.0417051055,-0.8119259477,-2.2024452686,-1.8816294670,0.0276094750,0.0286797695,-1.7766211033,-0.0433470346,0.5409465432,0.3054506779,-2.6809372902,3.1873898506,0.1979211271,0.1190221906,-7.1418032646,-3.7307815552,-1.3740559816,3.3128159046,2.2357068062,1.1674804688,0.0459846370,0.0071031833,-2.6659672260,-2.7310557365,1.3743788004,-1.4948667288,3.4187386036,-2.3608117104,0.0619569235,0.0598277226,2.1600377560,-1.9083691835,0.5954258442,-3.8900358677,-1.4454206228,-4.1745910645,0.0521583743},
		{0.0938024521,1.8354763985,2.0390748978,1.7782045603,2.3457126617,-2.5836122036,0.0113567002,0.0572565384,0.0292825103,-0.6040725708,-4.3071784973,1.5384663343,1.7446465492,2.0253505707,0.3599997163,0.0059866025,0.0824877173,0.2409967035,-0.0186995380,3.9516191483,3.6102645397,-4.1976952553,-3.9672758579,-0.0002127064,0.1166493744,-0.1022792831,1.1637876034,-3.7542240620,0.3411002159,2.9743683338,-1.3526027203,1.1263500452,0.0369541012,2.2860844135,-1.2613040209,-3.4648177624,-1.7370548248,-0.6298654675,-8.7525081635,1.3524105549,0.0752166808,-1.4690160751,1.3494399786,-1.8124730587,0.9807844162,0.7277284265,-1.2681220770,1.2672618628,0.0276383255,2.0924801826,1.4775657654,1.0655258894,-2.7032184601,-1.6777663231,-0.6309412122,1.2862584591,0.0707609802,-0.1103453636,0.5151748061,1.6842700243,1.4036649466,1.7276440859,1.0148839951,0.5086287856},
		{0.0087054046,0.3466931880,-0.1528209001,-0.1072777808,-0.1137043834,-5.6016192436,0.1217039973,0.0108188121,0.0711157545,0.6084930897,2.9706604481,0.3318933845,0.4049075842,2.3218021393,1.0107823610,0.0554309562,0.0373966172,-0.2665547729,-2.2984919548,4.7131099701,5.1895380020,-5.3428745270,-0.3487697542,0.0017068973,0.0642834231,1.7056558132,0.3032623231,-4.1278343201,1.5444316864,2.5005319118,-0.7276236415,0.6692475677,0.0303239543,2.2097520828,-0.7029827833,-1.4169024229,-2.2788546085,-3.6182408333,0.3419445157,1.0151427984,0.0568826869,1.6730331182,2.8975551128,0.4823206365,-1.7130035162,-3.9895083904,-2.6018884182,0.9672625661,0.0447515808,1.8478246927,2.3232645988,5.3267760277,-0.6861777306,-0.0299016126,-5.1955800056,0.9336569905,0.1014130041,-0.2974632978,1.4305561781,2.7069442272,-0.6896243691,2.5634098053,-2.4636023045,0.6007676721}
	};
	double  B1[IMPLICITDEM]=
	{-0.1919712722,-1.6143378019,3.0309400558,-1.9166448116,0.3304476440,3.1510319710,1.0307453871,1.3527212143}
	;
//	double  A1[IMPLICITDEM];
	
	double  W2[OUTPUTDEM][IMPLICITDEM]={
		{-0.4972343147,-5.9174227715,3.3744943142,-5.9904694557,-3.7126841545,17.2322082520,-1.6223185062,-2.7246036530},
		{-13.0809593201,2.9710805416,5.7524342537,-6.1870079041,-10.2996187210,-5.2262315750,13.7801971436,7.2981524467},
		{4.6096968651,13.0992889404,13.8470678329,-12.4144105911,-2.2255043983,-2.0570797920,4.0330328941,-11.5482416153},
		{-5.1561913490,-4.8094496727,-9.4497251511,2.9515256882,-3.6377749443,-0.0646793768,-2.9509148598,-10.6974391937}
	};
	double  B2[OUTPUTDEM]=
	{-0.4109358490,-5.3574833870,1.6938216686,6.6316337585}
;
//	InitBPParameters_EX(INPUTDEM,IMPLICITDEM,OUTPUTDEM,(double **)W1,B1,(double **)W2,B2);
	LoadBPParameters("C:\\Type1_Paras.dat");
	PrintBPParameters("C:\\Type1_Paras.tex");

 	double result[4];
	Recognition(data,result);

	int out=0;
	for(int i=0;i<OUTPUTDEM;i++)
	{        
		if(result[OUTPUTDEM-1-i]>0.5) out=out*2+1;
		else    out*=2;
    }
	if(out==(int)data[INPUTDEM])	return true;	
	return false;

}

#define TEST_FILE   "C:\\BP\\testData.txt"  

void Main()
{
	char datr[200];
	int check=0;
	int total=0;
//	cout<<TEST_FILE<<endl;
	
	ifstream fi;
	fi.open(TEST_FILE,ios::in);
	if(!fi)
	{
		cout<<"File ont find"<<endl;
		return;
	}
	while(fi.getline(datr,200))    /*??????*/
	{   
		int begin=0;
		int point=0;
		int temp=0;
		int i=0;
		do{
			if(datr[i]==','||datr[i]=='\0')
			{
				data[point]=datr[i-1]-'0';
				if(begin==i-2)
					data[point]+=10*(datr[i-2]-'0');
				data[point]=data[point]/16;   //??????????????????????,0 1 ?.
				point++;
				begin=i+1;	     
			}
			i++;
		}while(i<200&&datr[i-1]!='\0');
		data[point-1]=data[point-1]*16;
		if(test())  check++;
		total++;
	}
	fi.close();
	
	char Rates[100];
	sprintf(Rates,"%f",check*100.0/total);
	AfxMessageBox(Rates);
		
}
void CChildView::OnBpnetRecognize() 
{
//	LoadBPParameters("C:\\001278.dat");
	Main();
	PrintBPParameters("C:\\001278.txt");
//	test();
	
 	HDIB newDIB= ReadDIBFile("C:\\9.bmp");
 	double outCode[64];
 	BPEncode(newDIB,outCode);
 	double result[4];
	Recognition(outCode,result);
	// TODO: Add your command handler code here
	OnImgprcAll();
	//判断是否经过了归一划的处理
	if(gyhfinished==false)
	{
		//如果没有进行提示错误并返回
		::MessageBox(NULL,"没有进行归一划预处理",NULL,MB_ICONSTOP);
		return;
	}
	//获得指向DIB的指针
	BYTE *lpDIB=(BYTE*)::GlobalLock((HGLOBAL) m_hDIB);
	
	//获得指向DIB象素的指针，并指向象素的起始位置
	BYTE *lpDIBBits =(BYTE*)::FindDIBBits((char *)lpDIB);
	
	//获得颜色信息
	int numColors=(int) ::DIBNumColors((char *)lpDIB);
	//不是灰度图返回
    if (numColors!=256) 
	{
		::GlobalUnlock((HGLOBAL) m_hDIB);
		::MessageBox(NULL,"只能处理256色图像",NULL,MB_ICONSTOP);
		return;
	}
	
	int digicount=GetSegmentCount(m_charRectID);

	//获取图像的宽度
    LONG lWidth = (LONG) ::DIBWidth((char *)lpDIB); 
	
	//获取图像的高度
	LONG lHeight = (LONG) ::DIBHeight((char *)lpDIB);
	
	//计算图像每行的字节数
	LONG lLineByte = (lWidth+3)/4*4; 
	
	//归一化的宽度
	LONG lSwidth = w_sample;
	
	//归一化的高度
	LONG LSheight = h_sample;
	
	// 读取结点信息
	int n[3];
	if(r_num(n,"num")==false)
		return;
	//获得输入层结点数目
	int  n_in=n[0];
	//获得隐层结点数目
	int  n_hidden=n[1];
	//获得输出层结点数目
	int  n_out=n[2];  
	
	//判断待识别样本的归一划信息是否与训练时相同
	if(n_in!=lSwidth*LSheight)
	{
		//如果不相同提示错误并返回
		::MessageBox(NULL,"归一划尺寸与上一次训练时不一致",NULL,MB_ICONSTOP);
		return;
	}
	
	//指向输入样本的特征向量的指针  
	double **data_in;
	//从输入的训练样本中提取特征向量
	data_in = code ( lpDIBBits, digicount,  lLineByte, lSwidth, LSheight);
	
	//根据提取的特征进行样本识别
	CodeRecognize(data_in, digicount,n_in,n_hidden,n_out);
	::GlobalUnlock(m_hDIB);
	
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);
	
}

void CChildView::OnDraw(CDC *pDC)
{
	if(m_hDIB!=NULL) 
		DisplayDIB(pDC->m_hDC,m_hDIB);
}


void CChildView::Onaver() 
{
	// TODO: Add your command handler code here
//设定模板参数
	double tem[9]={1,1,1,
		           1,1,1,
				   1,1,1};

    //设定模板系数
    double  xishu = 0.111111;   

    //进行模板操作
	m_hDIB =Template(m_hDIB,tem ,3,3, xishu);

	//显示图像
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);
	
}

void CChildView::OnGass() 
{
	// TODO: Add your command handler code here
   
	//设定模板参数
	double tem[9]={1,2,1,
		           2,4,2,
				   1,2,1};

    //设定模板系数
    double  xishu = 0.0625;   

    //进行模板操作
	m_hDIB =Template(m_hDIB,tem ,3,3, xishu);

	//显示图像
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);
}

void CChildView::OnMid() 
{
	// TODO: Add your command handler code here
	//进行中值滤波
	m_hDIB =MidFilter(m_hDIB,3,3);

	//显示图像
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);
}

void CChildView::OnImgprcEqualize() 
{
	Equalize(m_hDIB);

	//显示图像
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);
}
