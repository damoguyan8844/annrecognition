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
		 {0.0265854374,0.0493980236,0.1221159995,0.1048806757,0.0854556412,0.0361720622,0.0578478351,0.0417455062,0.0917996764,0.1226996705,0.1243247762,0.0387852117,0.0786576420,0.0093653677,0.0939970091,0.1073755622,0.0567796864,0.0280198064,0.0291680656,0.1066126004,0.1062425599,0.0645504594,0.0734046474,0.0576265752,0.1206091493,0.0139469588,0.0654774606,0.0587405004,0.0591944642,0.1076921895,0.1125370041,0.0475249477,0.0003242592,0.0157742538,0.0717070550,0.0933942720,0.1011040062,0.0632610545,0.0768951997,0.0917920470,0.0066225166,0.0886371955,0.0599230938,0.0436185785,0.0641422793,0.0339136943,0.0278099924,0.0409138761,0.0101321451,0.0437177643,0.0453695804,0.0174565874,0.1107058898,0.0382740274,0.0598239079,0.0103534041,0.0483794659,0.0376216918,0.0167012550,0.1161877811,0.0356494337,0.0768570527,0.0287179183,0.0128597366},
		 {0.0060274056,0.0542580932,0.1222380772,0.0969573036,0.0167699214,0.1005508602,0.0745300129,0.0223509930,0.0963049680,0.0337649174,0.0064546647,0.0680753514,0.1174123362,0.1110492274,0.1010010093,0.0071718497,0.1068987101,0.0411351360,0.0146298101,0.0385296196,0.0606250204,0.1033051535,0.1241378486,0.0118717002,0.1132961512,0.1063913405,0.1183698550,0.0378048047,0.1215781122,0.0433820598,0.0248764008,0.1195867807,0.0316248052,0.1095347479,0.0736411661,0.0058023315,0.0896671936,0.0644703507,0.1128383726,0.0257805102,0.0166249573,0.1062501892,0.1039231569,0.1068071499,0.0805459768,0.1128765196,0.0359508060,0.1100154147,0.0627117231,0.0441946164,0.0363818780,0.0590647608,0.0512482077,0.1096644476,0.0182882175,0.0233504754,0.0581568331,0.1199568138,0.0162739959,0.1050141901,0.0184522532,0.0028839991,0.0509659126,0.0413716547},
		 {0.1025841534,0.0346728414,0.1126514450,0.0562532432,0.1053766012,0.0467009507,0.0193296615,0.0945348963,0.0375721008,0.0496459864,0.0514580235,0.1009743065,0.0017929624,0.0681974217,0.0770020112,0.1122394502,0.0646649078,0.0469260216,0.0769028291,0.0430234671,0.0421040989,0.0833002701,0.0413487665,0.0070993681,0.0329752490,0.0779442713,0.0186735131,0.0457892083,0.0251892153,0.1107097045,0.0472579114,0.1037972644,0.0860736445,0.0386898406,0.0134968106,0.0296754353,0.0861499384,0.1247558519,0.0412648395,0.0658131689,0.0931272283,0.0676671639,0.0807977542,0.1052354500,0.0102313301,0.1174390391,0.0525719486,0.0551278740,0.0956869721,0.0219542533,0.0890148655,0.0843836814,0.0624485016,0.1135364845,0.1046556011,0.0950041190,0.0200735498,0.0314188041,0.1125408188,0.0697653145,0.0827623829,0.0138859218,0.0175138097,0.0109027373},
		 {0.0454916544,0.0520149842,0.0719969794,0.0955572650,0.0118526258,0.0939092711,0.0874927491,0.0441908017,0.1120563373,0.0148815885,0.0563142784,0.0933713764,0.0900868252,0.1142078936,0.1199034080,0.1086039320,0.0823885277,0.0450262465,0.0429319143,0.0422490612,0.0704786852,0.0957975984,0.1143337786,0.1071466729,0.0261963252,0.0324640647,0.0279816575,0.0403187647,0.0269287694,0.0757240504,0.0517708361,0.0622692034,0.0556314290,0.0249336231,0.1060670763,0.0747665316,0.0130275888,0.1030304879,0.0602320917,0.0776314586,0.0318117328,0.0340624712,0.1182020009,0.0975104496,0.1201666296,0.0526787639,0.1018059328,0.1090578958,0.0995056033,0.1071352288,0.0028801844,0.1165196672,0.1137386709,0.0801873803,0.1127735227,0.0011940367,0.0499130227,0.0394947678,0.1003067121,0.0527359843,0.0225493629,0.0918034911,0.0968771949,0.0397427306},
		 {0.0103228856,0.0425084680,0.1104197800,0.1241035163,0.0842768624,0.0869357884,0.0715811625,0.0944700465,0.0650273114,0.0804620534,0.0531594269,0.0833536759,0.0454840250,0.0951338261,0.0315256193,0.0637836829,0.0254257340,0.1132580042,0.0393307284,0.0229003262,0.0028496659,0.1167638153,0.0680143163,0.0309228804,0.0499511696,0.0566652417,0.0935964510,0.1023361906,0.0416844673,0.1049989313,0.0500045791,0.0328989550,0.0173764769,0.0290612504,0.0116084777,0.0836130828,0.0933866426,0.0903538615,0.0342188776,0.0701696798,0.0322084725,0.0579508357,0.0412152484,0.1040185243,0.0111163668,0.1137958914,0.0791116059,0.0436567292,0.0706656054,0.1146313399,0.0791802704,0.0272148810,0.0361262858,0.1129795238,0.0744346455,0.0199018829,0.0173039958,0.1020081192,0.0086138491,0.1224822253,0.0296639912,0.0115932189,0.0864818245,0.0742973089},
		 {0.0024910735,0.0665875748,0.1200827062,0.0719855353,0.0095942561,0.0976096392,0.0668965727,0.1099352986,0.0854480118,0.0342265069,0.0156254768,0.0637798682,0.0352984704,0.1062921509,0.0499549843,0.0674764216,0.0036011841,0.0845553428,0.1191557050,0.0761398673,0.1023094878,0.0868823826,0.0141567737,0.0146641443,0.0227668080,0.0565431677,0.0362521745,0.1121135578,0.0024109622,0.0806337148,0.0107043674,0.1211241484,0.0797296092,0.0290345475,0.1102748215,0.0590418726,0.0086787008,0.0569055751,0.0157895144,0.0450109877,0.0161366612,0.0761474967,0.0038453322,0.1197126657,0.0234076977,0.1138607413,0.1137195975,0.1220015585,0.0155072175,0.0382244326,0.1085695997,0.0091326637,0.0541856140,0.0697653145,0.0081293676,0.0698492378,0.1183774844,0.0585039817,0.0327845104,0.0419400632,0.0181127358,0.0699331611,0.1113658547,0.0836359784},
		 {0.0902737528,0.0045358134,0.0962668210,0.0831400529,0.0434011370,0.1095805243,0.0918568969,0.0496383570,0.0929975286,0.0157551803,0.0916852355,0.1209181473,0.0206228830,0.0041924804,0.0367671736,0.0758842751,0.0982810482,0.1056245640,0.1055711508,0.0768799409,0.1215208918,0.0841319039,0.1100192294,0.0324373618,0.0541627258,0.0920628980,0.0011825922,0.0901974514,0.0462775044,0.1171338558,0.0275276955,0.0372707285,0.0246780291,0.0303964354,0.0262001399,0.0729392394,0.1176679283,0.1061281189,0.0153775141,0.1060022265,0.1131855249,0.0722106099,0.0680829808,0.0416196175,0.0871799365,0.0621662028,0.0104716634,0.0850207508,0.0263145845,0.0273636580,0.0500656143,0.0268715471,0.0818124935,0.1055253744,0.0018578143,0.0377132483,0.0285920277,0.0041390727,0.0928067863,0.0612392053,0.0664693117,0.0198637340,0.0126537373,0.0838190839},
		 {0.0071298867,0.0709440932,0.0035515917,0.1220740378,0.0491157249,0.0688573867,0.1176602989,0.0858142301,0.0705664232,0.0923070461,0.0282143615,0.0647907928,0.0371982493,0.0850665271,0.0195776243,0.0354091004,0.1067651883,0.0626506880,0.0985976756,0.0955458209,0.0257499926,0.0790276825,0.0838496014,0.0754303113,0.0038796656,0.1174123362,0.0535027608,0.0706427172,0.0312471390,0.0418332480,0.0229117703,0.0652256832,0.0404026918,0.0587672032,0.0355349891,0.0003242592,0.0032464066,0.0754226819,0.0382969156,0.0813470855,0.1151310802,0.0290192869,0.0999595597,0.0541589111,0.0308923610,0.0737441629,0.0935850069,0.1181943715,0.0552919097,0.0745338276,0.0267342143,0.0820223093,0.0026017029,0.0445303209,0.0154538099,0.0543305762,0.0027581104,0.1003181562,0.0856387541,0.0257271044,0.0938634872,0.0717032403,0.0479064286,0.0094378488}
};
     double  B1[IMPLICITDEM]=
	 {0.0424588770,0.0304307695,0.0116008483,0.0922803432,0.0655423105,0.0376178771,0.0034180731,0.0243919194}
;
 //    double  A1[IMPLICITDEM];

     double  W2[OUTPUTDEM][IMPLICITDEM]={
		 {0.0328121558,0.0474864505,0.3456012309,0.2370869666,0.2129499167,0.2818651497,0.3101347387,0.2401405126},
		 {0.1511776000,0.1180957034,0.3241832256,0.3157023489,0.0668327808,0.1341187358,0.3007043600,0.2322207093},
		 {0.0219143331,0.0850030109,0.1551806629,0.3106418848,0.0574347600,0.0663472340,0.0509608053,0.0531187877},
		 {0.3101455271,0.1613201350,0.3073617518,0.1812922806,0.1790695637,0.2375401407,0.1648376435,0.0151598416}
};
     double  B2[OUTPUTDEM]=
{0.1575652361,0.0623118021,0.1410782337,0.0401169322}
;
	InitBPParameters_EX(INPUTDEM,IMPLICITDEM,OUTPUTDEM,(double **)W1,B1,(double **)W2,B2);
	SaveBPParameters("C:\\BPMYParas.dat");
//	LoadBPParameters("C:\\Type1_Paras.dat");
//	PrintBPParameters("C:\\Type1_Paras.tex");

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
	test();
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
