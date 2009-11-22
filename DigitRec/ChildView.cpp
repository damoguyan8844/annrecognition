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
		{0.0621204264,3.4936389923,3.0880053043,2.1127407551,0.6690436602,-4.4679989815,0.0494018383,0.0807100162,0.0507560968,0.5717294812,-1.3001636267,-2.8884997368,0.1357838511,6.5220193863,0.9694105387,0.0121935857,0.0472502820,2.7780897617,0.8082616329,0.6163881421,4.9815740585,-7.1181836128,-4.2313857079,0.0595169105,0.0156254768,-1.4709511995,1.9206835032,-4.0488452911,3.2583816051,1.9559578896,-4.7548580170,0.0894860476,0.1141354069,1.4876691103,0.5291944146,-2.5992300510,0.5398299694,-5.8505358696,-4.3423032761,0.0545146577,0.0678083152,-1.4536417723,-1.2533981800,-2.5627088547,1.1013879776,1.3389041424,0.0129502146,-0.0572521426,0.0566614270,-0.1061283350,0.2756666541,5.6841883659,-5.0411558151,-0.1254496425,3.5076298714,-0.0363937616,0.1194875911,-0.1604546756,-1.2280062437,4.0034294128,0.9304835200,2.1332948208,2.4703264236,0.0866401941},
		{0.0603999458,0.1480410844,0.2204557806,0.2248867452,0.2320095152,1.3512248993,0.0319299921,0.0648632795,0.0929326788,0.1090108752,-0.2271863520,1.5948387384,0.8893073201,-1.3688926697,-0.4080530405,0.1592775285,0.0824648291,-0.9192004800,-0.0592565835,0.4190350771,0.7929465175,0.0355434902,-0.6831963658,0.1084345430,0.0245063640,-0.8137071133,-0.2484051287,1.8210595846,0.2499032617,-2.1080269814,-1.0479021072,0.1923886389,0.1014168188,-1.1286523342,-1.4824903011,0.2363986522,-0.6173895001,-1.9800792933,-0.8443037868,0.3696690202,0.0647106841,-1.2896862030,-2.8530235291,-1.0815734863,-0.1762146950,1.6311353445,-0.4812989235,0.3407555521,0.0705282763,-1.1740239859,-0.8927523494,0.2576379180,1.7877017260,1.5230132341,-1.2311445475,0.3555247188,0.0931997150,0.2289366424,1.0339896679,-0.1986891478,-0.9963364601,-1.5553503036,0.1536559910,0.2129922807},
		{0.0444463938,0.3045566082,0.0441513620,0.8775362372,0.6996621490,2.4080321789,0.1216963679,0.0756096095,0.0336581022,-0.5564237237,-1.0898340940,2.9420332909,0.2122963518,-1.3133382797,-1.7204040289,0.0976725072,0.0782952383,-0.6047321558,0.7287892103,-0.2802377045,-0.2839799225,-1.3500785828,-3.5423913002,0.0836606920,0.1228598878,-0.8613137007,-0.7939772010,3.3253512383,-0.6253499985,-2.0574109554,-3.5199556351,0.2157480121,0.1177175194,0.9846104980,0.7469834089,0.9208371639,-0.5154463053,-3.1649985313,-3.0433869362,0.3900827169,0.1221579611,0.3648737669,-0.0458848365,0.0863328874,-1.3523625135,1.5180633068,-0.0752127469,0.4259355068,0.0480094291,-2.1819531918,-4.1028780937,1.5974575281,2.9680714607,0.6321669221,-0.4916414618,0.4011505246,0.0439352095,-0.0952026993,-0.2629151642,-0.6038938761,1.5485219955,1.0189462900,1.4386831522,0.2299031913},
		{0.1177671105,0.4719461501,-0.0094297323,0.0526065417,-0.1956310719,-1.9865964651,0.0833918303,0.0849635303,0.0730155334,0.4669470787,0.2087704837,1.5540541410,0.6064919829,0.9418329000,0.5553287268,0.1250000000,0.1032288596,-0.6513032317,-1.3881337643,0.4992741942,1.7977168560,0.0365655087,0.8763507605,0.0557845049,0.0505539104,-0.8262526393,0.1879277080,-1.0850272179,3.0148487091,-0.7690313458,0.3836478889,0.1079901606,0.0403073207,-1.4714074135,-2.7539672852,-2.2259364128,-0.3137337863,-1.6366512775,0.9165482521,0.1259790361,0.0468802452,-1.6007207632,-1.1490691900,0.2098260671,0.1442246437,-2.8782622814,-0.1992293000,0.0921245590,0.0377704687,-0.6736344695,0.4348546565,3.9409139156,-1.4662518501,0.0400763303,-0.7032508254,0.0639750361,0.0707762390,0.3093293905,0.5788117051,3.1369237900,-0.5814247131,-0.9495973587,-0.5011208653,0.0944860354},
		{0.0318346210,-0.6824984550,-1.2860394716,-1.1004599333,-1.0691033602,-1.5368250608,0.0025788140,0.0046731466,0.0583933517,0.2996366620,-1.2789797783,1.1868287325,1.7599546909,0.5247182846,0.5016676188,-0.0011310857,0.0748542771,0.1989520043,-1.8961572647,-0.6321940422,-1.1660673618,1.3684847355,1.3616666794,-0.0106680999,0.1215285212,0.2447604090,-1.9061206579,0.9214422703,-0.4216948152,0.8910884261,1.9208997488,-0.0554203019,0.1119037420,1.1018857956,0.4129649103,-0.0032569773,0.0697198436,1.2708295584,-0.9897751212,-0.2952001989,0.0099032568,2.0974607468,3.3045873642,-0.0026113933,-0.3663848042,-3.4077970982,-0.3133615851,-0.2438556403,0.0227591787,3.8123965263,3.9910194874,0.4748217463,0.1976723373,-3.5230059624,1.6251168251,-0.3528831601,0.0594805740,0.6695393920,-1.8116531372,0.5147937536,3.5168380737,3.4693844318,0.1670868397,-0.2218877822},
		{0.0912999362,-0.0759699643,-1.1545557976,-0.7233441472,0.1476226598,0.0210877173,0.0113948481,0.0860011578,0.0939550474,-0.2282299846,-2.4769642353,1.4375460148,0.8117845058,-0.1905983686,-1.1269639730,0.0940221772,0.0421040989,0.6755397916,-0.8063930869,-0.2967829704,-0.6153559685,1.3887417316,-0.6756513715,0.0770311803,0.0832621232,0.5392810106,-1.9224377871,-1.5749773979,1.2187813520,-0.1304886639,0.8146696091,0.0855193883,0.0774101987,1.7366782427,0.4335124493,-2.5125985146,-2.1543200016,-1.0365236998,-0.9636009932,0.1466496438,0.0652867183,1.4422669411,0.6850469708,-2.4375534058,-3.1161181927,-1.1397885084,0.0850511044,0.1076703593,0.0647412017,0.4252900183,1.5252016783,1.8092608452,2.6079285145,0.1200590208,0.3083869219,0.0372152925,0.0702726841,-0.0104277628,-0.5144811273,-1.0237801075,2.9104037285,3.2126858234,1.4936749935,0.0514714085},
		{0.0167012550,-1.4918836355,-2.6034970284,-0.8807770014,-1.4974354506,-2.0257823467,0.0105174417,0.0652638301,0.0074732201,0.5412222743,1.1087839603,0.1134585738,-1.5337597132,-1.0559382439,1.9001694918,0.1209287420,0.1124149263,-1.5918996334,-1.7567219734,-0.8712399006,-1.8185462952,3.0719554424,5.5729355812,0.0813843831,0.0524460599,-0.0619193166,-3.2127513885,1.8802670240,0.6738877296,-0.8110777736,4.0261964798,-0.0440603048,0.0399563573,-1.7639992237,-2.0075097084,-1.6228590012,-2.1415400505,2.9367351532,5.5770707130,-0.1845607609,0.1240348518,-0.7107864022,-1.6314262152,-0.6800912619,-0.6969009638,-0.8084068894,-0.0231788736,-0.2256692499,0.0512710959,1.1306723356,0.0578280278,-2.2921447754,0.6747531891,2.9530658722,-1.9044407606,-0.2214576900,0.0172620323,2.5340983868,4.7162518501,4.2306942940,1.1091431379,-2.2350258827,-2.4182610512,-0.1510774493},
		{0.0422719494,-0.1548732072,-0.7825585008,-0.5540211797,0.6515002847,1.8163938522,0.0003738517,0.0861766413,0.0163502917,-1.1291278601,-1.8550955057,1.2292280197,-0.1646482497,-1.4426010847,-1.5217363834,0.0676182806,0.0820909739,0.2039087415,0.9826603532,0.2580947280,-0.8305925727,1.2262918949,-2.4721403122,0.1007346660,0.0551087968,0.1270838678,-0.8141337633,0.3419178724,0.2817242146,-0.0048989733,-0.8918387890,0.0939950645,0.0149082923,1.5580220222,1.3178024292,-2.2450942993,-2.3337500095,-2.1393117905,-2.3927099705,0.2517455816,0.0192152169,0.8686923981,0.1291248053,-3.4729979038,-3.6428551674,1.0644478798,-0.2976108491,0.3220500648,0.0214545131,-0.9122018814,-0.0908198357,2.1154782772,3.9986944199,0.9153431058,0.0900968835,0.3276729882,0.1163747087,-0.5271445513,-0.5830981135,-2.0928356647,1.2975499630,1.5939625502,2.0501468182,0.2509024739}
	};
	double  B1[IMPLICITDEM]=
{-3.6446194649,2.3408379555,2.5178689957,1.2957017422,-2.4612243176,1.4999641180,-0.3973942697,2.5792882442}
	;
//	double  A1[IMPLICITDEM];
	
	double  W2[OUTPUTDEM][IMPLICITDEM]={
		{-3.2851488590,6.0637249947,4.8293242455,2.8874158859,-12.0857019424,-5.9305477142,3.9253387451,-1.8288255930},
		{21.8363437653,-3.6658129692,1.8995436430,5.4405388832,-4.5799202919,-1.1089316607,-8.8693828583,-1.0868173838},
		{1.6765117645,2.9207370281,11.9105434418,-6.7630372047,-6.6905193329,1.9508867264,-12.4216060638,7.5955314636},
		{-3.0931060314,-5.6453251839,-3.0081601143,-6.8207769394,0.7892704010,-8.9540634155,1.5338423252,-7.4545121193}
	};
	double  B2[OUTPUTDEM]=
	{2.2202692032,-3.2683863640,-1.5379123688,7.4592680931}
;
	InitBPParameters_EX(INPUTDEM,IMPLICITDEM,OUTPUTDEM,(double **)W1,B1,(double **)W2,B2);
	SaveBPParameters("C:\\Type1_Paras.dat");
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
