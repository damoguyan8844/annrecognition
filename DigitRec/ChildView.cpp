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


//???????????
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

//??256?????
void CChildView::OnFileOpenBmp() 
{
	//???????????,??????????
	static char BASED_CODE szFilter[] = "256?????(*.bmp)|";
	CFileDialog dlg(TRUE,NULL,NULL,OFN_HIDEREADONLY|OFN_OVERWRITEPROMPT,szFilter,NULL);
    if(dlg.DoModal() == IDOK)
	   strPathName = dlg.GetPathName();
	else return;
	//????????
   	//CFile file;
	//?????????
	//file.Open (strPathName.c_str(),CFile::modeRead);
	//?????HDIB???. ??:???????????????????,?????
	m_hDIB=::ReadDIBFile (strPathName.c_str());
	//HDIB??: ??????????????????
	//HDIB????:?????????(?????)?DIB????
	//????
	//file.Close ();
	//??DIB???(???????)
	BYTE* lpDIB=(BYTE*)::GlobalLock ((HGLOBAL)m_hDIB);
	// ??DIB??????????
	WORD wNumColors;	
	wNumColors = ::DIBNumColors((char*)lpDIB);	
	// ?????256???
	if (wNumColors != 256)
	{
		// ????
		MessageBox("?256???!", "????" , MB_ICONINFORMATION | MB_OK);
		// ????
		::GlobalUnlock((HGLOBAL)m_hDIB);
		// ??
		return;
	}
	//????????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
	//??????????????
	fileloaded=true;
    //gyhinfoinput=false;          //2004.4.26??
	gyhfinished=false;
}

//??????,????????
void CChildView::OnFileReLoadBmp() 
{
	//??????????????????,??????????
	if(fileloaded==false)
	{
		OnFileOpenBmp();
		if(fileloaded==false)
			return;
	}
	//????????
//   	CFile file;
	//?????????
//	file.Open (strPathName,CFile::modeReadWrite);
	m_hDIB=::ReadDIBFile (strPathName.c_str());
	//????
//	file.Close ();
	//????????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);		
}


void CChildView::OnFileSaveBmp() 
{
	//???????????,??????????
	static char BASED_CODE szFilter[] = "256?????(*.bmp)|";
	CFileDialog dlg(FALSE,NULL,NULL,OFN_HIDEREADONLY|OFN_OVERWRITEPROMPT,szFilter,NULL);
    if(dlg.DoModal() == IDOK)
	   strPathNameSave =dlg.GetPathName();
	else return;

	//???????.bmp??
	//strPathNameSave+=".bmp";
	//???????????????????,????
// 	CFile file(strPathNameSave, CFile::modeReadWrite|CFile::modeCreate);
	
	::SaveDIB (m_hDIB,(char *)strPathNameSave.c_str());
	//????
//	file.Close ();	
}

//??????
void CChildView::OnImgprcAll() 
{
	if(fileloaded==false)
	{
		if(::AfxMessageBox ("????????????????!",MB_YESNO|MB_ICONSTOP)==IDNO)
		   return;
	}
	//????
	OnFileReLoadBmp();
	//???????????????????
	if(gyhinfoinput==false) OnInputGuiyihuaInfo();
	//?256????????
	OnIMGPRC256ToGray();
	//???????
	OnIMGPRCGrayToWhiteBlack();
	//????
	//OnImgprcSharp();
	//????????
	OnImgprcRemoveNoise();
	//???????????
	OnImgprcAdjustSlope();
	//???????
	OnImgprcDivide();
	//???????????????,???????BP???????
	OnImgprcStandarize();
	ConvertGrayToWhiteBlack(m_hDIB);
	//????????
	OnImgprcShrinkAlign();
	//?????????????????????????bmp??,????????
	//OnImgprcToDibAndSave();
	//OnPreprocThin();	
}

//??????1?:?256??????????
void CChildView::OnIMGPRC256ToGray() 
{	
	Convert256toGray(m_hDIB);	
	//????????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
}

//??????2?:???????
void CChildView::OnIMGPRCGrayToWhiteBlack()
{
	ConvertGrayToWhiteBlack(m_hDIB);
	//????????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
}

//??????3?:????
void CChildView::OnImgprcSharp() 
{
	GradientSharp(m_hDIB);
	//????????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);		
}

//??????4?:???????
void CChildView::OnImgprcRemoveNoise() 
{
	RemoveScatterNoise(m_hDIB);
	//????????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
}

//??????5?:?????
void CChildView::OnImgprcAdjustSlope() 
{
    SlopeAdjust(m_hDIB);
	//????????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
}

//??????6?:??,????????????????
void CChildView::OnImgprcDivide() 
{
	m_charRectID=CharSegment(m_hDIB);
	//????????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
	DrawFrame(pDC->m_hDC,m_hDIB,m_charRectID,2,RGB(20,60,200));
}

//??????7?:?????
//???????????????????????
void CChildView::OnImgprcStandarize() 
{
	StdDIBbyRect(m_hDIB,m_charRectID,w_sample,h_sample);
	

	//????????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
	DrawFrame(pDC->m_hDC,m_hDIB,m_charRectID,2,RGB(21,255,25));
	gyhfinished=true;
}

//??????8?:???????????????,?????????
void CChildView::OnImgprcShrinkAlign() 
{
	m_hDIB=AutoAlign(m_hDIB,m_charRectID);
	//????????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);	
	DrawFrame(pDC->m_hDC,m_hDIB,m_charRectID,1,RGB(252,115,27));
}

//??????9?:???????????????????HDIB??,???.bmp??
void CChildView::OnImgprcToDibAndSave() 
{
	string strFolder=strPathName.substr(0,strPathName.find_last_of("\\"));

	SaveSegment(m_hDIB,m_charRectID,(char *)strFolder.c_str());
}

void CChildView::OnImgprcThinning() 
{
	 Thinning(m_hDIB);	
	//????????
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
	//?????????????
	if(gyhfinished==false)
	{
		//?????????????
		::MessageBox(NULL,"??????????",NULL,MB_ICONSTOP);
		return;
	}
	
	//??BP?????????
	
	CDBpParamater BpPa;
	
	//?????
	BpPa.m_a=0;
	BpPa.m_eta=0.015;
	BpPa.m_ex=0.001;
	BpPa.m_hn=10;
	
	// ?????
	if(BpPa.DoModal()!=IDOK)
	{
		//??
		return;
	}
	//????????
	
	//????
	double  momentum=BpPa.m_a; 
	//??????
	double  min_ex=BpPa.m_ex; 
	//????
	int  n_hidden=BpPa.m_hn; 
	//????
	double eta=BpPa.m_eta;
	
	int digicount=GetSegmentCount(m_charRectID);
	
	//????DIB???
	BYTE *lpDIB=(BYTE*)::GlobalLock((HGLOBAL) m_hDIB);
	
	//????DIB?????,??????????
	BYTE *lpDIBBits =(BYTE*)::FindDIBBits((char *)lpDIB);
	
	//??????
	int numColors=(int) ::DIBNumColors((char *)lpDIB);
	//???????
    if (numColors!=256) 
	{
		::GlobalUnlock((HGLOBAL) m_hDIB);
		::MessageBox(NULL,"????????",NULL,MB_ICONSTOP);
		return;
	}
	
	//???????
    LONG lWidth = (LONG) ::DIBWidth((char *)lpDIB); 
	
	//???????
	LONG lHeight = (LONG) ::DIBHeight((char *)lpDIB);
	
	//??????????
	LONG lLineByte = (lWidth+3)/4*4; 
	
	//??????
	LONG lSwidth = w_sample;
	
	//??????
	LONG LSheight = h_sample;
	
	//??????????????  
	double **data_in;
	//???????????????
	data_in = code ( lpDIBBits, digicount,  lLineByte, lSwidth, LSheight);
	
	//??????????
	
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
// 	double  W1[IMPLICITDEM][INPUTDEM]={
// 		{0.1066393033,-3.2052233219,9.7285556793,-1.8455461264,1.0186408758,-1.5408438444,2.7950203419,6.3435359001,0.0299331248,-0.1025468186,-2.7309470177,3.6938765049,-2.0454607010,-0.4524833560,2.1562144756,2.6078166962,2.2797882557,-1.3546419144,7.1972179413,2.4898145199,-5.7449502945,-5.9540762901,-7.9935116768,-3.7096538544,-0.8291637897,3.8494873047,4.7657399178,-2.0403230190,-4.1985220909,-0.2918533385,1.0101605654,0.7796560526,0.0881556496,10.2956132889,2.8108661175,-5.7627911568,-0.6721009016,-2.4625301361,7.0386352539,0.0437788032,0.4478905201,-9.1105308533,-3.8427855968,-5.7548828125,1.4509664774,-2.0327036381,-0.8845379353,-0.1244209483,0.2738425136,-8.1232614517,6.6034169197,4.4525103569,-3.6985476017,-2.2103986740,-1.8570427895,-1.1952803135,0.0933233649,-5.1421699524,-1.5546836853,-2.5110514164,-0.9905135036,-2.3662757874,-7.5514464378,-5.8571591377},
// 		{0.0337878056,1.6129671335,-2.8710832596,1.2701293230,-7.4879727364,-1.0618114471,3.4330551624,1.5117549896,-0.4010130167,-1.4094812870,5.3077378273,-0.8197857738,-1.0688602924,1.4099220037,-0.2004078180,0.7262368202,-1.0819879770,3.3144273758,2.1425166130,-1.9019464254,-0.2259994745,6.1263828278,5.0114741325,-0.7026527524,0.0474705398,-4.5709843636,-4.1792888641,-9.8118982315,-4.6330776215,-6.4398550987,-0.3497201502,0.0779402032,0.0686306208,8.6070089340,-1.1384557486,-3.4819293022,-1.0293254852,-4.9135289192,-9.2218542099,0.1246681139,-0.0464390926,-5.0641937256,6.4198966026,0.6558352113,-8.3407459259,2.4767165184,0.4115117192,0.6965102553,-0.3128885925,0.0259560551,3.1511042118,7.9979977608,8.0287370682,5.6208391190,-2.4019176960,9.1595201492,0.1045851782,-0.1102072746,1.4427648783,3.0653121471,3.7463767529,5.7652344704,2.6723716259,-0.5744327903},
// 		{0.0251854006,-0.2480383366,-0.5447093248,7.5243310928,1.8129006624,-4.8900976181,-2.6028220654,-1.1492097378,0.2053524256,-4.1352853775,-0.0014414958,-9.1153268814,5.1403403282,6.7582354546,6.2227025032,2.2694182396,0.1734057367,8.0852003098,2.5822069645,14.8411760330,-2.1220488548,2.8575227261,-2.9475347996,-5.3064022064,0.0982597768,6.7661824226,-0.3953982890,6.2900285721,4.4865808487,-1.6593840122,-9.6155805588,-0.4720501304,0.0840655565,5.0167427063,3.6915626526,-2.1711027622,-5.0981812477,-9.8536596298,2.9404428005,0.0433782451,0.2829898596,-4.0159854889,-7.6428937912,-13.3518457413,1.2132741213,6.9293746948,3.7751872540,-0.3862084448,0.4352482855,-2.0877439976,0.7243337035,-4.7543611526,-2.2840402126,5.9261665344,-0.7706043124,2.4000821114,0.0597931221,2.5012164116,-1.5974259377,2.8450815678,10.2675476074,-11.5372543335,-1.3814098835,4.2376770973},
// 		{0.0088885156,-0.2385535836,-2.9986126423,3.7189314365,-0.8832290769,9.1957902908,1.4783101082,-5.6567225456,0.0581717193,7.1918349266,7.2119512558,1.3868361712,7.0726132393,6.6304383278,-0.4016958177,-4.8587865829,0.1002760008,5.4828257561,-9.9149847031,0.6984826326,7.9451875687,5.4196710587,7.9362425804,0.4798666239,0.1173554957,-0.5011360049,-1.8904894590,-7.2086577415,1.5056868792,5.0117330551,-6.6299157143,0.1124832183,0.0108210621,0.7591616511,-4.6863760948,2.8118064404,-3.1681935787,-10.7632961273,-7.5585761070,0.0603007600,0.0499448255,-6.2656726837,-8.0694265366,2.2870557308,-9.8598651886,-6.2570285797,0.2861019671,0.0161624383,0.1575687528,0.5549848676,3.2207276821,-4.2240328789,4.4117984772,7.6889052391,-1.1868141890,0.0842565224,0.0154101811,-0.0178093240,4.0941772461,14.9937953949,3.1509001255,-3.0742540359,5.4926366806,2.2607901096},
// 		{0.0217101052,-0.6519173980,3.5760383606,2.0543088913,-1.0441411734,4.5138487816,1.9282532930,4.6892266273,0.6101500392,-1.0727148056,2.3023591042,2.6419599056,-0.6388912797,-0.5073794127,-0.6245261431,1.3710625172,1.2936300039,2.7012896538,4.1563029289,3.0606350899,-2.3078298569,-1.1777862310,-2.4312431812,-0.9040547609,-1.9929069281,-2.7273726463,1.2255871296,-1.2518823147,-1.3443149328,1.3160055876,-2.5777721405,0.0361872688,0.0190662220,1.0725984573,2.3440709114,1.4038900137,-2.4798915386,-2.3802337646,-1.3039101362,0.0740684196,0.0076083750,-11.9640779495,-10.5064573288,-5.3869805336,-4.1292524338,-0.0104837306,-0.4811347723,1.2432601452,0.0825185180,-0.1159191653,0.7707782388,-1.6872881651,0.4254575074,-2.1579058170,0.8685564399,4.7422957420,0.0610113591,-0.7172562480,-3.7074930668,2.7468771935,-0.6812590361,-0.0628426820,-1.7433657646,1.7338122129},
// 		{0.0775818676,3.1930820942,-1.9586421251,-0.6365030408,-3.0433266163,-1.2350363731,-2.6512670517,-2.2552387714,0.1251166463,-3.7033033371,-0.7713021040,-4.6726269722,-5.0554943085,-6.4980812073,3.2941801548,-0.5677752495,0.1045268625,-1.0761914253,8.2664718628,8.5253353119,-2.5150792599,3.2415645123,-3.7306501865,3.9515459538,0.0433816016,3.2841815948,2.9516379833,-2.0877385139,-4.2322640419,1.9070712328,-2.8209850788,-0.0174140669,0.0935897157,0.9252257347,4.1788234711,0.2275872678,-0.4593605697,-4.1304388046,-6.8421063423,0.0971747488,0.3152389228,-5.4694724083,8.9988183975,6.1600766182,-5.3248953819,-2.5115327835,-3.7821960449,-2.2310085297,0.2489126176,0.6127425432,1.2056961060,0.6340270042,10.7158622742,1.5607330799,3.2704081535,6.9349546432,0.0011323493,-4.2616472244,-0.8110842109,-0.9853667021,4.2584900856,5.1023039818,0.6121526361,13.0077371597},
// 		{0.0464835055,-10.5378952026,6.6121931076,-7.3487401009,3.5804464817,3.7311408520,-2.0460851192,4.2766561508,-15.8849935532,-5.6067409515,-4.9344019890,-5.6670517921,6.1382436752,-6.6384572983,2.1789133549,-0.9672573209,1.5146924257,3.3238124847,-4.1656942368,10.2691679001,0.1927372813,-7.1721291542,-5.7605295181,-1.1013379097,0.1136149466,-3.5579140186,3.2859773636,-4.0945014954,-3.9950129986,-1.7243734598,1.1084862947,0.1151971668,0.0930882171,11.8367481232,1.6172417402,-6.9889678955,1.5421375036,6.5292711258,9.0763092041,0.0830790102,0.0591514036,-6.3318581581,1.9286847115,12.6845169067,5.0264225006,-1.7928023338,-5.5876750946,0.3338997364,0.0374731012,-1.0104850531,-3.6594378948,3.1240675449,7.3007822037,-1.4237818718,5.7408013344,3.0431005955,0.0721508637,8.5124816895,1.5265477896,-3.0038290024,-1.0847791433,6.1853713989,-0.9030399919,-0.3724892139},
// 		{0.0879200101,5.3577523232,-0.4074254632,0.7770090103,2.6624429226,2.4389121532,6.9999809265,6.1666340828,-3.8396060467,-0.7517715096,-1.8393645287,6.1242938042,2.8780481815,-0.8404768109,2.4379386902,2.4430644512,0.8821008801,2.8019373417,-2.9892125130,-6.5949316025,-9.3574790955,-10.4852418900,-8.4574899673,-1.0859792233,0.0071426537,-6.6046080589,-7.8773446083,1.3622335196,1.1651852131,-2.4765446186,-14.4776563644,0.0890713930,0.0237086453,1.3026459217,2.4240071774,-1.6114990711,0.9333811998,-0.7795327306,2.7357668877,0.0219847709,-0.6493101716,-6.7161874771,-4.3167972565,-2.3339009285,-1.3805546761,-0.8054878116,0.6384908557,-0.1800450683,-3.3914513588,-3.7524240017,0.3608673513,0.9789974689,3.4849472046,2.0087122917,-0.0946141407,-2.5723519325,0.1461606920,7.2237997055,3.3442633152,6.2218599319,-3.1663477421,5.0611753464,0.4283160865,6.8234162331}
// 	};
// 	double  B1[IMPLICITDEM]=
// 	{4.3683695793,-10.9778852463,-4.6350030899,-2.5917737484,-2.5448243618,2.1170504093,-6.4326963425,5.3697748184}
// 	;
// //	double  A1[IMPLICITDEM];
// 	
// 	double  W2[OUTPUTDEM][IMPLICITDEM]={
// 		{-2.1710667610,-18.7646312714,9.5999431610,22.1566390991,37.7439346313,-37.0359840393,20.4523200989,-7.8048381805},
// 		{-1.7041602135,23.3337116241,-15.2318172455,-14.9594211578,-25.2761287689,-25.7408351898,-2.8155255318,27.4694976807},
// 		{22.2753391266,-19.3318176270,-25.8812732697,-35.1686706543,15.4128513336,-11.0953636169,1.7674894333,10.7124996185},
// 		{-32.2430229187,-16.4421882629,13.7800722122,16.6755332947,2.5663218498,15.2792520523,-37.6981430054,-17.4322338104}
// 	};
// 	double  B2[OUTPUTDEM]=
// 	{-1.6040316820,22.7821445465,40.1802215576,-21.8244800568};
// 
// 	InitBPParameters(INPUTDEM,IMPLICITDEM,OUTPUTDEM,(double **)W1,B1,(double **)W2,B2);
//	SaveBPParameters("C:\\001278.dat");

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
	LoadBPParameters("C:\\001278.dat");
//	Main();
//	PrintBPParameters("C:\\001278.txt");
//	test();
	
 	HDIB newDIB= ReadDIBFile("C:\\9.bmp");
 	double outCode[64];
 	BPEncode(newDIB,outCode);
 	double result[4];
	Recognition(outCode,result);

	// TODO: Add your command handler code here
	OnImgprcAll();
	//?????????????
	if(gyhfinished==false)
	{
		//?????????????
		::MessageBox(NULL,"??????????",NULL,MB_ICONSTOP);
		return;
	}
	//????DIB???
	BYTE *lpDIB=(BYTE*)::GlobalLock((HGLOBAL) m_hDIB);
	
	//????DIB?????,??????????
	BYTE *lpDIBBits =(BYTE*)::FindDIBBits((char *)lpDIB);
	
	//??????
	int numColors=(int) ::DIBNumColors((char *)lpDIB);
	//???????
    if (numColors!=256) 
	{
		::GlobalUnlock((HGLOBAL) m_hDIB);
		::MessageBox(NULL,"????256???",NULL,MB_ICONSTOP);
		return;
	}
	
	int digicount=GetSegmentCount(m_charRectID);

	//???????
    LONG lWidth = (LONG) ::DIBWidth((char *)lpDIB); 
	
	//???????
	LONG lHeight = (LONG) ::DIBHeight((char *)lpDIB);
	
	//??????????
	LONG lLineByte = (lWidth+3)/4*4; 
	
	//??????
	LONG lSwidth = w_sample;
	
	//??????
	LONG LSheight = h_sample;
	
	// ??????
	int n[3];
	if(r_num(n,"num")==false)
		return;
	//?????????
	int  n_in=n[0];
	//????????
	int  n_hidden=n[1];
	//?????????
	int  n_out=n[2];  
	
	//?????????????????????
	if(n_in!=lSwidth*LSheight)
	{
		//????????????
		::MessageBox(NULL,"???????????????",NULL,MB_ICONSTOP);
		return;
	}
	
	//??????????????  
	double **data_in;
	//???????????????
	data_in = code ( lpDIBBits, digicount,  lLineByte, lSwidth, LSheight);
	
	//?????????????
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
//??????
	double tem[9]={1,1,1,
		           1,1,1,
				   1,1,1};

    //??????
    double  xishu = 0.111111;   

    //??????
	m_hDIB =Template(m_hDIB,tem ,3,3, xishu);

	//????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);
	
}

void CChildView::OnGass() 
{
	// TODO: Add your command handler code here
   
	//??????
	double tem[9]={1,2,1,
		           2,4,2,
				   1,2,1};

    //??????
    double  xishu = 0.0625;   

    //??????
	m_hDIB =Template(m_hDIB,tem ,3,3, xishu);

	//????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);
}

void CChildView::OnMid() 
{
	// TODO: Add your command handler code here
	//??????
	m_hDIB =MidFilter(m_hDIB,3,3);

	//????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);
}

void CChildView::OnImgprcEqualize() 
{
	Equalize(m_hDIB);

	//????
	CDC* pDC=GetDC();
	DisplayDIB(pDC->m_hDC,m_hDIB);
}
