
// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the ANNRECOGNITION_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// ANNRECOGNITION_API functions as being imported from a DLL, wheras this DLL sees symbols
// defined with this macro as being exported.
#ifdef ANNRECOGNITION_EXPORTS
#define ANNRECOGNITION_API extern "C" __declspec(dllexport)
#else
#define ANNRECOGNITION_API extern "C" __declspec(dllimport)
#endif

#include <iostream>
#include <deque>
#include <io.h>
#include <errno.h>

using namespace std;

#ifndef _WIN32_IE
	#define _WIN32_IE 0x0500
	#include <windef.h>
	#include <atlbase.h>
	#include <atlapp.h>
	#include <atlgdi.h>
#endif

#include <math.h>
#include <direct.h>
#include <exception>
#include <fstream>

using namespace std;

/*
 * Dib�ļ�ͷ��־���ַ���"BM"��дDIBʱ�õ��ó�����
 */
#define DIB_HEADER_MARKER   ((WORD) ('M' << 8) | 'B')

// DIB���
DECLARE_HANDLE(HDIB);

// DIB����
#define PALVERSION   0x300

/* DIB�� */

// �ж��Ƿ���Win 3.0��DIB
#define IS_WIN30_DIB(lpbi)  ((*(LPDWORD)(lpbi)) == sizeof(BITMAPINFOHEADER))

// �����������Ŀ��
#define RECTWIDTH(lpRect)     ((lpRect)->right - (lpRect)->left)

// �����������ĸ߶�
#define RECTHEIGHT(lpRect)    ((lpRect)->bottom - (lpRect)->top)

// �ڼ���ͼ���Сʱ�����ù�ʽ��biSizeImage = biWidth' �� biHeight��
// ��biWidth'��������biWidth�������biWidth'������4������������ʾ
// ���ڻ����biWidth�ģ���4�������������WIDTHBYTES������������
// biWidth'
#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)

// Logmessage Level

#define  LOG_ERROR 0x0000
#define  LOG_INFO 0x0010
#define  LOG_DEBUG 0x0100

//�汾��Ϣ
ANNRECOGNITION_API int		 ANNRecognitionVersion(void);
ANNRECOGNITION_API void		 ANNRecognitionLog(LPSTR message,int logType);

/************************************************************************/
/*   DIBAPI ����ԭ��                                                    */
/************************************************************************/

// ************************************************************************
//  �ļ�����dibapi.cpp
//
//  ���� DIB(Independent Bitmap) API�����⣺
//

//  PaintDIB()          - ����DIB����
ANNRECOGNITION_API BOOL      PaintDIB (HDC, LPRECT, HDIB, LPRECT, CPalette * pPal);
//  FindDIBBits()       - ����DIBͼ��������ʼλ��
ANNRECOGNITION_API LPSTR     FindDIBBits (LPSTR lpbi);
//  DIBWidth()          - ����DIB���
ANNRECOGNITION_API DWORD     DIBWidth (LPSTR lpDIB);
//  DIBHeight()         - ����DIB�߶�
ANNRECOGNITION_API DWORD     DIBHeight (LPSTR lpDIB);
//  DIBNumColors()      - ����DIB��ɫ����ɫ��Ŀ
ANNRECOGNITION_API WORD      DIBNumColors (LPSTR lpbi);
//  DIBBitCount()      - ����DIB��ɫ��Bit��Ŀ
ANNRECOGNITION_API WORD	     DIBBitCount(LPSTR lpbi);
//  CopyHandle()        - �����ڴ��
ANNRECOGNITION_API HGLOBAL   CopyHandle (HGLOBAL h);
//  NewDIB()            - �����ṩ�Ŀ��ߡ���ɫλ��������һ���µ�DIB
ANNRECOGNITION_API HDIB	     NewDIB(long width, long height,unsigned short biBitCount);
//  SaveDIB()           - ��DIB���浽ָ���ļ���
ANNRECOGNITION_API BOOL      SaveDIB (HDIB hDib, LPSTR file);
//  ReadDIBFile()       - ��ָ���ļ��ж�ȡDIB����
ANNRECOGNITION_API HDIB      ReadDIBFile(LPCSTR file);
//  PaletteSize()       - ����DIB��ɫ���С
ANNRECOGNITION_API WORD		 PaletteSize(LPSTR lpbi);
//  Release DIB File	- �ͷ�DIB�ռ�
ANNRECOGNITION_API BOOL		 ReleaseDIBFile(HDIB hDib);

/************************************************************************/
/*ͼ����                                                              */
/************************************************************************/

//�����Ļ
ANNRECOGNITION_API void		 ClearAll(HDC pDC);
//����Ļ����ʾλͼ
ANNRECOGNITION_API void		 DisplayDIB(HDC pDC,HDIB hDIB);
//�Էָ���λͼ���гߴ��׼��һ��
ANNRECOGNITION_API void		 StdDIBbyRect(HDIB hDIB, LONG charRectID,int tarWidth, int tarHeight);
//������ͼ����й�һ��
ANNRECOGNITION_API void		 StdDIB(HDIB hDIB,int tarWidth, int tarHeight);
//����б�ʵ���
ANNRECOGNITION_API void		 SlopeAdjust(HDIB hDIB);
//ȥ����ɢ������
ANNRECOGNITION_API void		 RemoveScatterNoise(HDIB hDIB);
//�ݶ���
ANNRECOGNITION_API void		 GradientSharp(HDIB hDIB);
//����
ANNRECOGNITION_API void		 DrawFrame(HDC pDC,HDIB hDIB, LONG charRectID,unsigned int linewidth,COLORREF color);
//���Ҷ�ͼ��ֵ��
ANNRECOGNITION_API void		 ConvertGrayToWhiteBlack(HDIB hDIB);
//��256ɫλͼתΪ�Ҷ�ͼ
ANNRECOGNITION_API void		 Convert256toGray(HDIB hDIB);
//ϸ��
ANNRECOGNITION_API void		 Thinning(HDIB hDIB);
//��λͼ���зָ�.����һ���洢��ÿ��ָ����������
ANNRECOGNITION_API LONG		 CharSegment(HDIB hDIB);
//���������ŵ���
ANNRECOGNITION_API HDIB		 AutoAlign(HDIB hDIB,LONG charRectID);
//�ж��Ƿ�����ɢ������
ANNRECOGNITION_API bool		 DeleteScaterJudge(LPSTR lpDIBBits,WORD lLineBytes, LPBYTE lplab, int lWidth, int lHeight, int x, int y, POINT lab[], int lianXuShu);
//��ͼ�����ģ�����
ANNRECOGNITION_API HDIB		 Template(HDIB hDIB,double * tem ,int tem_w,int tem_h,double xishu);
//��ͼ�������ֵ�˲�
ANNRECOGNITION_API HDIB		 MidFilter(HDIB hDIB,int tem_w,int tem_h);
//��ͼ�����ֱ��ͼ����
ANNRECOGNITION_API void		 Equalize(HDIB hDIB);
//��ȡͼ����ʶ��������
ANNRECOGNITION_API LONG		 GetSegmentCount(LONG charRectID);
//������ʱ���
ANNRECOGNITION_API void		 SaveSegment(HDIB hDIB,LONG charRectID,LPSTR destFolder);

/************************************************************************/
/* OCR Helper                                                            */
/************************************************************************/

enum MiLANGUAGES 
{
	miLANG_CHINESE_SIMPLIFIED = 2052,
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
{
	miFILE_FORMAT_DEFAULTVALUE = -1,
	miFILE_FORMAT_TIFF = 1,
	miFILE_FORMAT_TIFF_LOSSLESS = 2, 
	miFILE_FORMAT_MDI = 4
};
enum MiCOMP_LEVEL 
{
	miCOMP_LEVEL_LOW = 0,
	miCOMP_LEVEL_MEDIUM = 1, 
	miCOMP_LEVEL_HIGH = 2
};

ANNRECOGNITION_API BOOL      ConvertJPEG2BMP(LPSTR jpegFile , LPSTR bmpFile);
ANNRECOGNITION_API BOOL      ConvertBMP2TIF(LPSTR bmpFile , LPSTR tifFile);
ANNRECOGNITION_API BOOL      BlackWhiteBMP(LPSTR bmpFile,int threshold);
ANNRECOGNITION_API BOOL      RevertBlackWhiteBMP(LPSTR bmpFile);
ANNRECOGNITION_API BOOL      SaveBlockToBMP(LPSTR bmpFile,double leftRate,double topRate, double rightRate, double bottomRate,LPSTR bmpBlock);
ANNRECOGNITION_API BOOL      SaveBlockToBMP2(LPSTR bmpFile,long left,long top, long right, long bottom,LPSTR bmpBlock);
ANNRECOGNITION_API BOOL		 IsOCRAvailable();
ANNRECOGNITION_API LONG		 GetOCRLanguage();
ANNRECOGNITION_API void		 SetOCRLanguage(LONG language);
ANNRECOGNITION_API void		 SetWithAutoRotation(BOOL isUse);
ANNRECOGNITION_API void		 SetWithStraightenImage(BOOL isUse);
ANNRECOGNITION_API BOOL		 OCRFile(LPSTR fileName,LPSTR * content);

/************************************************************************/
/* BP ������ӿ�                                                      */
/************************************************************************/

ANNRECOGNITION_API BOOL		 LoadBPParameters(LPSTR settingFile);
ANNRECOGNITION_API BOOL		 SaveBPParameters(LPSTR settingFile);
ANNRECOGNITION_API BOOL		 PrintBPParameters(LPSTR textFile);

ANNRECOGNITION_API BOOL      InitTrainBPRandSeed(long seed=-1);
ANNRECOGNITION_API BOOL      InitTrainBPLearnSpeed(double dblSpeed=0.05);
ANNRECOGNITION_API BOOL      InitTrainBPWeights(double* difWeights=0);

ANNRECOGNITION_API BOOL		 InitBPParameters_EX(LONG input,LONG implicit,LONG output,double ** w1=0,double *b1=0,double **w2=0,double *b2=0);
ANNRECOGNITION_API BOOL		 InitBPParameters(LONG input,LONG implicit,LONG output);

ANNRECOGNITION_API double    GetLearningSpeed();
ANNRECOGNITION_API double    Training(double *input,double * dest);
ANNRECOGNITION_API double	 CheakDiffs(double *output,double * dest);
ANNRECOGNITION_API BOOL		 Recognition(double *intput,double * result);
ANNRECOGNITION_API BOOL	     BPEncode(HDIB hInputDIB,double * outCode,LONG top=0, LONG left=0,LONG right=0, LONG bottom=0,LPSTR gridFile=0);
