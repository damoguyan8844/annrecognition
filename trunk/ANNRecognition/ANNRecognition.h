
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
 * Dib文件头标志（字符串"BM"，写DIB时用到该常数）
 */
#define DIB_HEADER_MARKER   ((WORD) ('M' << 8) | 'B')

// DIB句柄
DECLARE_HANDLE(HDIB);

// DIB常量
#define PALVERSION   0x300

/* DIB宏 */

// 判断是否是Win 3.0的DIB
#define IS_WIN30_DIB(lpbi)  ((*(LPDWORD)(lpbi)) == sizeof(BITMAPINFOHEADER))

// 计算矩形区域的宽度
#define RECTWIDTH(lpRect)     ((lpRect)->right - (lpRect)->left)

// 计算矩形区域的高度
#define RECTHEIGHT(lpRect)    ((lpRect)->bottom - (lpRect)->top)

// 在计算图像大小时，采用公式：biSizeImage = biWidth' × biHeight。
// 是biWidth'，而不是biWidth，这里的biWidth'必须是4的整倍数，表示
// 大于或等于biWidth的，离4最近的整倍数。WIDTHBYTES就是用来计算
// biWidth'
#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)

// Logmessage Level

#define  LOG_ERROR 0x0000
#define  LOG_INFO 0x0010
#define  LOG_DEBUG 0x0100

//版本信息
ANNRECOGNITION_API int		 ANNRecognitionVersion(void);
ANNRECOGNITION_API void		 ANNRecognitionLog(LPSTR message,int logType);

/************************************************************************/
/*   DIBAPI 函数原型                                                    */
/************************************************************************/

// ************************************************************************
//  文件名：dibapi.cpp
//
//  公用 DIB(Independent Bitmap) API函数库：
//

//  PaintDIB()          - 绘制DIB对象
ANNRECOGNITION_API BOOL      PaintDIB (HDC, LPRECT, HDIB, LPRECT, CPalette * pPal);
//  FindDIBBits()       - 返回DIB图像象素起始位置
ANNRECOGNITION_API LPSTR     FindDIBBits (LPSTR lpbi);
//  DIBWidth()          - 返回DIB宽度
ANNRECOGNITION_API DWORD     DIBWidth (LPSTR lpDIB);
//  DIBHeight()         - 返回DIB高度
ANNRECOGNITION_API DWORD     DIBHeight (LPSTR lpDIB);
//  DIBNumColors()      - 计算DIB调色板颜色数目
ANNRECOGNITION_API WORD      DIBNumColors (LPSTR lpbi);
//  DIBBitCount()      - 计算DIB调色板Bit数目
ANNRECOGNITION_API WORD	     DIBBitCount(LPSTR lpbi);
//  CopyHandle()        - 拷贝内存块
ANNRECOGNITION_API HGLOBAL   CopyHandle (HGLOBAL h);
//  NewDIB()            - 根据提供的宽、高、颜色位数来创建一个新的DIB
ANNRECOGNITION_API HDIB	     NewDIB(long width, long height,unsigned short biBitCount);
//  SaveDIB()           - 将DIB保存到指定文件中
ANNRECOGNITION_API BOOL      SaveDIB (HDIB hDib, LPSTR file);
//  ReadDIBFile()       - 重指定文件中读取DIB对象
ANNRECOGNITION_API HDIB      ReadDIBFile(LPCSTR file);
//  PaletteSize()       - 返回DIB调色板大小
ANNRECOGNITION_API WORD		 PaletteSize(LPSTR lpbi);
//  Release DIB File	- 释放DIB空间
ANNRECOGNITION_API BOOL		 ReleaseDIBFile(HDIB hDib);

/************************************************************************/
/*图像处理                                                              */
/************************************************************************/

//清楚屏幕
ANNRECOGNITION_API void		 ClearAll(HDC pDC);
//在屏幕上显示位图
ANNRECOGNITION_API void		 DisplayDIB(HDC pDC,HDIB hDIB);
//对分割后的位图进行尺寸标准归一化
ANNRECOGNITION_API void		 StdDIBbyRect(HDIB hDIB, LONG charRectID,int tarWidth, int tarHeight);
//对整个图像进行归一化
ANNRECOGNITION_API void		 StdDIB(HDIB hDIB,int tarWidth, int tarHeight);
//整体斜率调整
ANNRECOGNITION_API void		 SlopeAdjust(HDIB hDIB);
//去除离散噪声点
ANNRECOGNITION_API void		 RemoveScatterNoise(HDIB hDIB);
//梯度锐化
ANNRECOGNITION_API void		 GradientSharp(HDIB hDIB);
//画框
ANNRECOGNITION_API void		 DrawFrame(HDC pDC,HDIB hDIB, LONG charRectID,unsigned int linewidth,COLORREF color);
//将灰度图二值化
ANNRECOGNITION_API void		 ConvertGrayToWhiteBlack(HDIB hDIB);
//将256色位图转为灰度图
ANNRECOGNITION_API void		 Convert256toGray(HDIB hDIB);
//细化
ANNRECOGNITION_API void		 Thinning(HDIB hDIB);
//对位图进行分割.返回一个存储着每块分割区域的链表
ANNRECOGNITION_API LONG		 CharSegment(HDIB hDIB);
//紧缩、重排调整
ANNRECOGNITION_API HDIB		 AutoAlign(HDIB hDIB,LONG charRectID);
//判断是否是离散噪声点
ANNRECOGNITION_API bool		 DeleteScaterJudge(LPSTR lpDIBBits,WORD lLineBytes, LPBYTE lplab, int lWidth, int lHeight, int x, int y, POINT lab[], int lianXuShu);
//对图像进行模板操作
ANNRECOGNITION_API HDIB		 Template(HDIB hDIB,double * tem ,int tem_w,int tem_h,double xishu);
//对图像进行中值滤波
ANNRECOGNITION_API HDIB		 MidFilter(HDIB hDIB,int tem_w,int tem_h);
//对图像进行直方图均衡
ANNRECOGNITION_API void		 Equalize(HDIB hDIB);
//获取图像中识别对象个数
ANNRECOGNITION_API LONG		 GetSegmentCount(LONG charRectID);
//保存临时结果
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
/* BP 神经网络接口                                                      */
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
