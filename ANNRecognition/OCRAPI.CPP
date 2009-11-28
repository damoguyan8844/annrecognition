#include "stdafx.h"
#include "../ANNRecognition.h"
#include <algorithm>
#include <windef.h>

#include <atlbase.h>
#include <atlapp.h>
#include <atlgdi.h>
#include <tiffio.h>
#include <tiffiop.h>
#include <ximage.h>
#include <ximatif.h>
#include <MODIVWCTL.h>

using namespace std;

MiLANGUAGES G_MILANGUAGE=miLANG_CHINESE_SIMPLIFIED;
BOOL G_WithAutoRotation=FALSE;
BOOL G_WithStraighten=FALSE;

ANNRECOGNITION_API BOOL ConvertJPEG2BMP(LPSTR jpegFile , LPSTR bmpFile)
{
	CxImage cxImage; 
	cxImage.Load(jpegFile,CXIMAGE_FORMAT_JPG); 
	cxImage.Save(bmpFile,CXIMAGE_FORMAT_BMP);
	
	return TRUE;
}
ANNRECOGNITION_API BOOL ConvertBMP2TIF(LPSTR bmpFile , LPSTR tifFile)
{
 	CxImage image; 
 	image.Load(bmpFile,CXIMAGE_FORMAT_BMP); 

	FILE* hFile;
 	hFile = fopen(tifFile,"w+b");
	
	if (!image.Encode(&CxIOFile(hFile),CXIMAGE_FORMAT_TIF)){
		ANNRecognitionLog((char *)image.GetLastError(),LOG_ERROR);
		fclose(hFile);
		return FALSE;
	}
	fclose(hFile);
	return TRUE;
}


ANNRECOGNITION_API BOOL BlackWhiteBMP( LPSTR bmpFile ,int threshold)
{
	CxImage cxImage; 
	cxImage.Load(bmpFile,CXIMAGE_FORMAT_BMP);
	
	BOOL bRet=TRUE;

	cxImage.IncreaseBpp(8);


//	bRet=bRet&&cxImage.Negative();
	bRet=bRet&&cxImage.GrayScale();
	if(bRet)
	{
		bRet=bRet&&cxImage.Threshold(threshold);
	}

	cxImage.IncreaseBpp(8);
	cxImage.DecreaseBpp(8,false);

	cxImage.Save(bmpFile,CXIMAGE_FORMAT_BMP);
	return bRet;
}

ANNRECOGNITION_API BOOL RevertBlackWhiteBMP( LPSTR bmpFile )
{
	CxImage cxImage; 
	cxImage.Load(bmpFile,CXIMAGE_FORMAT_BMP);
	cxImage.Negative();
	cxImage.Save(bmpFile,CXIMAGE_FORMAT_BMP);

	return TRUE;
}

ANNRECOGNITION_API BOOL SaveBlockToBMP(LPSTR bmpFile,double leftRate,double topRate, double rightRate, double bottomRate,LPSTR bmpBlock)
{
	CxImage cxImage; 
	cxImage.Load(bmpFile,CXIMAGE_FORMAT_BMP);

	RECT rect;
	rect.left=leftRate* cxImage.GetWidth();
	rect.top=topRate* cxImage.GetHeight();
	rect.right = rightRate*cxImage.GetWidth();
	rect.bottom = bottomRate*cxImage.GetHeight();

	CxImage imgBlock;
	
	if(cxImage.Crop(rect,&imgBlock))
		return imgBlock.Save(bmpBlock,CXIMAGE_FORMAT_BMP);

	return FALSE;
}

ANNRECOGNITION_API BOOL SaveBlockToBMP2(LPSTR bmpFile,long left,long top, long right, long bottom,LPSTR bmpBlock)
{
	CxImage cxImage; 
	cxImage.Load(bmpFile,CXIMAGE_FORMAT_BMP);
	
	RECT rect;
	rect.left=left;
	rect.top=top;
	rect.right = right;
	rect.bottom = bottom;
	
	CxImage imgBlock;
	
	if(cxImage.Crop(rect,&imgBlock))
		return imgBlock.Save(bmpBlock,CXIMAGE_FORMAT_BMP);
	
	return FALSE;
}

ANNRECOGNITION_API BOOL SaveBlockToBMP3(HBITMAP hDib,long left,long top, long right, long bottom,LPSTR bmpBlock)
{
	CxImage cxImage; 
	cxImage.CreateFromHBITMAP(hDib);

	RECT rect;
	rect.left=left;
	rect.top=top;
	rect.right = right;
	rect.bottom = bottom;
	
	CxImage imgBlock;
	
	if(cxImage.Crop(rect,&imgBlock))
		return imgBlock.Save(bmpBlock,CXIMAGE_FORMAT_BMP);
	
	return FALSE;
}

ANNRECOGNITION_API BOOL	IsOCRAvailable()
{
	try	
	{
		IDocument *pDoc = new IDocument;
		if(pDoc!=NULL && pDoc->CreateDispatch("MODI.Document"))
		{
			return TRUE;
		}
	}
	catch(...){}
	return FALSE;
}
ANNRECOGNITION_API LONG	GetOCRLanguage()
{
	return G_MILANGUAGE;
}

ANNRECOGNITION_API void	SetOCRLanguage(LONG language)
{
	G_MILANGUAGE=MiLANGUAGES(language);
}
ANNRECOGNITION_API void	SetWithAutoRotation(BOOL isUse)
{
	G_WithAutoRotation=isUse;	
}
ANNRECOGNITION_API void	SetWithStraightenImage(BOOL isUse)
{
	G_WithStraighten=isUse;
}
ANNRECOGNITION_API BOOL	OCRFile(LPSTR fullFileName,LPSTR content)
{
	string strFileName=fullFileName;
	
	if(strFileName.length()<4) return FALSE;	
	
	string strType=strFileName.substr(strFileName.length()-3,3);

	ENUM_CXIMAGE_FORMATS type=CXIMAGE_FORMAT_BMP;
	if( stricmp(strType.c_str(),"tif")==0||
		stricmp(strType.c_str(),"tiff")==0) 
		type=CXIMAGE_FORMAT_TIF;
	else if( stricmp(strType.c_str(),"jpg")==0 ) 
		type=CXIMAGE_FORMAT_JPG;
	else if( stricmp(strType.c_str(),"bmp")==0 ) 
		type=CXIMAGE_FORMAT_BMP;
	else
		return FALSE;

	ComEnvSetter envSetter;
	
	string strOut;

	IDocument *pDoc = new IDocument;
	pDoc->CreateDispatch("MODI.Document");
	pDoc->Create(strFileName.c_str());
	pDoc->OCR( G_MILANGUAGE, G_WithAutoRotation, G_WithStraighten);
	IImages images = pDoc->GetImages();
	long	  num =images.GetCount();
	for( int i = 0; i < num; i++ )
	{
		IImage  image = images.GetItem(i);
		ILayout layout = image.GetLayout();
		strOut+=layout.GetText();
	}
	pDoc->Close(0);
	pDoc->ReleaseDispatch();
	delete pDoc;
	
	if(!strOut.empty())
	{
		//*content= new char[strOut.size()];
		strncpy(content,strOut.c_str(),strOut.size());
	}

	return (num > 0) ? TRUE : FALSE;
}