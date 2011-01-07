#include "stdafx.h"
#include "ANNRecognition.h"
#include "Lock.h"

#include <algorithm>
#include <windef.h>
#include <comdef.h>
#include <atlbase.h>
#include <atlapp.h>
#include <atlgdi.h>
#include <tiffio.h>
#include <tiffiop.h>
#include <ximage.h>
#include <ximatif.h>
#include <MODIVWCTL.h>
#include <sys/stat.h>

using namespace std;

MiLANGUAGES G_MILANGUAGE=miLANG_CHINESE_SIMPLIFIED;
BOOL G_WithAutoRotation=FALSE;
BOOL G_WithStraighten=FALSE;

BOOL G_OCRDocumentError=FALSE;

ANNRECOGNITION_API BOOL ConvertJPEG2BMP(LPSTR jpegFile , LPSTR bmpFile)
{
	CxImage cxImage; 
	return cxImage.Load(jpegFile,CXIMAGE_FORMAT_JPG) && cxImage.Save(bmpFile,CXIMAGE_FORMAT_BMP);
}
ANNRECOGNITION_API BOOL ConvertBMP2TIF(LPSTR bmpFile , LPSTR tifFile)
{
//	Lock lock(&_csOCR);

 	CxImage image; 
 	if(!image.Load(bmpFile,CXIMAGE_FORMAT_BMP))
		return FALSE; 

	FILE* hFile;
 	hFile = fopen(tifFile,"w+b");
	
	if (!image.Encode(&CxIOFile(hFile),CXIMAGE_FORMAT_TIF)){
		fclose(hFile);
		DeleteFile(tifFile);

		ANNRecognitionLog(LOG_ERROR,(char *)string(image.GetLastError()).c_str());
		return FALSE;
	}
	fclose(hFile);
	return TRUE;
}


ANNRECOGNITION_API BOOL BlackWhiteBMP( LPSTR bmpFile ,int threshold)
{
//	Lock lock(&_cs);
	CxImage cxImage; 
	if(!cxImage.Load(bmpFile,CXIMAGE_FORMAT_BMP))
		return FALSE; 
	
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
//	Lock lock(&_cs);

	CxImage cxImage; 
	if(!cxImage.Load(bmpFile,CXIMAGE_FORMAT_BMP))
		return FALSE; 

	cxImage.Negative();
	cxImage.Save(bmpFile,CXIMAGE_FORMAT_BMP);

	return TRUE;
}

ANNRECOGNITION_API BOOL SaveBlockToBMP(LPSTR bmpFile,double leftRate,double topRate, double rightRate, double bottomRate,LPSTR bmpBlock)
{
	CxImage cxImage; 
	try
	{
		if(!cxImage.Load(bmpFile,CXIMAGE_FORMAT_BMP))
			return FALSE; 

		RECT rect;
		rect.left=leftRate* cxImage.GetWidth();
		rect.top=topRate* cxImage.GetHeight();
		rect.right = rightRate*cxImage.GetWidth();
		rect.bottom = bottomRate*cxImage.GetHeight();

		CxImage imgBlock;
		
		if(cxImage.Crop(rect,&imgBlock))
			return imgBlock.Save(bmpBlock,CXIMAGE_FORMAT_BMP);
	}
	catch(...)
	{
		string err="Save Block To BMP  Error!";
		err+=cxImage.GetLastError();
		ANNRecognitionLog(LOG_ERROR,const_cast<char *>(err.c_str()));
	}
	return FALSE;
}

ANNRECOGNITION_API BOOL SaveBlockToBMP2(LPSTR bmpFile,long left,long top, long right, long bottom,LPSTR bmpBlock)
{
//	Lock lock(&_cs);
	CxImage cxImage; 
	try
	{
		if(!cxImage.Load(bmpFile,CXIMAGE_FORMAT_BMP))
			return FALSE; 

		RECT rect;
		rect.left=left;
		rect.top=top;
		rect.right = right;
		rect.bottom = bottom;
		
		CxImage imgBlock;
		
		if(cxImage.Crop(rect,&imgBlock))
			return imgBlock.Save(bmpBlock,CXIMAGE_FORMAT_BMP);
	}
	catch(...)
	{
		string err="Save Block To BMP 2 Error!";
		err+=cxImage.GetLastError();
		ANNRecognitionLog(LOG_ERROR,const_cast<char *>(err.c_str()));
	}
	return FALSE;
}

ANNRECOGNITION_API BOOL SaveBlockToBMP3(HBITMAP hDib,long left,long top, long right, long bottom,LPSTR bmpBlock)
{
//	Lock lock(&_cs);
	CxImage cxImage; 
	try
	{
		if(!cxImage.CreateFromHBITMAP(hDib))
			return FALSE;

		RECT rect;
		rect.left=left;
		rect.top=top;
		rect.right = right;
		rect.bottom = bottom;
		
		CxImage imgBlock;
		
		if(cxImage.Crop(rect,&imgBlock))
			return imgBlock.Save(bmpBlock,CXIMAGE_FORMAT_BMP);
	}
	catch(...)
	{
		string err="Save Block To BMP 3 Error!";
		err+=cxImage.GetLastError();
		ANNRecognitionLog(LOG_ERROR,const_cast<char *>(err.c_str()));
	}

	return FALSE;
}

ANNRECOGNITION_API BOOL SaveBlockToBMP4(HBITMAP hDib,long left,long top, long right, long bottom,LPSTR bmpBlock,long threshold,bool needRevert)
{
	//	Lock lock(&_cs);
	HDIB hDibBlock=0;
	BOOL bRet=TRUE;
	CxImage cxImage; 
	try
	{
		if(!cxImage.CreateFromHBITMAP(hDib))
			return FALSE;
		
		RECT rect;
		rect.left=left;
		rect.top=top;
		rect.right = right;
		rect.bottom = bottom;
		
		CxImage imgBlock;
		bRet=cxImage.Crop(rect,&imgBlock);
		if(bRet)
		{
			//Revert
			if(needRevert)
				imgBlock.Negative();
			
			//Black White
			imgBlock.IncreaseBpp(8);

			//	bRet=bRet&&cxImage.Negative();
			bRet=bRet&&imgBlock.GrayScale();
			if(bRet)
			{
				bRet=bRet&&imgBlock.Threshold(threshold);
			}
			
			imgBlock.IncreaseBpp(8);
			imgBlock.DecreaseBpp(8,false);
			
			if(bRet)
			{
				bRet=imgBlock.Save(bmpBlock,CXIMAGE_FORMAT_BMP);
				imgBlock=NULL;

				if(bRet)
				{
					hDibBlock=ReadDIBFile(bmpBlock);
					if(hDib!=NULL)
					{
						bRet=bRet && Convert256toGray(hDibBlock);
						bRet=bRet && ConvertGrayToWhiteBlack(hDibBlock);
						bRet=bRet && RemoveScatterNoise(hDibBlock);
						bRet=bRet && SaveDIB(hDibBlock,bmpBlock);
						ReleaseDIBFile(hDibBlock);
						hDibBlock=NULL;
					}
					else
					{
						bRet=FALSE;
					}
				}
			}
		}
	}
	catch(...)
	{
		if(hDibBlock!=NULL)
			ReleaseDIBFile(hDibBlock);
		bRet=FALSE;

		string err="Save Block To BMP 4 Error!";
		err+=cxImage.GetLastError();
		ANNRecognitionLog(LOG_ERROR,const_cast<char *>(err.c_str()));
	}
	try
	{
		if(!bRet)
		{
			struct stat fileStates;
			if(::stat(bmpBlock,&fileStates)==0)
				DeleteFile(bmpBlock);
		}
	}
	catch(...)
	{
		ANNRecognitionLog(LOG_ERROR,"DeleteFile Error");
	}
	return bRet;
}

ANNRECOGNITION_API BOOL RecognitionWhiteText ( HBITMAP hDib, long left, long top, long right, long bottom, 
								   long threshHold, LPSTR tifFileName, LPSTR data )
{
//	Lock lock(&_cs);

	CxImage cxImage;
	if(!cxImage.CreateFromHBITMAP(hDib))
		return FALSE;

	RECT rect;
	rect.left = left>=0 && left<cxImage.GetWidth()?left: 0;
	rect.top = top>=0 && top<cxImage.GetHeight()?top: 0;
	rect.right = right>rect.left && right<=cxImage.GetWidth()?right:cxImage.GetWidth();
	rect.bottom = bottom>rect.top && bottom<=cxImage.GetHeight()?bottom:cxImage.GetHeight();

	CxImage imgBlock;

	BOOL bRet=TRUE;
	bRet = bRet && cxImage.Crop( rect, &imgBlock );
	bRet = bRet && imgBlock.Negative(); 
	
	if(bRet==FALSE)
	{
		ANNRecognitionLog(LOG_ERROR," White TextCrop , Negative File Come To Failure");
		return FALSE;
	}

	imgBlock.IncreaseBpp(8);
	bRet=bRet&&imgBlock.GrayScale();
	if(bRet)
	{
		bRet=bRet&&imgBlock.Threshold(threshHold);
	}
	
	imgBlock.IncreaseBpp(8);
	imgBlock.DecreaseBpp(8,false);

	FILE* hFile;
	hFile = fopen(tifFileName,"w+b");
	if (!imgBlock.Encode(&CxIOFile(hFile),CXIMAGE_FORMAT_TIF)){
		ANNRecognitionLog(LOG_ERROR,(char *)imgBlock.GetLastError());
		fclose(hFile);
		return FALSE;
	}
	fclose(hFile);

	return OCRFile( tifFileName, data );
}

ANNRECOGNITION_API BOOL RecognitionBlackText ( HBITMAP hDib, long left, long top, long right, long bottom, 
											  long threshHold, LPSTR tifFileName, LPSTR data )
{
	//	Lock lock(&_cs);
	
	CxImage cxImage;
	if(!cxImage.CreateFromHBITMAP(hDib))
		return FALSE;

	RECT rect;
	rect.left = left>=0 && left<cxImage.GetWidth()?left: 0;
	rect.top = top>=0 && top<cxImage.GetHeight()?top: 0;
	rect.right = right>rect.left && right<=cxImage.GetWidth()?right:cxImage.GetWidth();
	rect.bottom = bottom>rect.top && bottom<=cxImage.GetHeight()?bottom:cxImage.GetHeight();
	
	CxImage imgBlock;
	
	BOOL bRet=TRUE;
	bRet = bRet && cxImage.Crop( rect, &imgBlock );
	
	if(bRet==FALSE)
	{
		ANNRecognitionLog(LOG_ERROR," Black Text Crop , Negative File Come To Failure");
		return FALSE;
	}
	
	imgBlock.IncreaseBpp(8);
	bRet=bRet&&imgBlock.GrayScale();
	if(bRet)
	{
		bRet=bRet&&imgBlock.Threshold(threshHold);
	}
	
	imgBlock.IncreaseBpp(8);
	imgBlock.DecreaseBpp(8,false);
	
	FILE* hFile;
	hFile = fopen(tifFileName,"w+b");
	if (!imgBlock.Encode(&CxIOFile(hFile),CXIMAGE_FORMAT_TIF)){
		ANNRecognitionLog(LOG_ERROR,(char *)imgBlock.GetLastError());
		fclose(hFile);
		return FALSE;
	}
	fclose(hFile);
	
	return OCRFile( tifFileName, data );
}

ANNRECOGNITION_API BOOL	IsOCRAvailable()
{
	try	
	{
		IDocument *pDoc = new IDocument;
		if(pDoc!=NULL && pDoc->CreateDispatch("MODI.Document"))
		{
			delete pDoc;
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
	Lock lock(&_csOCR);
	
	if( !content )
		return FALSE;
	content[ 0 ] = '\0';

	struct stat fileStates;
	if(::stat(fullFileName,&fileStates)!=0)
		return FALSE;

	string strFileName=fullFileName;
	
	if(strFileName.length()<4) return FALSE;	
	
	string strType=strFileName.substr(strFileName.length()-3,3);

	ENUM_CXIMAGE_FORMATS type=CXIMAGE_FORMAT_BMP;
	if( stricmp(strType.c_str(),"tif")==0||
		stricmp(strType.c_str(),"iff")==0) 
		type=CXIMAGE_FORMAT_TIF;
	else if( stricmp(strType.c_str(),"jpg")==0 ) 
		type=CXIMAGE_FORMAT_JPG;
	else if( stricmp(strType.c_str(),"bmp")==0 ) 
		type=CXIMAGE_FORMAT_BMP;
	else
		return FALSE;

	ComEnvSetter envSetter;
	
	string strOut;
	
	int debug=0;
	string debugMessage;

	IDocument *pDoc=0;
	try
	{
		G_OCRDocumentError=FALSE;

//		ANNRecognitionLog(LOG_INFO,"OCR File Start");

		debugMessage="OCRFile: ";
		debugMessage+=string(fullFileName);
//		ANNRecognitionLog(LOG_INFO,(char *)debugMessage.c_str());

		pDoc = new IDocument;
		pDoc->CreateDispatch("MODI.Document");
		
		debug=1;
		if(!G_OCRDocumentError)
		{
			pDoc->Create(strFileName.c_str());
//			ANNRecognitionLog(LOG_INFO,"After Create Doc");
		}
		debug=2;
		if(!G_OCRDocumentError)
		{
			pDoc->OCR( G_MILANGUAGE, G_WithAutoRotation, G_WithStraighten);
//			ANNRecognitionLog(LOG_INFO,"After Create OCR");
		}

		debug=3;
		if(!G_OCRDocumentError)
		{
			IImages images = pDoc->GetImages();
		
			if(images!=0)
			{
//				ANNRecognitionLog(LOG_INFO,"After Create GetImages");

				long num =images.GetCount();
				for( int i = 0; i < num; i++ )
				{
					IImage  image = images.GetItem(i);
					if(!G_OCRDocumentError && image!=0)
					{
						ILayout layout = image.GetLayout();
						if(!G_OCRDocumentError && layout!=0)
						{
							strOut+=layout.GetText();
						}
					}
				}
			}
		}
		pDoc->Close(0);
		pDoc->ReleaseDispatch();

		delete pDoc;
		pDoc=0;
		
		if(!strOut.empty())
		{
			//*content= new char[strOut.size()];
			strncpy(content,strOut.c_str(),strOut.size());
			content[strOut.size()]='\0';
		}
		
		if(strOut.empty())
		{
			ANNRecognitionLog(LOG_INFO,"OCRFile: Return False ");
			return FALSE;
		}
		else
			ANNRecognitionLog(LOG_INFO,"OCRFile: Return TRUE  ");
		return TRUE;
	}
	catch(_com_error & e)
	{
		char temp[1024];
		sprintf(temp,"Debug=%d,OCR File %s Failure,Source=%s,Message=%s",debug,strFileName.c_str(),(const char *)e.Source(),e.ErrorMessage()?e.ErrorMessage():"");
		ANNRecognitionLog(LOG_ERROR,temp);
	}
	catch(exception &e)
	{
		char temp[1024];
		sprintf(temp,"Debug=%d,OCR File %s Failure,Exception=%s",debug,strFileName.c_str(),e.what());
		ANNRecognitionLog(LOG_ERROR,temp);
	}
	catch(...)
	{
		char temp[256];
		sprintf(temp,"Debug=%d,OCR File %s Failure,Unknown Error",debug,strFileName.c_str());
		ANNRecognitionLog(LOG_ERROR,temp);
	}
	
	if(pDoc!=0)
	{
		pDoc->Close(0);
		pDoc->ReleaseDispatch();
		delete pDoc;
	}
	return FALSE;
}