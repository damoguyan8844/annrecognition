#include "stdafx.h"
#include "ANNRecognition.h"
#include "Lock.h"

#include <fstream>
#include <ctime> 
#include <iostream>
#include <vector>
#include <math.h>
#include <ximacfg.h>
#include <ximage.h>
#include <ximatif.h>

using namespace std;

#define MAXDEM 150

long  G_IMPLICITDEM=0;
long  G_INPUTDEM=0;
long  G_OUTPUTDEM=0;

double G_LEARNSPEED=0.05;
double G_ACCEPTDEF=0.1;
long   G_RANDSEED=-1.0;
long   G_ACCEPTRATING=8;

double W_IMPLICITDEM_INPUTDEM[MAXDEM][MAXDEM];
double B_IMPLICITDEM[MAXDEM];

double W_OUTPUTDEM_IMPLICITDEM[MAXDEM][MAXDEM];
double B_OUTPUTDEM[MAXDEM];

double  A_IMPLICITDEM[MAXDEM];
double  A_OUTPUTDEM[MAXDEM];
double  DEF_OUTPUTDEM[MAXDEM];
double  DEF_WEIGHTS[MAXDEM];

string  G_SettingFile("ANNRecognition.dat");
string  G_ErrorRecordFolder("C:");

ANNRECOGNITION_API BOOL LoadBPParameters( LPSTR settingFile )
{
//	Lock lock(&_cs);

	FILE *fp;
	if((fp=fopen(settingFile,"rb"))==NULL)
	{
		ANNRecognitionLog("Open File Failure",LOG_ERROR);
		return FALSE;
	}
	
	//??????,????,??????
	long *bufferLong;
	bufferLong=(long *)malloc(3*sizeof(long));
	fread((char*)bufferLong,sizeof(long),3,fp);

	if(bufferLong[0]>=MAXDEM || bufferLong[1]>=MAXDEM ||bufferLong[2]>=MAXDEM)
	{
		ANNRecognitionLog("Dem Bigger Than MAXDEM",LOG_ERROR);
		free(bufferLong);
		fclose(fp);
		return FALSE;
	}

	G_INPUTDEM=bufferLong[0];
	G_IMPLICITDEM=bufferLong[1];
	G_OUTPUTDEM=bufferLong[2];
	free(bufferLong);
	
	if(feof(fp)) return FALSE;
	
	double dblTemp=0.0;
	//???????????????
	for(int i=0;i<G_IMPLICITDEM;i++)
	{
		for(int j=0;j<G_INPUTDEM;j++)
		{
			fread((char*)&dblTemp,sizeof(double),1,fp);
			W_IMPLICITDEM_INPUTDEM[i][j]=dblTemp;
		}
	}
	
	if(feof(fp)) return FALSE;


	for(i=0;i<G_IMPLICITDEM;i++)
	{
		fread((char*)&dblTemp,sizeof(double),1,fp);
		B_IMPLICITDEM[i]=dblTemp;
	}
	
	if(feof(fp)) return FALSE;

	for(i=0;i<G_OUTPUTDEM;i++)
	{
		for(int j=0;j<G_IMPLICITDEM;j++)
		{
			fread((char*)&dblTemp,sizeof(double),1,fp);
			W_OUTPUTDEM_IMPLICITDEM[i][j]=dblTemp;
		}
	}
	
	if(feof(fp)) return FALSE;

	for(i=0;i<G_OUTPUTDEM;i++)
	{
		fread((char*)&dblTemp,sizeof(double),1,fp);
		B_OUTPUTDEM[i]=dblTemp;
	}
	
	fclose(fp);

	G_SettingFile=settingFile;
	
	InitTrainBPWeights(0);

	return TRUE;	
}

ANNRECOGNITION_API BOOL SaveBPParameters( LPSTR settingFile )
{
	FILE *fp;
	fp=fopen(settingFile,"wb+");
	long *bufferLong;
	bufferLong=(long*)malloc(3*sizeof(long));
	bufferLong[0]=G_INPUTDEM;
	bufferLong[1]=G_IMPLICITDEM;
	bufferLong[2]=G_OUTPUTDEM;
	fwrite((char*)bufferLong,sizeof(long),3,fp);
	free(bufferLong);

	for(int k=0;k<G_IMPLICITDEM;k++)
	{
		for(int j=0;j<G_INPUTDEM;j++)
			fwrite((char*)&W_IMPLICITDEM_INPUTDEM[k][j],sizeof(double),1,fp);
	}
	for( k=0;k<G_IMPLICITDEM;k++)
		fwrite((char*)&B_IMPLICITDEM[k],sizeof(double),1,fp);
	
	for( k=0;k<G_OUTPUTDEM;k++)
	{	
		for(int j=0;j<G_IMPLICITDEM;j++)
			fwrite((char*)&W_OUTPUTDEM_IMPLICITDEM[k][j],sizeof(double),1,fp);
	}
	for( k=0;k<G_OUTPUTDEM;k++)
		fwrite((char*)&B_OUTPUTDEM[k],sizeof(double),1,fp);
	
	fclose(fp);
	return TRUE;
}


ANNRECOGNITION_API BOOL	PrintBPParameters(LPSTR textFile)
{
	FILE *fp;
	fp=fopen(textFile,"wt+");

	char  time[10];
	char  date[10];
    _strtime(time);  
	_strdate(date);
	fprintf(fp,"BPParameters:Version:%d \t Date: %s \t Time:%s \n",ANNRecognitionVersion(),date,time);    //????????	
	fprintf(fp,"InputLayerDem:%d\tImplicitLayerDem:%d\tOutputLayerDem:%d\n",G_INPUTDEM,G_IMPLICITDEM,G_OUTPUTDEM);

	fprintf(fp,"\nW_IMPLICITDEM_INPUTDEM:\n{\n");
	for(int k=0;k<G_IMPLICITDEM;k++)
	{
		fprintf(fp,"{");
		for(int i=0;i<G_INPUTDEM;i++)
			fprintf(fp,"%5.10f,",W_IMPLICITDEM_INPUTDEM[k][i]);
		fprintf(fp,"}\n");
	}
	fprintf(fp,"\n}\n");

	fprintf(fp,"\nB_IMPLICITDEM:\n{\n");
	for(k=0;k<G_IMPLICITDEM;k++)
	{
		fprintf(fp,"%5.10f,",B_IMPLICITDEM[k]);
	}
	fprintf(fp,"\n}\n");
	
	fprintf(fp,"\nW_OUTPUTDEM_IMPLICITDEM:\n{\n");
	for(k=0;k<G_OUTPUTDEM;k++)
	{
		fprintf(fp,"{");
		for(int i=0;i<G_IMPLICITDEM;i++)
			fprintf(fp,"%5.10f,",W_OUTPUTDEM_IMPLICITDEM[k][i]);
		fprintf(fp,"}\n");
	}
	fprintf(fp,"\n}\n");
	
	fprintf(fp,"\nB_OUTPUTDEM:\n{\n");
	for(k=0;k<G_OUTPUTDEM;k++)
	{
		fprintf(fp,"%5.10f,",B_OUTPUTDEM[k]);
	}
	fprintf(fp,"\n}\n");
	
	fclose(fp);

	return TRUE;
}


ANNRECOGNITION_API BOOL	InitBPParameters(LONG input,LONG implicit,LONG output)
{
	BOOL bRet=InitBPParameters_EX(input,implicit,output);
	bRet = bRet && 	InitTrainBPWeights(0);
	return bRet;
}

ANNRECOGNITION_API BOOL	InitBPParameters_EX(LONG input,LONG implicit,LONG output,double ** w1,double *b1,double **w2,double *b2)
{
	BOOL bRet=FALSE;
	try
	{
		if(input>=MAXDEM || implicit>=MAXDEM ||output>=MAXDEM)
		{
			ANNRecognitionLog("Dem Bigger Than MAXDEM",LOG_ERROR);
			return FALSE;
		}
		
		G_INPUTDEM=input;
		G_IMPLICITDEM=implicit;
		G_OUTPUTDEM=output;
		
		double * w1_point=(double *)w1;
		double * w2_point=(double *)w2;
		
		if(G_RANDSEED==-1)
			srand(time(NULL));  
		else
			srand(G_RANDSEED);

		double temp=pow(G_INPUTDEM,0.5);
		for(int i=0;i<G_IMPLICITDEM;i++)        //???W1,B1
		{  
			int num=G_INPUTDEM;
			while(num>0)   
			{
				--num;
				if(w1_point!=0)
					W_IMPLICITDEM_INPUTDEM[i][num]=*(w1_point+i*G_INPUTDEM+num);
				else
					W_IMPLICITDEM_INPUTDEM[i][num]=rand()*1.0/temp/RAND_MAX;
			}

			if(b1!=0)
				B_IMPLICITDEM[i]=b1[i];
			else
				B_IMPLICITDEM[i]=rand()*1.0/temp/RAND_MAX;
//				B_IMPLICITDEM[i]=0.0;
		}    
		temp=pow(G_IMPLICITDEM,0.5);	
		for(   i=0;i<G_OUTPUTDEM;i++)     //???W2,B2
		{
			int num=G_IMPLICITDEM;
			while(num>0)  
			{
				--num;
				if(w2_point!=0)
					W_OUTPUTDEM_IMPLICITDEM[i][num]=*(w2_point+i*G_IMPLICITDEM+num);
				else
					W_OUTPUTDEM_IMPLICITDEM[i][num]=rand()*1.0/temp/RAND_MAX;
			}
			if(b2!=0)
				B_OUTPUTDEM[i]=b2[i];
			else
				B_OUTPUTDEM[i]=rand()*1.0/temp/RAND_MAX;
//				B_OUTPUTDEM[i]=0.0;
		}
		
		bRet=TRUE;
	}
	catch(...)
	{
		ANNRecognitionLog("InitParameter Exception",LOG_ERROR);
	}
	return bRet;
}

ANNRECOGNITION_API BOOL InitTrainBPLearnSpeed(double dblSpeed)
{	
	G_LEARNSPEED=dblSpeed;
	return TRUE;
}
ANNRECOGNITION_API BOOL InitTrainBPRandSeed(long seed)
{
	G_RANDSEED=seed;
	return TRUE;
}
ANNRECOGNITION_API BOOL InitTrainBPWeights(double*difWeights)
{
	for(int i=0; i<G_OUTPUTDEM;++i)
	{
		if(difWeights!=0)
			DEF_WEIGHTS[i]=difWeights[i];
		else
			DEF_WEIGHTS[i]=1.0;
	}
	return TRUE;
}

ANNRECOGNITION_API double    GetLearningSpeed()
{
	return G_LEARNSPEED;
}
ANNRECOGNITION_API double	 CheakDiffs(double *output,double * dest)
{
	int k=0;
	double aver_def=0;
	while(k<G_OUTPUTDEM) 
	{
		DEF_OUTPUTDEM[k]=(dest[k]-output[k])*DEF_WEIGHTS[k];		 		
		aver_def+=DEF_OUTPUTDEM[k]*DEF_OUTPUTDEM[k]*0.5;         
		k++; 
	}
	aver_def/=G_OUTPUTDEM;
	return aver_def;
}
ANNRECOGNITION_API double   Training(double *input,double * dest)
{
	double avrg_diff=0.0;

	if(input==0 || dest==0)
		return avrg_diff;
	try
	{
		if(!Recognition(input,A_OUTPUTDEM))
			return avrg_diff;
		
		//????  data[INDESENSION] ? ???t(???)
		avrg_diff=CheakDiffs(A_OUTPUTDEM,dest);
		
		//????????
		//S2
		double S2_OUTPUTDEM[MAXDEM];
		for(int	i=0;i<G_OUTPUTDEM;i++)
			S2_OUTPUTDEM[i]=(-2)*(1-A_OUTPUTDEM[i])*A_OUTPUTDEM[i]*DEF_OUTPUTDEM[i];
		//S1
		double S1_IMPLICITDEM[MAXDEM];
		//?????
		double temp_IMPLICITDEM_OUTPUTDEM[MAXDEM][MAXDEM];
		for(    i=0;i<G_IMPLICITDEM;i++)
			for(int j=0;j<G_OUTPUTDEM;j++)
				temp_IMPLICITDEM_OUTPUTDEM[i][j]=W_OUTPUTDEM_IMPLICITDEM[j][i];
		
		for(    i=0;i<G_IMPLICITDEM;i++)
		{   
			S1_IMPLICITDEM[i]=0;
			for(int j=0;j<G_OUTPUTDEM;j++)
				S1_IMPLICITDEM[i]+=temp_IMPLICITDEM_OUTPUTDEM[i][j]*S2_OUTPUTDEM[j];		 
		}
		for(    i=0;i<G_IMPLICITDEM;i++)
		{
			S1_IMPLICITDEM[i]*=(1-A_IMPLICITDEM[i])*A_IMPLICITDEM[i];			  
		}
		
		//??????
		double a=GetLearningSpeed();
		//W2??
		for(   i=0;i<G_OUTPUTDEM;i++)
			for( int j=0;j<G_IMPLICITDEM;j++)
				W_OUTPUTDEM_IMPLICITDEM[i][j]-=a*S2_OUTPUTDEM[i]*A_IMPLICITDEM[j];
		//B2??
		for(   i=0;i<G_OUTPUTDEM;i++)
			B_OUTPUTDEM[i]-=a*S2_OUTPUTDEM[i];
		//W1??
		for(   i=0;i<G_IMPLICITDEM;i++)
			for( int j=0;j<G_INPUTDEM;j++)
				W_IMPLICITDEM_INPUTDEM[i][j]-=a*S1_IMPLICITDEM[i]*input[j];	 
		//B1??
		for(   i=0;i<G_IMPLICITDEM;i++)
			B_IMPLICITDEM[i]-=a*S1_IMPLICITDEM[i];
	}
	catch(...)
	{
		ANNRecognitionLog("Training By Input Exception",LOG_ERROR);
	}

	return avrg_diff;
}

ANNRECOGNITION_API BOOL	Recognition(double *input,double * result)
{
	if(result==0 || input==0) return FALSE;
	
//	Lock lock(&_cs);
	
	//??????a1=logsign(W1*a0+B1) 
	for(int i=0;i<G_IMPLICITDEM;i++)
	{    
		int num=0;
		A_IMPLICITDEM[i]=0;
		while(num<G_INPUTDEM)   
		{
			A_IMPLICITDEM[i]+=W_IMPLICITDEM_INPUTDEM[i][num]*input[num];
			num++;
		}
		A_IMPLICITDEM[i]+=B_IMPLICITDEM[i];
		A_IMPLICITDEM[i]=1/(1+exp(-A_IMPLICITDEM[i]));
	}
	//??????a2=logsign(W2*a1+B2)
	for(   i=0;i<G_OUTPUTDEM;i++)
	{    
		int num=0;
		A_OUTPUTDEM[i]=0;
		while(num<G_IMPLICITDEM)   
		{
			A_OUTPUTDEM[i]+=W_OUTPUTDEM_IMPLICITDEM[i][num]*A_IMPLICITDEM[num];
			num++;
		}
		A_OUTPUTDEM[i]+=B_OUTPUTDEM[i];
		A_OUTPUTDEM[i]=1/(1+exp(-A_OUTPUTDEM[i]));
	}

	try
	{
		for(   i=0;i<G_OUTPUTDEM;i++)
		{    
			*(result+i)=A_OUTPUTDEM[i];
		}
		return TRUE;
	}
	catch(...)
	{
		ANNRecognitionLog("Recognition Error",LOG_ERROR);
	}
	return FALSE;
}


ANNRECOGNITION_API BOOL	BPEncode(HDIB hInputDIB,double * outCode,LONG top, LONG left,LONG right, LONG bottom,LPSTR gridFile)
{
	if(hInputDIB==0 || outCode==0) return FALSE;

//	Lock lock(&_cs);

	try
	{
		for(int i=0; i<G_INPUTDEM ; i++)
			outCode[i]=0.0;
	}
	catch(...)
	{
		return FALSE;
	}

	// ??DIB???
	LPSTR	lpDIB;
	
	// ?DIB????DIB?????DIB
	lpDIB = (LPSTR) ::GlobalLock((HGLOBAL)hInputDIB);
	
	// ??DIB????????
	LPSTR   lpDIBBits;
	
	// ??DIB?????
	BYTE *	lpSrc;	
	BYTE byte255=255;

	// ????
	LONG	lWidth;	
	
	// ????
	LONG	lHeight;	
	// ????????
	LONG	lLineBytes;	
	
	// ??DIB????????
	lpDIBBits = ::FindDIBBits(lpDIB);	
	
	// ??????
	lWidth = ::DIBWidth(lpDIB);	
	
	// ??????
	lHeight = ::DIBHeight(lpDIB);	
	
	// ??????????
	lLineBytes = WIDTHBYTES(lWidth * 8);	
	
	RECT range;
	range.left=left;
	range.top=top;	
	range.right=right;
	range.bottom=bottom;

	if(range.right<=0 || range.right>=lWidth)
		range.right=lWidth;

	if(range.bottom<=0 || range.bottom>=lHeight)
		range.bottom=lHeight;
	
	BOOL bRet=FALSE;
	
	ofstream * pfout=0;

	string filename(gridFile==0?"":gridFile);

	if(!filename.empty())
		pfout=new ofstream(filename.c_str(),ios::app);

	if(G_INPUTDEM==64)
	{
		//??????,????????,?????,???

		RECT smallRange;
		smallRange.left=range.right;
		smallRange.right = range.left;
		smallRange.top=range.bottom;
		smallRange.bottom=range.top;

		for(int ii = range.top; ii < range.bottom; ii++)
		{	
			for(int jj = range.left; jj < range.right; jj++)
			{
				lpSrc = (unsigned char*)lpDIBBits + lLineBytes * ii + jj;					
				if(*lpSrc<200) //Black Point
				{
					if(ii<smallRange.top)
						smallRange.top=ii;
					if(ii>smallRange.bottom)
						smallRange.bottom=ii;
					if(jj<smallRange.left)
						smallRange.left=jj;
					if(jj>smallRange.right)
						smallRange.right=jj;
				}
			}
		}
		range=smallRange;
		if(pfout) 
		{
			(*pfout)<<"\nWidth:"<<lWidth<<",Height:"<<lHeight<<endl;
			(*pfout)<<"\nLeft:"<<range.left<<",Right:"<<range.right<<",Top:"<<lHeight-range.bottom<<",Bottom:"<<lHeight-range.top<<"\n";
		}

		RECT enlargedRange=range;
		
		int widthBlank=8-RECTWIDTH(&range)%8;
		
		if(widthBlank!=8)
		{
			if(widthBlank<4) //Enlarge
			{
				enlargedRange.left=range.left-widthBlank/2;
				enlargedRange.right=range.right + widthBlank -widthBlank/2;
			}
			else //Smaller
			{
				widthBlank=8-widthBlank;

				int move=widthBlank/2;
				if(range.left<(lWidth-range.right))
					move+=1;

				enlargedRange.left=range.left+move;
				enlargedRange.right=range.right - (widthBlank -move);
			}
		}

		int heightBlank=8-RECTHEIGHT(&range)%8;
		
		if(heightBlank!=8)
		{
			if(heightBlank<4) //Enlarge
			{
				enlargedRange.top=range.top-heightBlank/2;
				enlargedRange.bottom=range.bottom+heightBlank-heightBlank/2;
			}
			else //Smaller
			{
				heightBlank=8-heightBlank;

				int move=heightBlank/2;
				if(range.top<(lHeight-range.bottom))
					move+=1;

				enlargedRange.top=range.top+move;
				enlargedRange.bottom=range.bottom-(heightBlank-move);
			}
		}

		int gridWidth=RECTWIDTH(&enlargedRange)/8;
		int gridHeight=RECTHEIGHT(&enlargedRange)/8;
		
		if(pfout) 
		{
			(*pfout)<<"\nLeft:"<<enlargedRange.left<<",Right:"<<enlargedRange.right<<",Top:"<<lHeight-enlargedRange.bottom<<",Bottom:"<<lHeight-enlargedRange.top<<"\n";
		}

		double factor=1.0/(gridWidth*gridHeight);
		//????
		for(int i = enlargedRange.bottom; i > enlargedRange.top; i --)
		{		
		//	int line=lHeight-(i-enlargedRange.top);
			int line=i;

			//????
			for(int j = enlargedRange.left; j < enlargedRange.right; j++)
			{	
				int col=j;
		
				if(line<=range.top || line>range.bottom ||col<range.left || col>=range.right)
					lpSrc = & byte255; 					
				else
					// ??DIB?i?,?j??????
					lpSrc = (unsigned char*)lpDIBBits + lLineBytes * line + col;
			
				//??150,?????
				if(*lpSrc<200)
				{
					if(pfout) (*pfout)<<1<<",";
					*(outCode+((enlargedRange.bottom -i)/gridHeight)*8+(j-enlargedRange.left)/gridWidth)+=factor;
				}
				else
				{
					if(pfout) (*pfout)<<"0"<<",";
				}
			}
			if(pfout) (*pfout)<<endl;
		}
		if(pfout)
		{
			for( i=0;i<8;i++)
			{
				for(int j=0;j<8;j++)
					(*pfout)<<(*(outCode+i*8+j))*gridWidth*gridHeight<<",";
				(*pfout)<<endl;
			}

		}
		bRet = TRUE;
	}

	if(pfout)
	{
		(*pfout).close();
		delete pfout;
	}

	//????
	::GlobalUnlock((HGLOBAL)hInputDIB);

	return bRet;
}


typedef deque<RECT>  CRectLink;
typedef deque<HDIB>  HDIBLink;
extern int m_lianXuShu;
extern std::vector<CRectLink> g_CharSegmentRet;

ANNRECOGNITION_API BOOL	Recognition_EX(HDIB hInputDIB,LONG charRectID,int * outValues)
{
	if(charRectID>=g_CharSegmentRet.size()) return FALSE;
	
//	Lock lock(&_cs);
	
	CRectLink m_charRect = g_CharSegmentRet[charRectID];
	CRectLink m_charRectCopy=m_charRect;
	
	int digicount=GetSegmentCount(charRectID);
	
	HDIBLink m_dibRect;
	HDIBLink m_dibRectCopy;
	
	unsigned char* lpSrc;
	int w,h;
	w=RECTWIDTH(&m_charRect.front ());
	h=RECTHEIGHT(&m_charRect.front ());
	
	int i_src,j_src;
	int i,j;
	int counts=0;
	RECT rect;
	BYTE* lpDIB=(BYTE*)::GlobalLock ((HGLOBAL)hInputDIB);
	BYTE* lpDIBBits=(BYTE*)::FindDIBBits ((char*)lpDIB);
	BYTE* lpNewDIBBits;
	BYTE* lpDst;
	LONG lLineBytes=(digicount*w+3)/4*4;
	LONG lLineBytesnew =(w+3)/4*4;
	HDIB hDIB=NULL;
	while(!m_charRect.empty ())
	{
		hDIB=::NewDIB (w,h,8);
		lpDIB=(BYTE*) ::GlobalLock((HGLOBAL)hDIB);	
		lpNewDIBBits = (BYTE*)::FindDIBBits((char*)lpDIB);
		lpDst=(BYTE*)lpNewDIBBits;
		memset(lpDst,(BYTE)255,lLineBytesnew * h);		
		rect=m_charRect.front ();
		m_charRect.pop_front ();
		for(i=0;i<h;i++)
			for(j=0;j<w;j++)
			{
				i_src=rect.top + i;
				j_src=j+counts*w;
				lpSrc=(BYTE *)lpDIBBits + lLineBytes *  i_src + j_src;
				lpDst=(BYTE *)lpNewDIBBits + lLineBytesnew * i + j;
				*lpDst=*lpSrc;
			}
			::GlobalUnlock (hDIB);
			
			RemoveScatterNoise(hDIB);
			m_dibRect.push_back (hDIB);
			counts++;
	}
	m_charRect=m_charRectCopy;
	m_dibRectCopy=m_dibRect;

	counts=0;

	std::vector<double> input(G_INPUTDEM,0.0);
	std::vector<double> result(G_OUTPUTDEM,0.0);
	
	BOOL bHasError=FALSE;
	while(!m_dibRect.empty ())
	{
		if(!bHasError)
		{
			for(int i=0;i<G_INPUTDEM;i++)  input[i]=0.0;
			for(int j=0;i<G_OUTPUTDEM;j++)  result[j]=0.0;
			
			if(BPEncode(m_dibRect.front (),&input[0],0,0,0,0,0))
			{
				Recognition(&input[0],&result[0]);
				
				int intRes=0;

				if(result[0]>0.5)
					intRes+=1;
				if(result[1]>0.5)
					intRes+=2;
				if(result[2]>0.5)
					intRes+=4;
				if(result[3]>0.5)
					intRes+=8;
				
				if(intRes>=10) 
					bHasError=TRUE;

				outValues[counts]=intRes;
				counts++;
			}
		}
		ReleaseDIBFile(m_dibRect.front ());
		m_dibRect.pop_front ();
	}
	m_dibRect=m_dibRectCopy;
	outValues[counts]=-1;

	if(bHasError)
	{
		outValues[0]=-1;
		SaveSegment(hInputDIB,charRectID,(LPSTR)G_ErrorRecordFolder.c_str());
	}
	return !bHasError;
}


ANNRECOGNITION_API BOOL SetErrorRecordFolder( LPSTR lpErrorFolder )
{
	G_ErrorRecordFolder=lpErrorFolder==0?"C:":lpErrorFolder;
	return TRUE;
}