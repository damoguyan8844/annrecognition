#include "stdafx.h"
#include "ANNRecognition.h"
#include <fstream>
#include <ctime> 
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

ANNRECOGNITION_API BOOL LoadBPParameters( LPSTR settingFile )
{
	FILE *fp;
	if((fp=fopen(settingFile,"rb"))==NULL)
	{
		ANNRecognitionLog("Open File Failure",LOG_ERROR);
		return FALSE;
	}
	
	//读取输入节点，隐层节点，输出节点维度
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
	//读取输入层隐层节点权值以及偏移
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
	fprintf(fp,"BPParameters:Version:%d \t Date: %s \t Time:%s \n",ANNRecognitionVersion(),date,time);    //获取当前时间信息	
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

ANNRECOGNITION_API BOOL	InitBPParameters(int input,int implicit,int output,double ** w1,double *b1,double **w2,double *b2)
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
		for(int i=0;i<G_IMPLICITDEM;i++)        //初始化W1,B1
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
//				B_IMPLICITDEM[i]=rand()*1.0/temp/RAND_MAX;
				B_IMPLICITDEM[i]=0.0;
		}    
		temp=pow(G_IMPLICITDEM,0.5);	
		for(   i=0;i<G_OUTPUTDEM;i++)     //初始化W2,B2
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
//				B_OUTPUTDEM[i]=rand()*1.0/temp/RAND_MAX;
				B_OUTPUTDEM[i]=0.0;
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
		
		//计算误差  data[INDESENSION] 中 存储有t（目标值）
		avrg_diff=CheakDiffs(A_OUTPUTDEM,input);
		
		//反向传播敏感性值
		//S2
		double S2_OUTPUTDEM[MAXDEM];
		for(int	i=0;i<G_OUTPUTDEM;i++)
			S2_OUTPUTDEM[i]=(-2)*(1-A_OUTPUTDEM[i])*A_OUTPUTDEM[i]*DEF_OUTPUTDEM[i];
		//S1
		double S1_IMPLICITDEM[MAXDEM];
		//矩阵的转置
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
		
		//确定学习速度
		double a=GetLearningSpeed();
		//W2更新
		for(   i=0;i<G_OUTPUTDEM;i++)
			for( int j=0;j<G_IMPLICITDEM;j++)
				W_OUTPUTDEM_IMPLICITDEM[i][j]-=a*S2_OUTPUTDEM[i]*A_IMPLICITDEM[j];
		//B2更新
		for(   i=0;i<G_OUTPUTDEM;i++)
			B_OUTPUTDEM[i]-=a*S2_OUTPUTDEM[i];
		//W1更新
		for(   i=0;i<G_IMPLICITDEM;i++)
			for( int j=0;j<G_INPUTDEM;j++)
				W_IMPLICITDEM_INPUTDEM[i][j]-=a*S1_IMPLICITDEM[i]*input[j];	 
		//B1更新
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

	//第一层的输出a1=logsign(W1*a0+B1) 
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
	//第二层的输出a2=logsign(W2*a1+B2)
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


ANNRECOGNITION_API BOOL	BPEncode(HDIB hInputDIB,double * outCode,LONG top, LONG left,LONG right, LONG bottom)
{
	if(hInputDIB==0 || outCode==0) return FALSE;
	
	try
	{
		for(int i=0; i<G_INPUTDEM ; i++)
			outCode[i]=0.0;
	}
	catch(...)
	{
		return FALSE;
	}

	// 指向DIB的指针
	LPSTR	lpDIB;
	
	// 由DIB句柄得到DIB指针并锁定DIB
	lpDIB = (LPSTR) ::GlobalLock((HGLOBAL)hInputDIB);
	
	// 指向DIB象素数据区的指针
	LPSTR   lpDIBBits;
	
	// 指向DIB象素的指针
	BYTE *	lpSrc;	
	
	// 图像宽度
	LONG	lWidth;	
	
	// 图像高度
	LONG	lHeight;	
	// 图像每行的字节数
	LONG	lLineBytes;	
	
	// 找到DIB图像象素起始位置
	lpDIBBits = ::FindDIBBits(lpDIB);	
	
	// 获取图像宽度
	lWidth = ::DIBWidth(lpDIB);	
	
	// 获取图像高度
	lHeight = ::DIBHeight(lpDIB);	
	
	// 计算图像每行的字节数
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

//	ofstream fout("C:\\Range.txt",ios::app);

	if(G_INPUTDEM==64)
	{
		//八行八列取值，截图范围整数倍化，按边缘补齐，行优先

		RECT enlargedRange=range;
		
		int widthBlank=RECTWIDTH(&range)%8;
		enlargedRange.left=range.left-widthBlank/2;
		if(enlargedRange.left<0)
			enlargedRange.left=0;
		if(widthBlank<=0)
			widthBlank=8;

		enlargedRange.right=enlargedRange.left + RECTWIDTH(&range)-widthBlank+8;

		int heightBlank=RECTHEIGHT(&range)%8;
		enlargedRange.top=range.top-heightBlank/2;
		if(enlargedRange.top<0)
			enlargedRange.top=0;
		if(heightBlank<=0)
			heightBlank=8;

		enlargedRange.bottom=enlargedRange.top+RECTHEIGHT(&range)-heightBlank+8;

		int gridWidth=RECTWIDTH(&enlargedRange)/8;
		int gridHeight=RECTHEIGHT(&enlargedRange)/8;
		
		double factor=1.0/(gridWidth*gridHeight);
		//逐行扫描
		for(int i = enlargedRange.top; i < enlargedRange.bottom; i++)
		{		
			int line=enlargedRange.bottom-i;
			if(line<range.top)
				line=range.top;
			else if (line>=range.bottom)
				line=range.bottom-1;
			//逐列扫描
			for(int j = enlargedRange.left; j < enlargedRange.right; j++)
			{	
				int col=j;
		
				if(col<range.left)
					col=range.left;
				else if (col>=range.right)
					col=range.right-1;
			
				// 指向DIB第i行，第j个象素的指针
				lpSrc = (unsigned char*)lpDIBBits + lLineBytes * line + col;
			
				//小于150，作为有效点
				if(*lpSrc<150)
				{
//					fout<<1<<",";
					*(outCode+((i-enlargedRange.top)/gridHeight)*8+(j-enlargedRange.left)/gridWidth)+=factor;
				}
				else
				{
//					fout<<"0"<<",";
				}
			}
//			fout<<endl;
		}
		
// 		for( i=0;i<8;i++)
// 		{
// 			for(int j=0;j<8;j++)
// 				fout<<(*(outCode+i*8+j))*gridWidth*gridHeight<<",";
// 			fout<<endl;
// 		}

		bRet = TRUE;
	}

//	fout.close();

	//解除锁定
	::GlobalUnlock((HGLOBAL)hInputDIB);

	return bRet;
}