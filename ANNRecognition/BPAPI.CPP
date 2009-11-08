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
double G_RANDSEED=-1.0;
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
			fprintf(fp,"%f,",W_IMPLICITDEM_INPUTDEM[k][i]);
		fprintf(fp,"}\n");
	}
	fprintf(fp,"\n}\n");

	fprintf(fp,"\nB_IMPLICITDEM:\n{\n");
	for(k=0;k<G_IMPLICITDEM;k++)
	{
		fprintf(fp,"%f,",B_IMPLICITDEM[k]);
	}
	fprintf(fp,"\n}\n");
	
	fprintf(fp,"\nW_OUTPUTDEM_IMPLICITDEM:\n{\n");
	for(k=0;k<G_OUTPUTDEM;k++)
	{
		fprintf(fp,"{");
		for(int i=0;i<G_IMPLICITDEM;i++)
			fprintf(fp,"%f,",W_OUTPUTDEM_IMPLICITDEM[k][i]);
		fprintf(fp,"}\n");
	}
	fprintf(fp,"\n}\n");
	
	fprintf(fp,"\nB_OUTPUTDEM:\n{\n");
	for(k=0;k<G_OUTPUTDEM;k++)
	{
		fprintf(fp,"%f,",B_OUTPUTDEM[k]);
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

		srand(time(NULL));  
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
ANNRECOGNITION_API BOOL InitTrainBPRandSeed(double seed)
{
	G_RANDSEED=seed;
	return TRUE;
}
ANNRECOGNITION_API BOOL InitTrainBPAcceptMark(long lngRating ,double dblDiff,double*difWeights)
{
	G_ACCEPTDEF=dblDiff;
	G_ACCEPTRATING = lngRating;

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
ANNRECOGNITION_API BOOL		 CheakDiffs(double *output,double * dest)
{
	int k=0;
	double aver_def=0;
	while(k<G_OUTPUTDEM) 
	{
		DEF_OUTPUTDEM[k]=(dest[k]-output[k])*DEF_WEIGHTS[k];		 
		
		if(DEF_OUTPUTDEM[k]<0)  
			aver_def-=DEF_OUTPUTDEM[k];
		else          
			aver_def+=DEF_OUTPUTDEM[k]; 
		k++; 
	}

	aver_def/=G_OUTPUTDEM;
	
	static long static_nearnum=0;

	if(aver_def<G_ACCEPTDEF) 
	{
		static_nearnum++;
		if(static_nearnum>=G_ACCEPTRATING) 
			return TRUE;
	}
	
	static_nearnum=0;

	return FALSE;
}
ANNRECOGNITION_API BOOL      Training(double *input,double * dest)
{
	if(input==0 || dest==0)
		return FALSE;
	try
	{
		if(!Recognition(input,A_OUTPUTDEM))
			return FALSE;
		
		//计算误差  data[INDESENSION] 中 存储有t（目标值）
		if(CheakDiffs(A_OUTPUTDEM,input))
		{
			SaveBPParameters((char *)G_SettingFile.c_str());
			return TRUE;
		}
	}
	catch(...)
	{
		ANNRecognitionLog("Training By Input Exception",LOG_ERROR);
	}

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
	 return 0;
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
		ANNRecognitionLog("",LOG_ERROR);
	}
	return FALSE;
}
