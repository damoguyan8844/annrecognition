#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <ctime> 
#include <cstdlib>
#include <math.h>
#include <conio.h>
 

using namespace std;

#define INPUTDEM      64
#define IMPLICITDEM   8
#define OUTPUTDEM     4 
#define SEED          1
#define LEARNSPEED    0.05
#define STOPDEF       0.001
#define STOPNEARNUM   8
#ifdef  WIN32
#define TRA_FILE   "C:\\BP\\trainingData.txt"  
#define SAV_FILE   "C:\\BP\\trainingUpdate.txt"
#define RES_FILE   "C:\\BP\\trainingResult.txt"
#else
#define TRA_FILE   "bp/trainingData.txt"  
#define SAV_FILE   "bp/trainingUpdate.txt"
#define RES_FILE   "bp/trainingResult.txt"
#endif

ifstream fi;
float  data[INPUTDEM+1];
float  W1[IMPLICITDEM][INPUTDEM];
float  B1[IMPLICITDEM];
float  A1[IMPLICITDEM];
float  W2[OUTPUTDEM][IMPLICITDEM];
float  B2[OUTPUTDEM];
float  A2[OUTPUTDEM];
float  DEF[OUTPUTDEM];

int    SAVECOUNTER=1;
void initialMatrix( )    //初始化矩阵
{
	 srand(time(NULL));  
	 float temp=pow(INPUTDEM,0.5);
	 for(int i=0;i<IMPLICITDEM;i++)        //初始化W1,B1
	 {  
		 int num=INPUTDEM;
		 while(num>0)   W1[i][--num]=rand()*1.0/temp/RAND_MAX;
         B1[i]=rand()*1.0/temp/RAND_MAX;
	 }    
	 temp=pow(IMPLICITDEM,0.5);	
	 for(   i=0;i<OUTPUTDEM;i++)     //初始化W2,B2
	 {
		int num=IMPLICITDEM;
		while(num>0)   W2[i][--num]=rand()*1.0/temp/RAND_MAX;
        B2[i]=rand()*1.0/temp/RAND_MAX;
	 }
}
float getLearningSpeed()
{
	return (float) LEARNSPEED;
}

void saveMatrix(char *indicate,float *matrix,int num,int line=1)   //保存矩阵
{
    FILE * fi=fopen(RES_FILE,"a+");
	char  time[10];
	char  date[10];
    _strtime(time);  
	_strdate(date);
 	fprintf(fi,"第%d记录(%s/%s\t):%s\n{\n",SAVECOUNTER++,time,date,indicate);    //获取当前时间信息
	for(int k=0;k<line;k++)
	{
		fprintf(fi,"{");
		for(int i=0;i<num;i++)
		    fprintf(fi,"%5.10f,",*(matrix+k*num+i));
		fprintf(fi,"}\n");
	}
	fprintf(fi,"\n}\n");
	fclose(fi);
}
float check_def(float *a,float t)
{ 
	 static int rec=0;

	 int k=0;
	 float aver_def=0;
	 int temp=(int)t;
	 while(k<OUTPUTDEM) 
	 {
//		 DEF[k]=(temp%2-a[k])*(temp%2-a[k])/2.0;
		 DEF[k]=temp%2-a[k];

		 temp/=2;
		 aver_def+=fabs(DEF[k])*fabs(DEF[k])*0.5; 
		 k++; 
	 }
	 aver_def/=OUTPUTDEM;
	 return aver_def; 	 
}
float  training()
{	  
	
	 float avrDef=0.0;
	 //第一层的输出a1=logsign(W1*a0+B1) 
	 for(int i=0;i<IMPLICITDEM;i++)
	 {    
          int num=0;
		  A1[i]=0;
		  while(num<INPUTDEM)   
		  {
			  A1[i]+=W1[i][num]*data[num];
			  num++;
		  }
          A1[i]+=B1[i];
		  A1[i]=1/(1+exp(-A1[i]));
	 }
	 //第二层的输出a2=logsign(W2*a1+B2)
     for(   i=0;i<OUTPUTDEM;i++)
	 {    
          int num=0;
		  A2[i]=0;
		  while(num<IMPLICITDEM)   
		  {
			  A2[i]+=W2[i][num]*A1[num];
			  num++;
		  }
          A2[i]+=B2[i];
		  A2[i]=1/(1+exp(-A2[i]));
	 }	 
	 //计算误差  data[INDESENSION] 中 存储有t（目标值）
	 avrDef=check_def(A2,data[INPUTDEM]);
	
     //反向传播敏感性值
	 //S2
	 float S2[OUTPUTDEM];
	 for(    i=0;i<OUTPUTDEM;i++)
           S2[i]=(-2)*(1-A2[i])*A2[i]*DEF[i];
	 //S1
     float S1[IMPLICITDEM];
	 //矩阵的转置
	 float temp[IMPLICITDEM][OUTPUTDEM];
	 for(    i=0;i<IMPLICITDEM;i++)
		 for(int j=0;j<OUTPUTDEM;j++)
			 temp[i][j]=W2[j][i];

	 for(    i=0;i<IMPLICITDEM;i++)
	 {   
		 S1[i]=0;
		 for(int j=0;j<OUTPUTDEM;j++)
			 S1[i]+=temp[i][j]*S2[j];		 
	 }
	 for(    i=0;i<IMPLICITDEM;i++)
	 {
		 S1[i]*=(1-A1[i])*A1[i];			  
	 }
	 
	 //确定学习速度
	 float a=getLearningSpeed();
	 //W2更新
	 for(   i=0;i<OUTPUTDEM;i++)
		 for( int j=0;j<IMPLICITDEM;j++)
			 W2[i][j]-=a*S2[i]*A1[j];
	 //B2更新
	 for(   i=0;i<OUTPUTDEM;i++)
		     B2[i]-=a*S2[i];
	 //W1更新
	 for(   i=0;i<IMPLICITDEM;i++)
		 for( int j=0;j<INPUTDEM;j++)
			 W1[i][j]-=a*S1[i]*data[j];	 
	 //B1更新
     for(   i=0;i<IMPLICITDEM;i++)
		     B1[i]-=a*S1[i];
	 return avrDef;
}
void  TrainingStart()      /*OK*/
{  
 
    char datr[200];
	fi.open(TRA_FILE,ios::in);
	if(!fi)
	{
		 cout<<"File ont find"<<endl;
		 return;
	}

	FILE * fiSav=fopen(SAV_FILE,"w+");
	long   trainingCount=0;

	while(1)
	{

		float avrDef=0.0;
		long numCount=0;

		while(fi.getline(datr,200))    /*读取训练样本*/
		{   
			 numCount++;

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
					data[point]=data[point]/16;   //为使计算机结果保持在一定范围内对数据进行改动,0 1 化.
					point++;
					begin=i+1;
				 }
				 i++;
			 }while(i<200&&datr[i-1]!='\0');
			data[point-1]=data[point-1]*16;
			avrDef+=training();

	//		fprintf(fiSav," Training Data:%d AvrgDif:%f",numCount,avrDef);
	
		}

		fi.clear();
		fi.seekg(0,ios::beg);
		
		avrDef/=numCount;
		
		char  time[10];
		char  date[10];
		_strtime(time);  
		_strdate(date);
		
		fprintf(fiSav," Training Count: %d Date:%s Time: %s AvrgDif:%f \n",trainingCount++,time,date,avrDef);
		if(avrDef<0.0015)
		{
			saveMatrix("data",data,INPUTDEM+1);
			saveMatrix("W1",W1[0],INPUTDEM,IMPLICITDEM);
			saveMatrix("B1",B1,IMPLICITDEM); 
			saveMatrix("W2",W2[0],IMPLICITDEM,OUTPUTDEM);
			saveMatrix("B2",B2,OUTPUTDEM);
			saveMatrix("out",A2,OUTPUTDEM);

		}
		if(avrDef<STOPDEF)
			break;
	}
	//记录训练结果
	saveMatrix("data",data,INPUTDEM+1);
	saveMatrix("W1",W1[0],INPUTDEM,IMPLICITDEM);
	saveMatrix("B1",B1,IMPLICITDEM); 
	saveMatrix("W2",W2[0],IMPLICITDEM,OUTPUTDEM);
	saveMatrix("B2",B2,OUTPUTDEM);
	saveMatrix("out",A2,OUTPUTDEM);
	fi.close();
	fclose(fiSav);
	return ;
}
void main()
{
	initialMatrix();
  	TrainingStart();
}