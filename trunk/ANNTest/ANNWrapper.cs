using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
namespace ANNTest
{
    class ANNWrapper
    {
        [DllImport("ANNRecognition.dll", EntryPoint = "ConvertJPEG2BMP")]
        public static extern bool ConvertJPEG2BMP(string jpegFile,string bmpFile);

        [DllImport("ANNRecognition.dll", EntryPoint = "ConvertBMP2TIF")]
        public static extern bool ConvertBMP2TIF(string bmpFile, string tifFile);

        [DllImport("ANNRecognition.dll", EntryPoint = "BlackWhiteBMP")]
        public static extern bool BlackWhiteBMP(string bmpFile, int threshold);

        [DllImport("ANNRecognition.dll", EntryPoint = "RevertBlackWhiteBMP")]
        public static extern bool RevertBlackWhiteBMP(string bmpFile);

        [DllImport("ANNRecognition.dll", EntryPoint = "SaveBlockToBMP")]
        public static extern bool SaveBlockToBMP(string bmpFile, 
                double leftRate, double topRate, double rightRate, double bottomRate,
                string blockBMPFile);
        
        [DllImport("ANNRecognition.dll", EntryPoint = "SaveBlockToBMP2")]
        public static extern bool SaveBlockToBMP2(string bmpFile,
                long leftRate, long topRate, long rightRate, long bottomRate,
                string blockBMPFile);

        [DllImport("ANNRecognition.dll", EntryPoint = "OCRFile")]
        public static extern bool OCRFile(string bmpFile,out string content);

        [DllImport("ANNRecognition.dll", EntryPoint = "ReadDIBFile")]
        public static extern IntPtr ReadDIBFile( string fileName);

        //
        [DllImport("ANNRecognition.dll", EntryPoint = "BPEncode")]
        public static extern bool BPEncode(IntPtr hInputDIB, double[] outCode, long top, long left, long right, long bottom);

        [DllImport("ANNRecognition.dll", EntryPoint = "LoadBPParameters")]
        public static extern bool LoadBPParameters(string paraFile);

        [DllImport("ANNRecognition.dll", EntryPoint = "SaveBPParameters")]
        public static extern bool SaveBPParameters(string paraFile);

        [DllImport("ANNRecognition.dll", EntryPoint = "PrintBPParameters")]
        public static extern bool PrintBPParameters(string txtParaFile);

        [DllImport("ANNRecognition.dll", EntryPoint = "InitTrainBPLearnSpeed")]
        public static extern bool InitTrainBPLearnSpeed(double learningSpeed);

        [DllImport("ANNRecognition.dll", EntryPoint = "InitBPParameters")]
        public static extern bool InitBPParameters(int inputDim,int implicitDim,int outputDim,double[][] w1,double[] b1,double[][] w2,double[] b2);

        [DllImport("ANNRecognition.dll", EntryPoint = "Training")]
        public static extern double Training(double[] input, double[] dest);

        [DllImport("ANNRecognition.dll", EntryPoint = "ReleaseDIBFile")]
        public static extern bool ReleaseDIBFile(IntPtr hInputDIB);
    }
}
