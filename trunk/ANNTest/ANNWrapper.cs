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

        [DllImport("ANNRecognition.dll", EntryPoint = "OCRFile")]
        public static extern bool OCRFile(string fullFileName,out string strContent);
    }
}
