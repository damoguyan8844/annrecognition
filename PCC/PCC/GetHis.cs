using System;

using System.Collections.Generic;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.IO;

namespace PicCompare.GetHisogram
{
    public class GetHis
    {
        /// <summary>
        /// 重设图片大小
        /// </summary>
        /// <param name="imageFile">原始图片</param>
        /// <param name="newImageFile">新图片</param>
        /// <returns>返回一个Bitmap对象</returns>
        public Bitmap Resized(string imageFile, string newImageFile)
        {
            Bitmap img = new Bitmap(imageFile);
            

            try
            {
                ResizeBMP(256, img).Save(newImageFile,ImageFormat.Jpeg);
                return ResizeBMP(256, img);
            }
            catch (Exception ex)
            {
                throw new Exception(ex.Message);
            }
            finally
            {
                //img.Dispose();
                //imgOutput.Dispose();
            }
        }
        /// <summary>
        /// 计算图像的直方图
        /// </summary>
        /// <param name="img">图片</param>
        /// <returns>返回直方图量度</returns>

        public int[] GetRHisogram(Bitmap img)
        {
            return GetHisogram(img, 2);
        }
        public int[] GetGHisogram(Bitmap img)
        {
            return GetHisogram(img, 1);
        }
        public int[] GetBHisogram(Bitmap img)
        {
            return GetHisogram(img, 0);
        }
        public int[] GetGrayHisogram(Bitmap img)
        {
            return GetHisogram(img, 4);
        }
        private int[] GetHisogram(Bitmap img,Int32 type)
        {
            
            BitmapData data = img.LockBits(new System.Drawing.Rectangle(0, 0, img.Width, img.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            int[] histogram = new int[256];
            unsafe
            {
                byte* ptr = (byte*)data.Scan0;
                int remain = data.Stride - data.Width * 3;
                for (int i = 0; i < histogram.Length; i++)
                    histogram[i] = 0;
                for (int i = 0; i < data.Height; i++)
                {
                    for (int j = 0; j < data.Width; j++)
                    {
                        int mean =0;
                        if( type>=0 && type<=2)
                        {
                            mean = ptr[type];
                        }
                        else
                        { 
                            //Get Gray from B G R number
                            //int mean = ptr[0] + ptr[1] + ptr[2];
                            mean = ptr[0]  + ptr[1]  + ptr[2] ;
                            mean /= 3;
                        }
                        
                        histogram[mean]++;
                        ptr += 3;
                    }
                    ptr += remain;
                }
            }
            img.UnlockBits(data);
            return histogram;
        }
        /// <summary>
        /// 获取绝对值
        /// </summary>
        /// <param name="firstNum"></param>
        /// <param name="secondNum"></param>
        /// <returns></returns>
        private float GetAbs(int firstNum, int secondNum)
        {
            float abs = Math.Abs((float)firstNum - (float)secondNum);
            float result = Math.Max(firstNum, secondNum);
            if (result == 0)
                result = 1;
            return abs / result;
        }

        /// <summary>
        ///最终计算结果
        /// </summary>
        /// <param name="firstNum">图片一的直方图量度</param>
        /// <param name="scondNum">图片二的直方图量度</param>
        /// <returns>计算结果</returns>
        public float GetResult(int[] firstNum, int[] scondNum)
        {
            if (firstNum == null || scondNum == null)
                return 0;

            if (firstNum.Length != scondNum.Length)
            {
                return 0;
            }
            else
            {
                float result = 0;
                int j = firstNum.Length;
                for (int i = 0; i < j; i++)
                {
                    result += 1 - GetAbs(firstNum[i], scondNum[i]);
                    Console.WriteLine(i + "----" + result);
                }
                return result / j;
            }
        }

        /// <summary>
        /// 图片大小缩放(正方形)
        /// 作者：Jack Fan
        /// </summary>
        /// <param name="sideSize">指定大小</param>
        /// <param name="srcBMP">原始图片</param>
        /// <returns>返回缩放后的Bitmap图片</returns>
        public Bitmap ResizeBMP(int sideSize, Bitmap srcBMP)
        {
            Bitmap bmp = new Bitmap(sideSize, sideSize);

            Rectangle srcRec = new Rectangle(0, 0, srcBMP.Width, srcBMP.Height);
            Rectangle destRec = new Rectangle(0, 0, sideSize, sideSize);

            Graphics g = Graphics.FromImage(bmp);
            g.DrawImage(srcBMP, destRec, srcRec, GraphicsUnit.Pixel);
            g.Dispose();

            return bmp;
        }
        /// <summary>
        /// 图片大小缩放（矩形）
        /// </summary>
        /// <param name="height">指定的高</param>
        /// <param name="width">指定的宽</param>
        /// <param name="srcBMP">原始图片</param>
        /// <returns>返回缩放后的Bitmap图片</returns>
        public Bitmap ResizeBMP(int height, int width, Bitmap srcBMP)
        {
            Bitmap bmp = new Bitmap(height, width);
            Graphics g = Graphics.FromImage(bmp);

            Rectangle srcRec = new Rectangle(0, 0, srcBMP.Width, srcBMP.Height);
            Rectangle destRec = new Rectangle(0, 0, height, width);

            g.DrawImage(srcBMP, destRec, srcRec, GraphicsUnit.Pixel);
            g.Dispose();

            return bmp;
        }
    }
}
