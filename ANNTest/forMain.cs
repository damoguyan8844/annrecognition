using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Runtime.InteropServices;

namespace ANNTest
{
    public partial class formMain : Form
    {
        public formMain()
        {
            InitializeComponent();
            ANNWrapper.SetLogHandler(new LogCallbackDelegate(LoggerFunction));

            DirectoryInfo Dir = new DirectoryInfo(Application.StartupPath + "\\ErrorRec");
            if (!Dir.Exists)
                Dir.Create();
            ANNWrapper.SetErrorRecordFolder(Application.StartupPath+"\\ErrorRec");
        }

        private static void LoggerFunction(Int32 logType,string message)
        {
            MessageBox.Show(message);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            ANNWrapper.ConvertJPEG2BMP(Application.StartupPath + "\\" + textInputJPG.Text, Application.StartupPath + "\\" + textInputBMP.Text);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            ANNWrapper.ConvertBMP2TIF(Application.StartupPath + "\\" + textInputBMP.Text, Application.StartupPath + "\\" + textInputTIF.Text);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            //DirectoryInfo Dir = new DirectoryInfo(Application.StartupPath + "\\");
            //foreach (FileInfo f in Dir.GetFiles("*Capture.bmp"))
            //{
            //   textInputBMP.Text = f.Name;
                ANNWrapper.BlackWhiteBMP(Application.StartupPath + "\\" + textInputBMP.Text, Int32.Parse(textInputInt.Text));
            //}
        }

        private void button4_Click(object sender, EventArgs e)
        {
             //DirectoryInfo Dir = new DirectoryInfo(Application.StartupPath + "\\");
             //foreach (FileInfo f in Dir.GetFiles("*Capture.bmp"))
             //{
             //    textInputBMP.Text = f.Name;
                 ANNWrapper.RevertBlackWhiteBMP(Application.StartupPath + "\\" + textInputBMP.Text);
             //}
        }

        private void button5_Click(object sender, EventArgs e)
        {
            ANNWrapper.SaveBlockToBMP(Application.StartupPath + "\\"+textSubsystemBMP.Text,
                Double.Parse(textLeft.Text),
                Double.Parse(textTop.Text),
                Double.Parse(textRight.Text),
                Double.Parse(textBottom.Text), 
                Application.StartupPath + "\\"+textToPath.Text);
        }

        private void button6_Click(object sender, EventArgs e)
        {
            string strContent;
            byte[] tempParas = new byte[1024];
            if (ANNWrapper.OCRFile(Application.StartupPath + "\\" + textInputTIF.Text, tempParas))
            {
                strContent = System.Text.Encoding.GetEncoding("GB2312").GetString(tempParas, 0, tempParas.Length);
                textOCRContent.Text = strContent;
            }
            else
            {
                MessageBox.Show("OCRFile Failure");
            }
        }

        private void EncodeBMPs(DirectoryInfo Dir, int dest)
        {
            double[] dblEncode = new double[64];

            try
            {
                string gridFile=Dir.FullName+"\\Grid.text";

                FileInfo fGrid = new FileInfo(gridFile);
                fGrid.Delete(); fGrid = null;

                foreach (FileInfo f in Dir.GetFiles("*.bmp"))
                {
                    IntPtr hdibHandle = ANNWrapper.ReadDIBFile(f.FullName);
                    if (ANNWrapper.BPEncode(hdibHandle, dblEncode, 0, 0, 0, 0, gridFile))
                    {
                        string strCodes = "";
                        foreach (double dblValue in dblEncode)
                        {
                            strCodes += dblValue.ToString() + ",";
                        }

                        strCodes += dest.ToString();
                        if (textTraingInputs.Lines.Length == 0)
                            textTraingInputs.AppendText(strCodes);
                        else
                            textTraingInputs.AppendText("\r\n" + strCodes);

                    }
                    ANNWrapper.ReleaseDIBFile(hdibHandle);
                }
            }
            catch (Exception exp)
            {
                MessageBox.Show(exp.Message);
            }   

        }
        private void button7_Click(object sender, EventArgs e)
        {
            if (textParas.Text.Length > 0)
                ANNWrapper.LoadBPParameters(Application.StartupPath + "\\" + textParas.Text);
            else
                ANNWrapper.InitBPParameters(64, 8, 4);

            
            DirectoryInfo Dir = new DirectoryInfo(Application.StartupPath + "\\"+textBMPFolders.Text);
            try
            {
                foreach (DirectoryInfo d in Dir.GetDirectories())       
                {
                    if(d.ToString()=="BMP0")
                        EncodeBMPs(d,0);
                    else if(d.ToString()=="BMP1")
                        EncodeBMPs(d,1);
                    else if(d.ToString()=="BMP2")
                        EncodeBMPs(d,2);
                    else if(d.ToString()=="BMP3")
                        EncodeBMPs(d,3);
                    else if(d.ToString()=="BMP4")
                        EncodeBMPs(d,4);
                    else if(d.ToString()=="BMP5")
                        EncodeBMPs(d,5);
                    else if(d.ToString()=="BMP6")
                        EncodeBMPs(d,6);
                    else if (d.ToString() == "BMP7")
                        EncodeBMPs(d,7);
                    else if(d.ToString()=="BMP8")
                        EncodeBMPs(d,8);
                    else if(d.ToString()== "BMP9")
                        EncodeBMPs(d,9);
                }
            }
            catch (Exception exp)
            {
                MessageBox.Show(exp.Message);
            }   
        }

        private void button8_Click(object sender, EventArgs e)
        {
            System.Drawing.Bitmap tempBit= new System.Drawing.Bitmap(Application.StartupPath + "\\" + textSubsystemBMP.Text);
            System.Drawing.Rectangle rect = new System.Drawing.Rectangle(
                Int32.Parse(textLeft.Text),
                Int32.Parse(textTop.Text),
                Int32.Parse(textRight.Text) - Int32.Parse(textLeft.Text),
                Int32.Parse(textBottom.Text) - Int32.Parse(textTop.Text)
                );

            System.Drawing.Bitmap block = tempBit.Clone(rect,tempBit.PixelFormat);
            ANNWrapper.SaveBlockToBMP3(block.GetHbitmap(), 0, 0, rect.Width, rect.Height, Application.StartupPath + "\\" + textToPath.Text);


            //ANNWrapper.SaveBlockToBMP3(block.GetHbitmap(),
            // Int32.Parse(textLeft.Text),
            // Int32.Parse(textTop.Text),
            // Int32.Parse(textRight.Text),
            // Int32.Parse(textBottom.Text),
            // Application.StartupPath + "\\" + textToPath.Text);

            //ANNWrapper.SaveBlockToBMP2(Application.StartupPath + "\\" + textSubsystemBMP.Text,
            // Int32.Parse(textLeft.Text),
            // Int32.Parse(textTop.Text),
            // Int32.Parse(textRight.Text),
            // Int32.Parse(textBottom.Text),
            // Application.StartupPath + "\\" + textToPath.Text);
        }

        private bool m_stop = false;
        private void btTraining_Click(object sender, EventArgs e)
        {
            m_stop = false;
           // btTraining.Enabled = false;

            if (textParas.Text.Length > 0)
                ANNWrapper.LoadBPParameters(Application.StartupPath + "\\" + textParas.Text);
            else
                ANNWrapper.InitBPParameters(64,8,4);

            if (textTraingInputs.Lines.Length<1)
                return ;

            MessageBox.Show(textTraingInputs.Lines.Length.ToString() + " Lines!");

            int divideFactor = Int32.Parse(textParaFactor.Text);
            int count = 0;

            ANNWrapper.InitTrainBPLearnSpeed(Double.Parse(textSpeed.Text));
            double accpt_diff = Double.Parse(textAvrgDiff.Text.ToString());

            double[] inputs = new double[64];
            double[] dests = new double[4];
            while(true)
            {
                double this_dif=0.0;
                foreach(string line in textTraingInputs.Lines)
                {
                    count++;

                    string[] strs = line.Split(',');
                    if(strs.Length!=65) continue; 
                    
                    for(int i=0;i<64;i++)
                    {
                        inputs[i] = Double.Parse(strs[i]) / divideFactor;
                    }

                    string dest=Convert.ToString(Int32.Parse(strs[64]), 2);

                    if (dest.Length > 0 && dest[dest.Length-1-0] == '1')
                        dests[0] = 1.0;
                    else
                        dests[0] = 0.0;

                    if (dest.Length > 1 && dest[dest.Length-1-1] == '1')
                        dests[1] = 1.0;
                    else
                        dests[1] = 0.0;

                    if (dest.Length > 2 && dest[dest.Length-1-2] == '1')
                        dests[2] = 1.0;
                    else
                        dests[2] = 0.0;

                    if (dest.Length > 3 && dest[dest.Length-1-3] == '1')
                        dests[3] = 1.0;
                    else
                        dests[3] = 0.0;

                    double dif=ANNWrapper.Training(inputs, dests);
                    this_dif += dif;
                }

                this_dif /= textTraingInputs.Lines.Length;

                if (chkAutoSave.Checked)
                {
                    if (textParas.Text.Length < 1)
                        textParas.Text = "Training.dat";

                    ANNWrapper.SaveBPParameters(Application.StartupPath + "\\" + textParas.Text);
                }
                textUnMatch.AppendText(this_dif.ToString()+"\r\n");
                btTraining.Text=this_dif.ToString();
                btTraining.Update();
                if(this_dif<=accpt_diff || m_stop==true)
                {
                    ANNWrapper.SaveBPParameters(Application.StartupPath + "\\" + textParas.Text);
                    break;
                }
            }
            btTraining.Enabled =true;
            btTraining.Text = "Train(" + count.ToString()+")";
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            if (textParas.Text.Length < 1)
            {
                MessageBox.Show("Para Setting Is Empty");
                return;
            }
            if (textTraingInputs.Lines.Length < 1)
            {
                MessageBox.Show("Test Input Is Empty");
                return;
            }

            ANNWrapper.LoadBPParameters(Application.StartupPath + "\\" + textParas.Text);
            //ANNWrapper.PrintBPParameters("C:\\TypePara.text");

            double[] inputs = new double[64];
            double[] dests = new double[4];
            
            int divideFactor = Int32.Parse(textParaFactor.Text);

            
            int matchCount=0;
            foreach (string line in textTraingInputs.Lines)
            {
                string[] strs = line.Split(',');
                if (strs.Length != 65) continue;

                for (int i = 0; i < 64; i++)
                {
                    inputs[i] = Double.Parse(strs[i]) / divideFactor;
                }

               /* string dest = Convert.ToString(Int32.Parse(strs[64]), 2);*/

                for (int i = 0; i < 4; i++)
                {
                    dests[i] = 0.0;
                }

                if (!ANNWrapper.Recognition(inputs, dests))
                {
                    MessageBox.Show("Recognition Error:" + line);
                    continue;
                }
                Int32 dest = 0;
                if (dests[0] >0.5)
                    dest+=1;
                if (dests[1] > 0.5)
                    dest += 2;
                if (dests[2] > 0.5)
                    dest += 4;
                if (dests[3] > 0.5)
                    dest += 8;

                if (dest == Int32.Parse(strs[64]))
                    matchCount += 1;
                else
                {
                    textUnMatch.AppendText(line + "|" + dests[0].ToString() + "," + dests[1].ToString() + "," + dests[2].ToString() + "," + dests[3].ToString() + ""+"\r\n");
                }
            }
            double dblRate=matchCount * 100.0 / textTraingInputs.Lines.Length;
            MessageBox.Show("%"+dblRate.ToString() + "(" + matchCount.ToString() + "/" + textTraingInputs.Lines.Length + ")");
        }

        private void button9_Click(object sender, EventArgs e)
        {
           DirectoryInfo Dir = new DirectoryInfo(Application.StartupPath + "\\");
           foreach (FileInfo f in Dir.GetFiles("*Capture.bmp"))
           {

            try
            {   
                    textToPath.Text = f.Name;

                    IntPtr hdibHandle = ANNWrapper.ReadDIBFile(Application.StartupPath + "\\" + textToPath.Text);

                    ANNWrapper.Convert256toGray(hdibHandle);

                    ANNWrapper.SaveDIB(hdibHandle, Application.StartupPath + "\\Convert256toGray.bmp");

                    ANNWrapper.ConvertGrayToWhiteBlack(hdibHandle);

                    ANNWrapper.SaveDIB(hdibHandle, Application.StartupPath + "\\ConvertGrayToWhiteBlack.bmp");

                    //ANNWrapper.GradientSharp(hdibHandle);
                    ANNWrapper.RemoveScatterNoise(hdibHandle);

                    ANNWrapper.SaveDIB(hdibHandle, Application.StartupPath + "\\RemoveScatterNoise.bmp");

                    //ANNWrapper.SlopeAdjust(hdibHandle);

                    Int32 charRectID = ANNWrapper.CharSegment(hdibHandle);

                    if (charRectID >= 0)
                    {
                        //ANNWrapper.StdDIBbyRect(hdibHandle, charRectID, 16, 16);
                        IntPtr newHdibHandle = ANNWrapper.AutoAlign(hdibHandle, charRectID);
                        ANNWrapper.SaveSegment(newHdibHandle, charRectID, Application.StartupPath + "\\");
                        ANNWrapper.ReleaseDIBFile(newHdibHandle);
                    }
                    else
                    {
                        MessageBox.Show("CharSegment Step False");
                    }

                    ANNWrapper.ReleaseDIBFile(hdibHandle);
                }
                catch (Exception exp)
                {
                    MessageBox.Show(textToPath.Text);
                }
            }
            
        }

        private void textParaFactor_TextChanged(object sender, EventArgs e)
        {

        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void button10_Click(object sender, EventArgs e)
        {
            try
            {
                int[] intRes = new int[64];

                ANNWrapper.BlackWhiteBMP(Application.StartupPath + "\\" + textToPath.Text, Int32.Parse(textInputInt.Text));

                IntPtr hdibHandle = ANNWrapper.ReadDIBFile(Application.StartupPath + "\\" + textToPath.Text);

                ANNWrapper.Convert256toGray(hdibHandle);

                ANNWrapper.SaveDIB(hdibHandle, Application.StartupPath + "\\Convert256toGray.bmp");

                ANNWrapper.ConvertGrayToWhiteBlack(hdibHandle);

                ANNWrapper.SaveDIB(hdibHandle, Application.StartupPath + "\\ConvertGrayToWhiteBlack.bmp");

                //ANNWrapper.GradientSharp(hdibHandle);
                ANNWrapper.RemoveScatterNoise(hdibHandle);

                ANNWrapper.SaveDIB(hdibHandle, Application.StartupPath + "\\RemoveScatterNoise.bmp");

                //ANNWrapper.SlopeAdjust(hdibHandle);

                Int32 charRectID = ANNWrapper.CharSegment(hdibHandle);

                if (charRectID >= 0)
                {
                    ANNWrapper.LoadBPParameters(Application.StartupPath + "\\" + textParas.Text);
           
                    //ANNWrapper.StdDIBbyRect(hdibHandle, charRectID, 16, 16);
                    IntPtr newHdibHandle = ANNWrapper.AutoAlign(hdibHandle, charRectID);
                    ANNWrapper.SaveDIB(newHdibHandle, Application.StartupPath + "\\AutoAlign.bmp");
                    //charRectID = ANNWrapper.CharSegment(newHdibHandle);

                    if (charRectID >= 0)
                    {
                        //ANNWrapper.SaveSegment(newHdibHandle, charRectID, Application.StartupPath + "\\");
                        if(ANNWrapper.Recognition_EX(newHdibHandle, charRectID, intRes))
                        {
                            string res = "";
                            foreach (int value in intRes)
                            {
                                if (value == -1)
                                    break;
                                res += value.ToString();
                            }

                            MessageBox.Show(res);
                        }
                        else
                        {
                            MessageBox.Show("Recognition Failure");
                        }
                    }

                    ANNWrapper.ReleaseDIBFile(newHdibHandle);
                }
                else
                {
                    MessageBox.Show("CharSegment Step False");
                }

                ANNWrapper.ReleaseDIBFile(hdibHandle);
            }
            catch (Exception exp)
            {
                MessageBox.Show(textToPath.Text);
            }
        }

        private void button11_Click(object sender, EventArgs e)
        {

            try
            {
                DirectoryInfo Dir = new DirectoryInfo(Application.StartupPath + "\\" + textBMPFolders.Text);
           
                foreach (FileInfo f in Dir.GetFiles("*.bmp"))
                {
                    textInputBMP.Text = textBMPFolders.Text+"\\"+f.Name;
                    button4_Click(sender, e);
                }
            }
            catch (Exception exp)
            {
                MessageBox.Show(textToPath.Text);
            }
        }

        private void button12_Click(object sender, EventArgs e)
        {
            try
            {

                DirectoryInfo DirNew = new DirectoryInfo(Application.StartupPath + "\\" + textBMPFolders.Text + "\\BMPFolders");
                if (!DirNew.Exists)
                    DirNew.Create();

                DirectoryInfo Dir = new DirectoryInfo(Application.StartupPath + "\\" + textBMPFolders.Text);

                foreach (FileInfo f in Dir.GetFiles("*.bmp"))
                {
                    textToPath.Text = textBMPFolders.Text + "\\" + f.Name;

                    ANNWrapper.BlackWhiteBMP(Application.StartupPath + "\\" + textToPath.Text, Int32.Parse(textInputInt.Text));

                    IntPtr hdibHandle = ANNWrapper.ReadDIBFile(Application.StartupPath + "\\" + textToPath.Text);

                    ANNWrapper.Convert256toGray(hdibHandle);

                    ANNWrapper.SaveDIB(hdibHandle, Application.StartupPath + "\\Convert256toGray.bmp");

                    ANNWrapper.ConvertGrayToWhiteBlack(hdibHandle);

                    ANNWrapper.SaveDIB(hdibHandle, Application.StartupPath + "\\ConvertGrayToWhiteBlack.bmp");

                    //ANNWrapper.GradientSharp(hdibHandle);
                    ANNWrapper.RemoveScatterNoise(hdibHandle);

                    ANNWrapper.SaveDIB(hdibHandle, Application.StartupPath + "\\RemoveScatterNoise.bmp");

                    //ANNWrapper.SlopeAdjust(hdibHandle);

                    Int32 charRectID = ANNWrapper.CharSegment(hdibHandle);

                    if (charRectID >= 0)
                    {
                        IntPtr newHdibHandle = ANNWrapper.AutoAlign(hdibHandle, charRectID);
                        ANNWrapper.SaveDIB(newHdibHandle, Application.StartupPath + "\\AutoAlign.bmp");
                        ANNWrapper.SaveSegment(newHdibHandle, charRectID, Application.StartupPath + "\\" + textBMPFolders.Text+"\\BMPFolders");
                        ANNWrapper.ReleaseDIBFile(newHdibHandle);
                    }
                    else
                    {
                        MessageBox.Show("CharSegment Step Failure !");
                    }

                    ANNWrapper.ReleaseDIBFile(hdibHandle);
                }
            }
            catch (Exception exp)
            {
                MessageBox.Show(textToPath.Text);
            }
        }

        private void button13_Click(object sender, EventArgs e)
        {
            try
            {
                DirectoryInfo DirNew = new DirectoryInfo(Application.StartupPath + "\\" + textBMPFolders.Text + "\\BMPFolders");
                if (!DirNew.Exists)
                    DirNew.Create();

                DirectoryInfo Dir = new DirectoryInfo(Application.StartupPath + "\\" + textBMPFolders.Text);

                foreach (FileInfo f in Dir.GetFiles("*.bmp"))
                {
                    textToPath.Text = textBMPFolders.Text + "\\" + f.Name;

                    //ANNWrapper.BlackWhiteBMP(Application.StartupPath + "\\" + textToPath.Text, Int32.Parse(textInputInt.Text));

                    IntPtr hdibHandle = ANNWrapper.ReadDIBFile(Application.StartupPath + "\\" + textToPath.Text);

                    ANNWrapper.Convert256toGray(hdibHandle);

                    //ANNWrapper.SaveDIB(hdibHandle, Application.StartupPath + "\\Convert256toGray.bmp");

                    ANNWrapper.ConvertGrayToWhiteBlack(hdibHandle);

                    //ANNWrapper.SaveDIB(hdibHandle, Application.StartupPath + "\\ConvertGrayToWhiteBlack.bmp");

                    ////ANNWrapper.GradientSharp(hdibHandle);
                    ANNWrapper.RemoveScatterNoise(hdibHandle);

                    //ANNWrapper.SaveDIB(hdibHandle, Application.StartupPath + "\\RemoveScatterNoise.bmp");

                    //ANNWrapper.SlopeAdjust(hdibHandle);

                    Int32 charRectID = ANNWrapper.CharSegment(hdibHandle);

                    if (charRectID >= 0)
                    {
                        IntPtr newHdibHandle = ANNWrapper.AutoAlign(hdibHandle, charRectID);
                        ANNWrapper.SaveDIB(newHdibHandle, Application.StartupPath + "\\AutoAlign.bmp");
                        ANNWrapper.SaveSegment(newHdibHandle, charRectID, Application.StartupPath + "\\" + textBMPFolders.Text + "\\BMPFolders");
                        ANNWrapper.ReleaseDIBFile(newHdibHandle);
                    }
                    else
                    {
                        MessageBox.Show("CharSegment Step Failure !");
                    }

                    ANNWrapper.ReleaseDIBFile(hdibHandle);
                }
            }
            catch (Exception exp)
            {
                MessageBox.Show(textToPath.Text);
            }
        }

        private void button14_Click(object sender, EventArgs e)
        {
            if (textParas.Text.Length > 0)
                ANNWrapper.LoadBPParameters(Application.StartupPath + "\\" + textParas.Text);
            else
                ANNWrapper.InitBPParameters(64, 8, 4);

            ANNWrapper.PrintBPParameters(Application.StartupPath + "\\" + textParas.Text+"Print.txt");
        }
    }
}