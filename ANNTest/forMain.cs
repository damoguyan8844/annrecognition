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
            ANNWrapper.BlackWhiteBMP(Application.StartupPath + "\\" + textInputBMP.Text, Int32.Parse(textInputInt.Text));
        }

        private void button4_Click(object sender, EventArgs e)
        {
            ANNWrapper.RevertBlackWhiteBMP(Application.StartupPath + "\\" + textInputBMP.Text);
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
            labelPath.Text = Application.StartupPath + "\\" + textInputTIF.Text;
            ANNWrapper.OCRFile(Application.StartupPath +"\\"+ textInputTIF.Text, out strContent);
            textOCRContent.Text = strContent;
        }

        private void EncodeBMPs(DirectoryInfo Dir, int dest)
        {
            double[] dblEncode = new double[64];

            try
            {
                foreach (FileInfo f in Dir.GetFiles("*.bmp"))
                {
                    IntPtr hdibHandle = ANNWrapper.ReadDIBFile(f.FullName);
                    if (ANNWrapper.BPEncode(hdibHandle, dblEncode, 0, 0, 0, 0))
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
                ANNWrapper.InitBPParameters(64, 8, 4, null, null, null, null);


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
            ANNWrapper.SaveBlockToBMP2(Application.StartupPath + "\\"+textSubsystemBMP.Text,
             Int32.Parse(textLeft.Text),
             Int32.Parse(textTop.Text),
             Int32.Parse(textRight.Text),
             Int32.Parse(textBottom.Text),
             Application.StartupPath + "\\"+textToPath.Text);
        }

        private bool m_stop = false;
        private void btTraining_Click(object sender, EventArgs e)
        {
            m_stop = false;
            btTraining.Enabled = false;

            if (textParas.Text.Length > 0)
                ANNWrapper.LoadBPParameters(Application.StartupPath + "\\" + textParas.Text);
            else
                ANNWrapper.InitBPParameters(64,8,4,null,null,null,null);

            if (textTraingInputs.Lines.Length<1)
                return ;
            
            int divideFactor = Int32.Parse(textParaFactor.Text);

            ANNWrapper.InitTrainBPLearnSpeed(Double.Parse(textSpeed.Text));
            double accpt_diff = Double.Parse(textAvrgDiff.Text.ToString());

            double[] inputs = new double[64];
            double[] dests = new double[4];
            while(true)
            {
                double this_dif=0.0;
                foreach(string line in textTraingInputs.Lines)
                {
                    string[] strs = line.Split(',');
                    if(strs.Length!=65) continue; 
                    
                    for(int i=0;i<64;i++)
                    {
                        inputs[i] = Double.Parse(strs[i]) / divideFactor;
                    }

                    string dest=Convert.ToString(Int32.Parse(strs[64]), 2);

                    if (dest.Length > 0 && dest[0] == '1')
                        dests[0] = 1.0;
                    else
                        dests[0] = 0.0;

                    if (dest.Length > 1 && dest[1] == '1')
                        dests[1] = 1.0;
                    else
                        dests[1] = 0.0;

                    if (dest.Length > 2 && dest[2] == '1')
                        dests[2] = 1.0;
                    else
                        dests[2] = 0.0;

                    if (dest.Length > 3 && dest[3] == '1')
                        dests[3] = 1.0;
                    else
                        dests[3] = 0.0;

                    double dif=ANNWrapper.Training(inputs, dests);
                    this_dif += dif;
                }

                this_dif /= textTraingInputs.Lines.Length;

                if (chkAutoSave.Checked)
                    ANNWrapper.SaveBPParameters(Application.StartupPath + "\\" + textParas.Text);

                btTraining.Text=this_dif.ToString();
                if(this_dif<=accpt_diff || m_stop==true)
                {
                    ANNWrapper.SaveBPParameters(Application.StartupPath + "\\" + textParas.Text);
                    break;
                }
            }
            btTraining.Enabled =true;
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            m_stop = true;
        }

        private void button9_Click(object sender, EventArgs e)
        {
            try
            {
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
            catch(Exception exp)
            {
                MessageBox.Show(exp.Message);
            }
        }
    }
}