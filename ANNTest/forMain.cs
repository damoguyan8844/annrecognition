using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

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
            ANNWrapper.ConvertJPEG2BMP(textInputJPG.Text,textInputBMP.Text);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            ANNWrapper.ConvertBMP2TIF(textInputBMP.Text, textInputTIF.Text);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            ANNWrapper.BlackWhiteBMP(textInputBMP.Text,Int32.Parse(textInputInt.Text));
        }

        private void button4_Click(object sender, EventArgs e)
        {
            ANNWrapper.RevertBlackWhiteBMP(textInputBMP.Text);
        }

        private void button5_Click(object sender, EventArgs e)
        {
            ANNWrapper.SaveBlockToBMP(textInputBMP.Text,
                Double.Parse(textLeft.Text),
                Double.Parse(textTop.Text),
                Double.Parse(textRight.Text),
                Double.Parse(textBottom.Text), 
                textOutPutBMP.Text);
        }

        private void button6_Click(object sender, EventArgs e)
        {
            string strContent;
            labelPath.Text = Application.StartupPath + "\\" + textInputTIF.Text;
            ANNWrapper.OCRFile(Application.StartupPath +"\\"+ textInputTIF.Text, out strContent);
            textOCRContent.Text = strContent;
        }
    }
}