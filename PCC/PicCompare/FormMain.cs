using System;

using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using Microsoft.WindowsMobile.Forms;

namespace PicCompare
{
    public partial class FormMain : Form
    {
        public FormMain()
        {
            InitializeComponent();
        }
        PicCompare.GetHisogram.GetHis getHis = new PicCompare.GetHisogram.GetHis();
        string pic1 = @"Storage Card/test.bmp";
        string pic2 = @"Storage Card/test2.bmp";
        int[] pic1t;
        int[] pic2t;

        string fileName1;
        string fileName2;
        private void menuItem2_Click(object sender, EventArgs e)
        {
            SelectPictureDialog spd = new SelectPictureDialog();
            spd.ShowDialog();
            fileName1 = spd.FileName;
            Bitmap bmp=new Bitmap(spd.FileName);
            pictureBox1.Image = bmp;//把处理后的图片放入pictureBox1进行预览
        }

        private void menuItem3_Click(object sender, EventArgs e)
        {
            SelectPictureDialog spd = new SelectPictureDialog();
            spd.ShowDialog();
            fileName2 = spd.FileName;
            Bitmap bmp = new Bitmap(spd.FileName);
            pictureBox2.Image = bmp;
        }

        private void menuItem4_Click(object sender, EventArgs e)
        {
            pictureBox1.Refresh();
            pictureBox2.Refresh();

            if (chkR.Checked)
                pic1t = getHis.GetRHisogram(getHis.Resized(fileName1, pic1));
            else if (chkG.Checked)
                pic1t = getHis.GetGHisogram(getHis.Resized(fileName1, pic1));
            else if (chkB.Checked)
                pic1t = getHis.GetBHisogram(getHis.Resized(fileName1, pic1));
            else
                pic1t = getHis.GetGrayHisogram(getHis.Resized(fileName1, pic1));//计算出图片一的直方图量度存放到一个pic1t的数组变量中

            if (chkR.Checked)
                pic2t = getHis.GetRHisogram(getHis.Resized(fileName2, pic2));
            else if (chkG.Checked)
                pic2t = getHis.GetGHisogram(getHis.Resized(fileName2, pic2));
            else if (chkB.Checked)
                pic2t = getHis.GetBHisogram(getHis.Resized(fileName2, pic2));
            else
                pic2t = getHis.GetGrayHisogram(getHis.Resized(fileName2, pic2));//计算出图片一的直方图量度存放到一个pic1t的数组变量中
            
            label2.Text = (getHis.GetResult(pic1t, pic2t) * 100).ToString() + "%";//计算最终结果
        }

        private Boolean m_isEnabled = true;
        private void SetEnable(Boolean isEnable)
        {
            m_isEnabled = isEnable;
        }
        private void chkR_CheckStateChanged(object sender, EventArgs e)
        {
            if (!m_isEnabled) return;
            SetEnable(false);
            if (chkR.Checked)
            {
                chkG.Checked = false;
                chkB.Checked = false;
                chkGray.Checked = false;
            }
            SetEnable(true);
        }

        private void chkG_CheckStateChanged(object sender, EventArgs e)
        {
            if (!m_isEnabled) return;
            SetEnable(false);
            if (chkG.Checked)
            {
                chkR.Checked = false;
                chkB.Checked = false;
                chkGray.Checked = false;
            }
            SetEnable(true);
        }

        private void chkB_CheckStateChanged(object sender, EventArgs e)
        {
            if (!m_isEnabled) return;
            SetEnable(false);
            if (chkB.Checked)
            {
                chkR.Checked = false;
                chkG.Checked = false;
                chkGray.Checked = false;
            }
            SetEnable(true);
        }

        private void chkGray_CheckStateChanged(object sender, EventArgs e)
        {
            if (!m_isEnabled) return;
            SetEnable(false);
            if (chkGray.Checked)
            {
                chkR.Checked = false;
                chkB.Checked = false;
                chkG.Checked = false;
            }
            SetEnable(true);
        }
    }
}