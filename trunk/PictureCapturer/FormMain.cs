using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace WindowsApplication1
{
	/// <summary>
	/// Form2 的摘要说明。
	/// </summary>
	public class FormMain : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Button btnStartCapture;
		private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.PictureBox pictureBoxImage;
        private Button btnSave;
        private TextBox textFileName;
		/// <summary>
		/// 必需的设计器变量。
		/// </summary>
		private System.ComponentModel.Container components = null;

		public FormMain()
		{
			//
			// Windows 窗体设计器支持所必需的
			//
			InitializeComponent();

			//
			// TODO: 在 InitializeComponent 调用后添加任何构造函数代码
			//
		}
		Bitmap cc;
		public FormMain(Bitmap xx)
		{
			//
			// Windows 窗体设计器支持所必需的
			//
			InitializeComponent();
cc=xx;
			//
			// TODO: 在 InitializeComponent 调用后添加任何构造函数代码
			//
		}

		/// <summary>
		/// 清理所有正在使用的资源。
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if(components != null)
				{
					components.Dispose();
				}
			}
			base.Dispose( disposing );
		}

		#region Windows 窗体设计器生成的代码
		/// <summary>
		/// 设计器支持所需的方法 - 不要使用代码编辑器修改
		/// 此方法的内容。
		/// </summary>
		private void InitializeComponent()
		{
            this.btnStartCapture = new System.Windows.Forms.Button();
            this.panel1 = new System.Windows.Forms.Panel();
            this.pictureBoxImage = new System.Windows.Forms.PictureBox();
            this.btnSave = new System.Windows.Forms.Button();
            this.textFileName = new System.Windows.Forms.TextBox();
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxImage)).BeginInit();
            this.SuspendLayout();
            // 
            // btnStartCapture
            // 
            this.btnStartCapture.Location = new System.Drawing.Point(40, 392);
            this.btnStartCapture.Name = "btnStartCapture";
            this.btnStartCapture.Size = new System.Drawing.Size(91, 23);
            this.btnStartCapture.TabIndex = 1;
            this.btnStartCapture.Text = "StartCapture";
            this.btnStartCapture.Click += new System.EventHandler(this.btnCapture_Click);
            // 
            // panel1
            // 
            this.panel1.AutoScroll = true;
            this.panel1.Controls.Add(this.pictureBoxImage);
            this.panel1.Location = new System.Drawing.Point(32, 24);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(344, 168);
            this.panel1.TabIndex = 2;
            // 
            // pictureBoxImage
            // 
            this.pictureBoxImage.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pictureBoxImage.Location = new System.Drawing.Point(0, 0);
            this.pictureBoxImage.Name = "pictureBoxImage";
            this.pictureBoxImage.Size = new System.Drawing.Size(344, 168);
            this.pictureBoxImage.TabIndex = 0;
            this.pictureBoxImage.TabStop = false;
            // 
            // btnSave
            // 
            this.btnSave.Location = new System.Drawing.Point(548, 392);
            this.btnSave.Name = "btnSave";
            this.btnSave.Size = new System.Drawing.Size(102, 23);
            this.btnSave.TabIndex = 4;
            this.btnSave.Text = "SaveToFile";
            this.btnSave.UseVisualStyleBackColor = true;
            this.btnSave.Click += new System.EventHandler(this.btnSave_Click);
            // 
            // textFileName
            // 
            this.textFileName.Location = new System.Drawing.Point(189, 394);
            this.textFileName.Name = "textFileName";
            this.textFileName.Size = new System.Drawing.Size(344, 21);
            this.textFileName.TabIndex = 5;
            this.textFileName.Text = "Capture.bmp";
            // 
            // FormMain
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(6, 14);
            this.ClientSize = new System.Drawing.Size(725, 452);
            this.Controls.Add(this.textFileName);
            this.Controls.Add(this.btnSave);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.btnStartCapture);
            this.Name = "FormMain";
            this.Text = "Main Form";
            this.Load += new System.EventHandler(this.Form2_Load);
            this.panel1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxImage)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion


        private int PicCount = 0;

		private void Form2_Load(object sender, System.EventArgs e)
		{
			pictureBoxImage.Image=cc;
		}

        private void btnCapture_Click(object sender, System.EventArgs e)
		{
            PicCount += 1;
			ScreenCap.FormCapture  fc=new ScreenCap.FormCapture();
			fc.dd+=new ScreenCap.FormCapture.xx(fc_dd);
			fc.Show();

            textFileName.Text = PicCount.ToString() + "Capture.bmp";
		}

		private void fc_dd(Bitmap image)
		{
			pictureBoxImage.Image=image;
		}

        private void btnSave_Click(object sender, EventArgs e)
        {
            pictureBoxImage.Image.Save(Application.StartupPath+"\\"+textFileName.Text);
        }
	}
}
