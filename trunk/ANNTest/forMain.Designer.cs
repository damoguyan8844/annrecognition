namespace ANNTest
{
    partial class formMain
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.button1 = new System.Windows.Forms.Button();
            this.textInputJPG = new System.Windows.Forms.TextBox();
            this.textInputBMP = new System.Windows.Forms.TextBox();
            this.textInputTIF = new System.Windows.Forms.TextBox();
            this.button2 = new System.Windows.Forms.Button();
            this.textOutPutBMP = new System.Windows.Forms.TextBox();
            this.button3 = new System.Windows.Forms.Button();
            this.textInputInt = new System.Windows.Forms.TextBox();
            this.button4 = new System.Windows.Forms.Button();
            this.button5 = new System.Windows.Forms.Button();
            this.textLeft = new System.Windows.Forms.TextBox();
            this.textTop = new System.Windows.Forms.TextBox();
            this.textRight = new System.Windows.Forms.TextBox();
            this.textBottom = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.button6 = new System.Windows.Forms.Button();
            this.textOCRContent = new System.Windows.Forms.TextBox();
            this.labelPath = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(359, 36);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(117, 28);
            this.button1.TabIndex = 0;
            this.button1.Text = "JPEG2BMP";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // textInputJPG
            // 
            this.textInputJPG.Location = new System.Drawing.Point(28, 16);
            this.textInputJPG.Name = "textInputJPG";
            this.textInputJPG.Size = new System.Drawing.Size(117, 21);
            this.textInputJPG.TabIndex = 1;
            this.textInputJPG.Text = "Sample.jpg";
            // 
            // textInputBMP
            // 
            this.textInputBMP.Location = new System.Drawing.Point(28, 43);
            this.textInputBMP.Name = "textInputBMP";
            this.textInputBMP.Size = new System.Drawing.Size(117, 21);
            this.textInputBMP.TabIndex = 2;
            this.textInputBMP.Text = "Sample.bmp";
            // 
            // textInputTIF
            // 
            this.textInputTIF.Location = new System.Drawing.Point(28, 70);
            this.textInputTIF.Name = "textInputTIF";
            this.textInputTIF.Size = new System.Drawing.Size(117, 21);
            this.textInputTIF.TabIndex = 3;
            this.textInputTIF.Text = "Sample.tif";
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(359, 66);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(117, 27);
            this.button2.TabIndex = 4;
            this.button2.Text = "BMP2TIF";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // textOutPutBMP
            // 
            this.textOutPutBMP.Location = new System.Drawing.Point(198, 43);
            this.textOutPutBMP.Name = "textOutPutBMP";
            this.textOutPutBMP.Size = new System.Drawing.Size(100, 21);
            this.textOutPutBMP.TabIndex = 5;
            this.textOutPutBMP.Text = "Sample2.bmp";
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(359, 99);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(114, 30);
            this.button3.TabIndex = 6;
            this.button3.Text = "BlackWhiteBMP";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.button3_Click);
            // 
            // textInputInt
            // 
            this.textInputInt.Location = new System.Drawing.Point(192, 108);
            this.textInputInt.Name = "textInputInt";
            this.textInputInt.Size = new System.Drawing.Size(106, 21);
            this.textInputInt.TabIndex = 7;
            this.textInputInt.Text = "128";
            // 
            // button4
            // 
            this.button4.Location = new System.Drawing.Point(359, 135);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(113, 28);
            this.button4.TabIndex = 8;
            this.button4.Text = "RevertBlackWhiteBMP";
            this.button4.UseVisualStyleBackColor = true;
            this.button4.Click += new System.EventHandler(this.button4_Click);
            // 
            // button5
            // 
            this.button5.Location = new System.Drawing.Point(359, 169);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(111, 34);
            this.button5.TabIndex = 9;
            this.button5.Text = "SaveBlockToBMP";
            this.button5.UseVisualStyleBackColor = true;
            this.button5.Click += new System.EventHandler(this.button5_Click);
            // 
            // textLeft
            // 
            this.textLeft.Location = new System.Drawing.Point(12, 177);
            this.textLeft.Name = "textLeft";
            this.textLeft.Size = new System.Drawing.Size(62, 21);
            this.textLeft.TabIndex = 10;
            this.textLeft.Text = "0.25";
            // 
            // textTop
            // 
            this.textTop.Location = new System.Drawing.Point(94, 177);
            this.textTop.Name = "textTop";
            this.textTop.Size = new System.Drawing.Size(67, 21);
            this.textTop.TabIndex = 11;
            this.textTop.Text = "0.25";
            // 
            // textRight
            // 
            this.textRight.Location = new System.Drawing.Point(176, 177);
            this.textRight.Name = "textRight";
            this.textRight.Size = new System.Drawing.Size(72, 21);
            this.textRight.TabIndex = 12;
            this.textRight.Text = "0.75";
            // 
            // textBottom
            // 
            this.textBottom.Location = new System.Drawing.Point(265, 177);
            this.textBottom.Name = "textBottom";
            this.textBottom.Size = new System.Drawing.Size(65, 21);
            this.textBottom.TabIndex = 13;
            this.textBottom.Text = "0.75";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 156);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(131, 12);
            this.label1.TabIndex = 14;
            this.label1.Text = "Left,Top,Right,Bottom";
            // 
            // button6
            // 
            this.button6.Location = new System.Drawing.Point(359, 225);
            this.button6.Name = "button6";
            this.button6.Size = new System.Drawing.Size(110, 29);
            this.button6.TabIndex = 15;
            this.button6.Text = "OCRFile";
            this.button6.UseVisualStyleBackColor = true;
            this.button6.Click += new System.EventHandler(this.button6_Click);
            // 
            // textOCRContent
            // 
            this.textOCRContent.Location = new System.Drawing.Point(12, 247);
            this.textOCRContent.Name = "textOCRContent";
            this.textOCRContent.Size = new System.Drawing.Size(317, 21);
            this.textOCRContent.TabIndex = 16;
            // 
            // labelPath
            // 
            this.labelPath.AutoSize = true;
            this.labelPath.Location = new System.Drawing.Point(11, 226);
            this.labelPath.Name = "labelPath";
            this.labelPath.Size = new System.Drawing.Size(119, 12);
            this.labelPath.TabIndex = 17;
            this.labelPath.Text = "OCRFile Full  Path ";
            // 
            // formMain
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(523, 393);
            this.Controls.Add(this.labelPath);
            this.Controls.Add(this.textOCRContent);
            this.Controls.Add(this.button6);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.textBottom);
            this.Controls.Add(this.textRight);
            this.Controls.Add(this.textTop);
            this.Controls.Add(this.textLeft);
            this.Controls.Add(this.button5);
            this.Controls.Add(this.button4);
            this.Controls.Add(this.textInputInt);
            this.Controls.Add(this.button3);
            this.Controls.Add(this.textOutPutBMP);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.textInputTIF);
            this.Controls.Add(this.textInputBMP);
            this.Controls.Add(this.textInputJPG);
            this.Controls.Add(this.button1);
            this.Name = "formMain";
            this.Text = "ANNTest";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.TextBox textInputJPG;
        private System.Windows.Forms.TextBox textInputBMP;
        private System.Windows.Forms.TextBox textInputTIF;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.TextBox textOutPutBMP;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.TextBox textInputInt;
        private System.Windows.Forms.Button button4;
        private System.Windows.Forms.Button button5;
        private System.Windows.Forms.TextBox textLeft;
        private System.Windows.Forms.TextBox textTop;
        private System.Windows.Forms.TextBox textRight;
        private System.Windows.Forms.TextBox textBottom;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button button6;
        private System.Windows.Forms.TextBox textOCRContent;
        private System.Windows.Forms.Label labelPath;
    }
}

