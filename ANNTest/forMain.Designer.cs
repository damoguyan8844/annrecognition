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
            this.button7 = new System.Windows.Forms.Button();
            this.textTraingInputs = new System.Windows.Forms.TextBox();
            this.textBMPFolders = new System.Windows.Forms.TextBox();
            this.textSubsystemBMP = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.button8 = new System.Windows.Forms.Button();
            this.btTraining = new System.Windows.Forms.Button();
            this.textToPath = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.textSpeed = new System.Windows.Forms.TextBox();
            this.textAvrgDiff = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.chkAutoSave = new System.Windows.Forms.CheckBox();
            this.label7 = new System.Windows.Forms.Label();
            this.textParas = new System.Windows.Forms.TextBox();
            this.btnStop = new System.Windows.Forms.Button();
            this.textParaFactor = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.button9 = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(337, 12);
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
            this.button2.Location = new System.Drawing.Point(460, 13);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(117, 27);
            this.button2.TabIndex = 4;
            this.button2.Text = "BMP2TIF";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // textOutPutBMP
            // 
            this.textOutPutBMP.Location = new System.Drawing.Point(204, 13);
            this.textOutPutBMP.Name = "textOutPutBMP";
            this.textOutPutBMP.Size = new System.Drawing.Size(100, 21);
            this.textOutPutBMP.TabIndex = 5;
            this.textOutPutBMP.Text = "Sample2.bmp";
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(583, 13);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(114, 30);
            this.button3.TabIndex = 6;
            this.button3.Text = "BlackWhiteBMP";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.button3_Click);
            // 
            // textInputInt
            // 
            this.textInputInt.Location = new System.Drawing.Point(204, 45);
            this.textInputInt.Name = "textInputInt";
            this.textInputInt.Size = new System.Drawing.Size(106, 21);
            this.textInputInt.TabIndex = 7;
            this.textInputInt.Text = "128";
            // 
            // button4
            // 
            this.button4.Location = new System.Drawing.Point(337, 46);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(113, 28);
            this.button4.TabIndex = 8;
            this.button4.Text = "RevertBlackWhiteBMP";
            this.button4.UseVisualStyleBackColor = true;
            this.button4.Click += new System.EventHandler(this.button4_Click);
            // 
            // button5
            // 
            this.button5.Location = new System.Drawing.Point(715, 106);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(134, 21);
            this.button5.TabIndex = 9;
            this.button5.Text = "(Rate)SaveBlockToBMP";
            this.button5.UseVisualStyleBackColor = true;
            this.button5.Click += new System.EventHandler(this.button5_Click);
            // 
            // textLeft
            // 
            this.textLeft.Location = new System.Drawing.Point(204, 125);
            this.textLeft.Name = "textLeft";
            this.textLeft.Size = new System.Drawing.Size(62, 21);
            this.textLeft.TabIndex = 10;
            this.textLeft.Text = "0.25";
            // 
            // textTop
            // 
            this.textTop.Location = new System.Drawing.Point(276, 125);
            this.textTop.Name = "textTop";
            this.textTop.Size = new System.Drawing.Size(67, 21);
            this.textTop.TabIndex = 11;
            this.textTop.Text = "0.25";
            // 
            // textRight
            // 
            this.textRight.Location = new System.Drawing.Point(354, 125);
            this.textRight.Name = "textRight";
            this.textRight.Size = new System.Drawing.Size(72, 21);
            this.textRight.TabIndex = 12;
            this.textRight.Text = "0.75";
            // 
            // textBottom
            // 
            this.textBottom.Location = new System.Drawing.Point(432, 125);
            this.textBottom.Name = "textBottom";
            this.textBottom.Size = new System.Drawing.Size(65, 21);
            this.textBottom.TabIndex = 13;
            this.textBottom.Text = "0.75";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(202, 110);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(131, 12);
            this.label1.TabIndex = 14;
            this.label1.Text = "Left,Top,Right,Bottom";
            // 
            // button6
            // 
            this.button6.Location = new System.Drawing.Point(583, 45);
            this.button6.Name = "button6";
            this.button6.Size = new System.Drawing.Size(114, 29);
            this.button6.TabIndex = 15;
            this.button6.Text = "OCRFile";
            this.button6.UseVisualStyleBackColor = true;
            this.button6.Click += new System.EventHandler(this.button6_Click);
            // 
            // textOCRContent
            // 
            this.textOCRContent.Location = new System.Drawing.Point(466, 76);
            this.textOCRContent.Name = "textOCRContent";
            this.textOCRContent.Size = new System.Drawing.Size(317, 21);
            this.textOCRContent.TabIndex = 16;
            // 
            // labelPath
            // 
            this.labelPath.AutoSize = true;
            this.labelPath.Location = new System.Drawing.Point(335, 79);
            this.labelPath.Name = "labelPath";
            this.labelPath.Size = new System.Drawing.Size(119, 12);
            this.labelPath.TabIndex = 17;
            this.labelPath.Text = "OCRFile Full  Path ";
            // 
            // button7
            // 
            this.button7.Location = new System.Drawing.Point(221, 171);
            this.button7.Name = "button7";
            this.button7.Size = new System.Drawing.Size(100, 23);
            this.button7.TabIndex = 18;
            this.button7.Text = "BPEncode";
            this.button7.UseVisualStyleBackColor = true;
            this.button7.Click += new System.EventHandler(this.button7_Click);
            // 
            // textTraingInputs
            // 
            this.textTraingInputs.Location = new System.Drawing.Point(28, 200);
            this.textTraingInputs.MaxLength = 3276700;
            this.textTraingInputs.Multiline = true;
            this.textTraingInputs.Name = "textTraingInputs";
            this.textTraingInputs.Size = new System.Drawing.Size(822, 315);
            this.textTraingInputs.TabIndex = 19;
            // 
            // textBMPFolders
            // 
            this.textBMPFolders.Location = new System.Drawing.Point(28, 171);
            this.textBMPFolders.Name = "textBMPFolders";
            this.textBMPFolders.Size = new System.Drawing.Size(162, 21);
            this.textBMPFolders.TabIndex = 20;
            this.textBMPFolders.Text = "BMPFolders";
            // 
            // textSubsystemBMP
            // 
            this.textSubsystemBMP.Location = new System.Drawing.Point(29, 126);
            this.textSubsystemBMP.Name = "textSubsystemBMP";
            this.textSubsystemBMP.Size = new System.Drawing.Size(161, 21);
            this.textSubsystemBMP.TabIndex = 21;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(27, 110);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(89, 12);
            this.label2.TabIndex = 22;
            this.label2.Text = "SubSystem BMP ";
            // 
            // button8
            // 
            this.button8.Location = new System.Drawing.Point(715, 126);
            this.button8.Name = "button8";
            this.button8.Size = new System.Drawing.Size(134, 22);
            this.button8.TabIndex = 23;
            this.button8.Text = "(Long)SaveBlocToBMP";
            this.button8.UseVisualStyleBackColor = true;
            this.button8.Click += new System.EventHandler(this.button8_Click);
            // 
            // btTraining
            // 
            this.btTraining.Location = new System.Drawing.Point(636, 535);
            this.btTraining.Name = "btTraining";
            this.btTraining.Size = new System.Drawing.Size(109, 25);
            this.btTraining.TabIndex = 24;
            this.btTraining.Text = "Training";
            this.btTraining.UseVisualStyleBackColor = true;
            this.btTraining.Click += new System.EventHandler(this.btTraining_Click);
            // 
            // textToPath
            // 
            this.textToPath.Location = new System.Drawing.Point(514, 124);
            this.textToPath.Name = "textToPath";
            this.textToPath.Size = new System.Drawing.Size(124, 21);
            this.textToPath.TabIndex = 25;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(513, 109);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(47, 12);
            this.label3.TabIndex = 26;
            this.label3.Text = "To File";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(27, 156);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(71, 12);
            this.label4.TabIndex = 27;
            this.label4.Text = "BMP Folders";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(27, 521);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(89, 12);
            this.label5.TabIndex = 28;
            this.label5.Text = "Learning Speed";
            // 
            // textSpeed
            // 
            this.textSpeed.Location = new System.Drawing.Point(29, 538);
            this.textSpeed.Name = "textSpeed";
            this.textSpeed.Size = new System.Drawing.Size(100, 21);
            this.textSpeed.TabIndex = 29;
            this.textSpeed.Text = "0.05";
            // 
            // textAvrgDiff
            // 
            this.textAvrgDiff.Location = new System.Drawing.Point(163, 538);
            this.textAvrgDiff.Name = "textAvrgDiff";
            this.textAvrgDiff.Size = new System.Drawing.Size(117, 21);
            this.textAvrgDiff.TabIndex = 30;
            this.textAvrgDiff.Text = "0.0015";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(161, 523);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(119, 12);
            this.label6.TabIndex = 31;
            this.label6.Text = "Accept Average Diff";
            // 
            // chkAutoSave
            // 
            this.chkAutoSave.AutoSize = true;
            this.chkAutoSave.Checked = true;
            this.chkAutoSave.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkAutoSave.Location = new System.Drawing.Point(495, 540);
            this.chkAutoSave.Name = "chkAutoSave";
            this.chkAutoSave.Size = new System.Drawing.Size(102, 16);
            this.chkAutoSave.TabIndex = 32;
            this.chkAutoSave.Text = "AutoSaveParas";
            this.chkAutoSave.UseVisualStyleBackColor = true;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(312, 525);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(83, 12);
            this.label7.TabIndex = 33;
            this.label7.Text = "BP Paras File";
            // 
            // textParas
            // 
            this.textParas.Location = new System.Drawing.Point(308, 538);
            this.textParas.Name = "textParas";
            this.textParas.Size = new System.Drawing.Size(142, 21);
            this.textParas.TabIndex = 34;
            this.textParas.Text = "001278.dat";
            // 
            // btnStop
            // 
            this.btnStop.Location = new System.Drawing.Point(762, 536);
            this.btnStop.Name = "btnStop";
            this.btnStop.Size = new System.Drawing.Size(87, 23);
            this.btnStop.TabIndex = 35;
            this.btnStop.Text = "Stop";
            this.btnStop.UseVisualStyleBackColor = true;
            this.btnStop.Click += new System.EventHandler(this.btnStop_Click);
            // 
            // textParaFactor
            // 
            this.textParaFactor.Location = new System.Drawing.Point(456, 173);
            this.textParaFactor.Name = "textParaFactor";
            this.textParaFactor.Size = new System.Drawing.Size(141, 21);
            this.textParaFactor.TabIndex = 36;
            this.textParaFactor.Text = "16";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(371, 177);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(83, 12);
            this.label8.TabIndex = 37;
            this.label8.Text = "Divide Factor";
            // 
            // button9
            // 
            this.button9.Location = new System.Drawing.Point(715, 151);
            this.button9.Name = "button9";
            this.button9.Size = new System.Drawing.Size(134, 22);
            this.button9.TabIndex = 38;
            this.button9.Text = "SplitToFiles";
            this.button9.UseVisualStyleBackColor = true;
            this.button9.Click += new System.EventHandler(this.button9_Click);
            // 
            // formMain
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(862, 580);
            this.Controls.Add(this.button9);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.textParaFactor);
            this.Controls.Add(this.btnStop);
            this.Controls.Add(this.textParas);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.chkAutoSave);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.textAvrgDiff);
            this.Controls.Add(this.textSpeed);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.textToPath);
            this.Controls.Add(this.btTraining);
            this.Controls.Add(this.button8);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.textSubsystemBMP);
            this.Controls.Add(this.textBMPFolders);
            this.Controls.Add(this.textTraingInputs);
            this.Controls.Add(this.button7);
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
        private System.Windows.Forms.Button button7;
        private System.Windows.Forms.TextBox textTraingInputs;
        private System.Windows.Forms.TextBox textBMPFolders;
        private System.Windows.Forms.TextBox textSubsystemBMP;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button button8;
        private System.Windows.Forms.Button btTraining;
        private System.Windows.Forms.TextBox textToPath;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox textSpeed;
        private System.Windows.Forms.TextBox textAvrgDiff;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.CheckBox chkAutoSave;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox textParas;
        private System.Windows.Forms.Button btnStop;
        private System.Windows.Forms.TextBox textParaFactor;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Button button9;
    }
}

