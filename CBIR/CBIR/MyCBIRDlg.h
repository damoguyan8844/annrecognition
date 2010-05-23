// MyCBIRDlg.h : 头文件
//

#pragma once

// CMyCBIRDlg 对话框
class CMyCBIRDlg : public CDialog
{
// CBIR相关变量
private:
	Bitmap *m_pTargetPic;		// 目标图片
	Bitmap *m_pLibPic[9];		// 右侧显示的9张图片
	bool	m_bLibLoaded;		// 图片库是否加载
	bool	m_bTargetLoaded;	// 目标图片是否加载
	bool	m_bSearched;		// 该图片是否搜索成功
	int		m_iPicNum;			// 图片库的图片数量
	int	   *m_pLBPBuf[200];		// 图片库的LBP底层特征集合
	int	   *m_pColorBuf[200];	// 图片库的颜色底层特征集合
	CArray<CString,CString&> fileSet;  // 保存图片库的文件名
	//  已存在的库类型 用于检查更新,计算公式是
	//	m_nLibType等于((int)m_bTextureBased)<<7 |  m_nLBPMethod  | ((int)m_bColorBased)<<15 | m_nColorMethod<<8
	int		m_nLibType;			

	int		index[9];
	int		result[9];

// 构造
public:
	CMyCBIRDlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
	enum { IDD = IDD_MYCBIR_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOpenPic();
	afx_msg void OnBnClickedLoadLib();
	afx_msg void OnBnClickedSearch();
public:
	// 是否基于纹理
	BOOL m_bTextureBased;
	// 调用的LBP方法 0 代表8领域 1代表插值+旋转不变性 2代表插值不旋转
	int m_nLBPMethod;
	BOOL m_bColorBased;
	int m_nColorMethod;

public:
	// 建立图片库特征集合
	void CreateLib();
	// 计算指定图片的LBP值,返回值存放在pLBP[256]
	void CaculateLBP(Bitmap* tempImage, int* pLBP);
	// 在右侧显示9张图片
	void UpdateShowPics(void);
	// 计算指定图片的颜色直方图,返回值存放在pColor[256]
	void CaculateColor(Bitmap* tempImage,int* pColor);
};
