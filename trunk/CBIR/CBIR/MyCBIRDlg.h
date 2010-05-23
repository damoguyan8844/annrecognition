// MyCBIRDlg.h : ͷ�ļ�
//

#pragma once

// CMyCBIRDlg �Ի���
class CMyCBIRDlg : public CDialog
{
// CBIR��ر���
private:
	Bitmap *m_pTargetPic;		// Ŀ��ͼƬ
	Bitmap *m_pLibPic[9];		// �Ҳ���ʾ��9��ͼƬ
	bool	m_bLibLoaded;		// ͼƬ���Ƿ����
	bool	m_bTargetLoaded;	// Ŀ��ͼƬ�Ƿ����
	bool	m_bSearched;		// ��ͼƬ�Ƿ������ɹ�
	int		m_iPicNum;			// ͼƬ���ͼƬ����
	int	   *m_pLBPBuf[200];		// ͼƬ���LBP�ײ���������
	int	   *m_pColorBuf[200];	// ͼƬ�����ɫ�ײ���������
	CArray<CString,CString&> fileSet;  // ����ͼƬ����ļ���
	//  �Ѵ��ڵĿ����� ���ڼ�����,���㹫ʽ��
	//	m_nLibType����((int)m_bTextureBased)<<7 |  m_nLBPMethod  | ((int)m_bColorBased)<<15 | m_nColorMethod<<8
	int		m_nLibType;			

	int		index[9];
	int		result[9];

// ����
public:
	CMyCBIRDlg(CWnd* pParent = NULL);	// ��׼���캯��

// �Ի�������
	enum { IDD = IDD_MYCBIR_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV ֧��


// ʵ��
protected:
	HICON m_hIcon;

	// ���ɵ���Ϣӳ�亯��
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOpenPic();
	afx_msg void OnBnClickedLoadLib();
	afx_msg void OnBnClickedSearch();
public:
	// �Ƿ��������
	BOOL m_bTextureBased;
	// ���õ�LBP���� 0 ����8���� 1�����ֵ+��ת������ 2�����ֵ����ת
	int m_nLBPMethod;
	BOOL m_bColorBased;
	int m_nColorMethod;

public:
	// ����ͼƬ����������
	void CreateLib();
	// ����ָ��ͼƬ��LBPֵ,����ֵ�����pLBP[256]
	void CaculateLBP(Bitmap* tempImage, int* pLBP);
	// ���Ҳ���ʾ9��ͼƬ
	void UpdateShowPics(void);
	// ����ָ��ͼƬ����ɫֱ��ͼ,����ֵ�����pColor[256]
	void CaculateColor(Bitmap* tempImage,int* pColor);
};
