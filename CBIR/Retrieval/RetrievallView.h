#if !defined(AFX_RETRIEVALLVIEW_H__0D9D7609_89D7_4D03_83A9_ABB468635411__INCLUDED_)
#define AFX_RETRIEVALLVIEW_H__0D9D7609_89D7_4D03_83A9_ABB468635411__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// RetrievallView.h : header file
//
#include "retrievalDoc.h"

/////////////////////////////////////////////////////////////////////////////
// CRetrievallView view

class CRetrievallView : public CScrollView
{
protected:
	CRetrievallView();           // protected constructor used by dynamic creation
	DECLARE_DYNCREATE(CRetrievallView)

// Attributes
public:
	CRetrievalDoc* GetDocument();
    float mintwo(float a,float b){return a<b?a:b;}

	float Caculate(int,int,int,int);
	float distance(int,int,int,int);
	float distance_weight(int,int,int,int);

    void RGBToMTM(int r,int g,int b,double *h,double *v,double *c);
	void RGBToHSV(int r,int g,int b,double *h,double *s,double *v);

	int mymin(int,int,int);
	int mymax(int,int,int);
	double mtm_fun(double);

	void hsv_general(int);
    void hsv_succession(int);
	void hsv_centerM(int);
    
	void mtm_general(int);
    void mtm_succession(int);
	void mtm_centerM(int);
  
	float result_te[40];
	int cixu[40];
	float tezheng[3][12];

    void sortimage(int);
	int m_exam;
	bool if_lbd;
	bool ifSelect;

	
// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CRetrievallView)
	protected:
	virtual void OnDraw(CDC* pDC);      // overridden to draw this view
	virtual void OnInitialUpdate();     // first time after construct
	//}}AFX_VIRTUAL

// Implementation
protected:
	virtual ~CRetrievallView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

	// Generated message map functions
	//{{AFX_MSG(CRetrievallView)
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnContextMenu(CWnd* pWnd, CPoint point);
	afx_msg void OnHSV_GEN_OU();
	afx_msg void OnHSV_GEN_QUAN();
	afx_msg void OnHSV_GEN_JIAO();
	afx_msg void OnHSV_SUC_OU();
	afx_msg void OnHSV_CEN_OU();
	afx_msg void OnMTM_GEN_OU();
	afx_msg void OnMTM_GEN_QUAN();
	afx_msg void OnMTM_GEN_JIAO();
	afx_msg void OnMTM_SUC_OU();
	afx_msg void OnMTM_CEN_OU();
	afx_msg void OnHSV1_2();
	afx_msg void OnHSV2_2();
	afx_msg void OnHSV3_2();
	afx_msg void OnHSV4_2();
	afx_msg void OnHSV4_3();
	afx_msg void OnHSV5_2();
	afx_msg void OnRotate();
	afx_msg void OnBig();
	afx_msg void OnSmall();
	afx_msg void OnMove();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_RETRIEVALLVIEW_H__0D9D7609_89D7_4D03_83A9_ABB468635411__INCLUDED_)
