; CLW file contains information for the MFC ClassWizard

[General Info]
Version=1
LastClass=CRetrievallView
LastTemplate=CScrollView
NewFileInclude1=#include "stdafx.h"
NewFileInclude2=#include "retrieval.h"
LastPage=0

ClassCount=7
Class1=CRetrievalApp
Class2=CRetrievalDoc
Class3=CRetrievallView
Class4=CMainFrame

ResourceCount=3
Resource1=IDR_MENU1
Class5=CAboutDlg
Class6=ResultView
Class7=CResultView
Resource2=IDR_MAINFRAME
Resource3=IDD_ABOUTBOX

[CLS:CRetrievalApp]
Type=0
HeaderFile=retrieval.h
ImplementationFile=retrieval.cpp
Filter=N

[CLS:CRetrievalDoc]
Type=0
HeaderFile=retrievalDoc.h
ImplementationFile=retrievalDoc.cpp
Filter=N
LastObject=CRetrievalDoc
BaseClass=CDocument
VirtualFilter=DC

[CLS:CMainFrame]
Type=0
HeaderFile=MainFrm.h
ImplementationFile=MainFrm.cpp
Filter=T
LastObject=ID_small
BaseClass=CFrameWnd
VirtualFilter=fWC




[CLS:CAboutDlg]
Type=0
HeaderFile=retrieval.cpp
ImplementationFile=retrieval.cpp
Filter=D

[DLG:IDD_ABOUTBOX]
Type=1
Class=CAboutDlg
ControlCount=4
Control1=IDC_STATIC,static,1342177283
Control2=IDC_STATIC,static,1342308480
Control3=IDC_STATIC,static,1342308352
Control4=IDOK,button,1342373889

[MNU:IDR_MAINFRAME]
Type=1
Class=CMainFrame
Command1=ID_FILE_NEW
Command2=ID_FILE_OPEN
Command3=ID_FILE_SAVE
Command4=ID_FILE_SAVE_AS
Command5=ID_FILE_PRINT
Command6=ID_FILE_PRINT_PREVIEW
Command7=ID_FILE_PRINT_SETUP
Command8=ID_FILE_MRU_FILE1
Command9=ID_APP_EXIT
Command10=ID_EDIT_UNDO
Command11=ID_EDIT_CUT
Command12=ID_EDIT_COPY
Command13=ID_EDIT_PASTE
Command14=ID_VIEW_TOOLBAR
Command15=ID_VIEW_STATUS_BAR
Command16=ID_WINDOW_SPLIT
Command17=ID_APP_ABOUT
Command18=ID_WINTER
Command19=ID_FLAG
Command20=ID_FLOWER
Command21=ID_MENUITEM32776
Command22=ID_HSV1_2
Command23=ID_MENUITEM32777
Command24=ID_HSV2_2
Command25=ID_MENUITEM32778
Command26=ID_HSV3_2
Command27=ID_MENUITEM32779
Command28=ID_HSV4_2
Command29=ID_HSV4_3
Command30=ID_MENUITEM32780
Command31=ID_HSV5_2
Command32=ID_MTM1
Command33=ID_MTM2
Command34=ID_MTM3
Command35=ID_MTM4
Command36=ID_MTM5
CommandCount=36

[ACL:IDR_MAINFRAME]
Type=1
Class=CMainFrame
Command1=ID_FILE_NEW
Command2=ID_FILE_OPEN
Command3=ID_FILE_SAVE
Command4=ID_FILE_PRINT
Command5=ID_EDIT_UNDO
Command6=ID_EDIT_CUT
Command7=ID_EDIT_COPY
Command8=ID_EDIT_PASTE
Command9=ID_EDIT_UNDO
Command10=ID_EDIT_CUT
Command11=ID_EDIT_COPY
Command12=ID_EDIT_PASTE
Command13=ID_NEXT_PANE
Command14=ID_PREV_PANE
CommandCount=14

[TB:IDR_MAINFRAME]
Type=1
Class=?
Command1=ID_FILE_NEW
Command2=ID_FILE_OPEN
Command3=ID_FILE_SAVE
Command4=ID_EDIT_CUT
Command5=ID_EDIT_COPY
Command6=ID_EDIT_PASTE
Command7=ID_FILE_PRINT
Command8=ID_APP_ABOUT
CommandCount=8

[CLS:ResultView]
Type=0
HeaderFile=ResultView.h
ImplementationFile=ResultView.cpp
BaseClass=CView
Filter=C
LastObject=ID_APP_ABOUT

[CLS:CResultView]
Type=0
HeaderFile=ResultView.h
ImplementationFile=ResultView.cpp
BaseClass=CScrollView
Filter=C

[CLS:CRetrievallView]
Type=0
HeaderFile=RetrievallView.h
ImplementationFile=RetrievallView.cpp
BaseClass=CScrollView
Filter=C
VirtualFilter=VWC
LastObject=ID_Move

[MNU:IDR_MENU1]
Type=1
Class=CRetrievallView
Command1=ID_Rotate
Command2=ID_small
Command3=ID_big
Command4=ID_Move
CommandCount=4

