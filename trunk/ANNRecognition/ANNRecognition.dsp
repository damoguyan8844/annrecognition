# Microsoft Developer Studio Project File - Name="ANNRecognition" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102

CFG=ANNRecognition - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "ANNRecognition.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "ANNRecognition.mak" CFG="ANNRecognition - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "ANNRecognition - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "ANNRecognition - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "ANNRecognition - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ANNRECOGNITION_EXPORTS" /Yu"stdafx.h" /FD /c
# ADD CPP /nologo /MD /W3 /GX /O2 /I ".\WTL71" /I ".\TIFF" /I ".\CXImage" /I ".\MODIOCR" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "_WIN32_DCOM" /D "ANNRECOGNITION_EXPORTS" /FD /c
# SUBTRACT CPP /YX /Yc /Yu
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib comsupp.lib /nologo /dll /machine:I386 /libpath:".\WTL71" /libpath:".\BMP2TIF" /libpath:".\CXImage"
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Desc=Copy Files
PostBuild_Cmds=Copy .\\Release\ANNRecognition.dll  ..\\DigitRec	Copy .\\ANNRecognition.h  ..\\DigitRec	Copy .\\Release\ANNRecognition.lib  ..\\DigitRec	Copy .\\Release\ANNRecognition.dll  ..\\ANNOut	copy .\\Release\ANNRecognition.dll C:\Code\CMPW\Digit	copy .\\Release\ANNRecognition.dll C:\Code\CMPW\Presentation	copy .\\Release\ANNRecognition.dll C:\Code\CMPW\out
# End Special Build Tool

!ELSEIF  "$(CFG)" == "ANNRecognition - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ANNRECOGNITION_EXPORTS" /Yu"stdafx.h" /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Gm /GX /ZI /Od /I ".\WTL71" /I ".\TIFF" /I ".\CXImage" /I ".\MODIOCR" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "_WIN32_DCOM" /D "ANNRECOGNITION_EXPORTS" /FD /GZ /c
# SUBTRACT CPP /YX /Yc /Yu
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib comsupp.lib /nologo /dll /debug /machine:I386 /pdbtype:sept
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Desc=Copy Files
PostBuild_Cmds=Copy .\\Debug\ANNRecognition.dll  ..\\DigitRec	Copy .\\ANNRecognition.h  ..\\DigitRec	Copy .\\Debug\ANNRecognition.lib  ..\\DigitRec	Copy .\\Debug\ANNRecognition.dll  ..\\ANNOut
# End Special Build Tool

!ENDIF 

# Begin Target

# Name "ANNRecognition - Win32 Release"
# Name "ANNRecognition - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\ANNInterface.def
# End Source File
# Begin Source File

SOURCE=.\ANNRecognition.cpp
# End Source File
# Begin Source File

SOURCE=.\BPAPI.CPP
# End Source File
# Begin Source File

SOURCE=.\BWIMGAPI.cpp
# End Source File
# Begin Source File

SOURCE=.\DIBAPI.CPP
# End Source File
# Begin Source File

SOURCE=.\OCRAPI.CPP
# End Source File
# Begin Source File

SOURCE=.\StdAfx.cpp
# ADD CPP /Yc"stdafx.h"
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\ANNRecognition.h
# End Source File
# Begin Source File

SOURCE=.\Lock.h
# End Source File
# Begin Source File

SOURCE=.\StdAfx.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# Begin Source File

SOURCE=.\CXImage\cximage.lib
# End Source File
# Begin Source File

SOURCE=.\CXImage\jasper.lib
# End Source File
# Begin Source File

SOURCE=.\CXImage\zlib.lib
# End Source File
# Begin Source File

SOURCE=.\CXImage\libdcr.lib
# End Source File
# Begin Source File

SOURCE=.\CXImage\mng.lib
# End Source File
# Begin Source File

SOURCE=.\CXImage\png.lib
# End Source File
# Begin Source File

SOURCE=.\CXImage\Tiff.lib
# End Source File
# Begin Source File

SOURCE=.\CXImage\Jpeg.lib
# End Source File
# Begin Source File

SOURCE=.\CXImage\jbig.lib
# End Source File
# End Group
# Begin Group "WTL71"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\WTL71\atlapp.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlcrack.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlctrls.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlctrlw.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlctrlx.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlddx.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atldlgs.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlframe.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlgdi.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlmisc.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlprint.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlres.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlresce.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlscrl.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlsplit.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atltheme.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atluser.h
# End Source File
# Begin Source File

SOURCE=.\WTL71\atlwinx.h
# End Source File
# End Group
# Begin Group "CXImage"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\CXImage\xfile.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximabmp.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximacfg.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximadef.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximage.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximagif.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximaico.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximaiter.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximajas.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximajbg.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximajpg.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximamng.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximapcx.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximapng.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximaraw.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximaska.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximatga.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximath.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximatif.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximawbmp.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\ximawmf.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\xiofile.h
# End Source File
# Begin Source File

SOURCE=.\CXImage\xmemfile.h
# End Source File
# End Group
# Begin Group "TIFF"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\TIFF\t4.h
# End Source File
# Begin Source File

SOURCE=.\TIFF\tif_dir.h
# End Source File
# Begin Source File

SOURCE=.\TIFF\tif_fax3.h
# End Source File
# Begin Source File

SOURCE=.\TIFF\tif_predict.h
# End Source File
# Begin Source File

SOURCE=.\TIFF\tiff.h
# End Source File
# Begin Source File

SOURCE=.\TIFF\tiffcomp.h
# End Source File
# Begin Source File

SOURCE=.\TIFF\tiffconf.h
# End Source File
# Begin Source File

SOURCE=.\TIFF\tiffio.h
# End Source File
# Begin Source File

SOURCE=.\TIFF\tiffiop.h
# End Source File
# Begin Source File

SOURCE=.\TIFF\tiffvers.h
# End Source File
# Begin Source File

SOURCE=.\TIFF\uvcode.h
# End Source File
# End Group
# Begin Group "MODIOCR"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\MODIOCR\MODIVWCTL.cpp
# End Source File
# Begin Source File

SOURCE=.\MODIOCR\MODIVWCTL.h
# End Source File
# Begin Source File

SOURCE=.\MODIOCR\OleDispatchDriver.cpp
# End Source File
# Begin Source File

SOURCE=.\MODIOCR\OleDispatchDriver.h
# End Source File
# End Group
# Begin Source File

SOURCE=.\ReadMe.txt
# End Source File
# End Target
# End Project
