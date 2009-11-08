# Microsoft Developer Studio Project File - Name="CCMD_OCR" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Application" 0x0101

CFG=CCMD_OCR - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "CCMD_OCR.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "CCMD_OCR.mak" CFG="CCMD_OCR - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "CCMD_OCR - Win32 Release" (based on "Win32 (x86) Application")
!MESSAGE "CCMD_OCR - Win32 Debug" (based on "Win32 (x86) Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "CCMD_OCR - Win32 Release"

# PROP BASE Use_MFC 6
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 6
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MD /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_AFXDLL" /Yu"stdafx.h" /FD /c
# ADD CPP /nologo /MD /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_AFXDLL" /Yu"stdafx.h" /FD /c
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x804 /d "NDEBUG" /d "_AFXDLL"
# ADD RSC /l 0x804 /d "NDEBUG" /d "_AFXDLL"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /subsystem:windows /machine:I386
# ADD LINK32 /nologo /subsystem:windows /machine:I386 /out:"..\\..\\ANNOut\CCMD_OCR.exe"

!ELSEIF  "$(CFG)" == "CCMD_OCR - Win32 Debug"

# PROP BASE Use_MFC 6
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 6
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MDd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_AFXDLL" /Yu"stdafx.h" /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Gm /GX /ZI /Od /I "D:\OFFICE OCR" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_AFXDLL" /Yu"stdafx.h" /FD /GZ /c
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x804 /d "_DEBUG" /d "_AFXDLL"
# ADD RSC /l 0x804 /d "_DEBUG" /d "_AFXDLL"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 /nologo /subsystem:windows /debug /machine:I386 /pdbtype:sept
# ADD LINK32 /nologo /subsystem:windows /debug /machine:I386 /out:"..\\..\\ANNOut\CCMD_OCR.exe" /pdbtype:sept /libpath:"D:\OFFICE OCR"

!ENDIF 

# Begin Target

# Name "CCMD_OCR - Win32 Release"
# Name "CCMD_OCR - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\CCMD_OCR.cpp
# End Source File
# Begin Source File

SOURCE=.\CCMD_OCR.rc
# End Source File
# Begin Source File

SOURCE=.\CCMD_OCRDoc.cpp
# End Source File
# Begin Source File

SOURCE=.\CCMD_OCRView.cpp
# End Source File
# Begin Source File

SOURCE=.\MainFrm.cpp
# End Source File
# Begin Source File

SOURCE=.\mdivwctl.cpp
# End Source File
# Begin Source File

SOURCE=.\StdAfx.cpp
# ADD CPP /Yc"stdafx.h"
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\CCMD_OCR.h
# End Source File
# Begin Source File

SOURCE=.\CCMD_OCRDoc.h
# End Source File
# Begin Source File

SOURCE=.\CCMD_OCRView.h
# End Source File
# Begin Source File

SOURCE=.\MainFrm.h
# End Source File
# Begin Source File

SOURCE=.\mdivwctl.h
# End Source File
# Begin Source File

SOURCE=.\Resource.h
# End Source File
# Begin Source File

SOURCE=.\StdAfx.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# Begin Source File

SOURCE=.\res\CCMD_OCR.ico
# End Source File
# Begin Source File

SOURCE=.\res\CCMD_OCR.rc2
# End Source File
# Begin Source File

SOURCE=.\res\CCMD_OCRDoc.ico
# End Source File
# Begin Source File

SOURCE=.\res\Toolbar.bmp
# End Source File
# End Group
# End Target
# End Project
# Section CCMD_OCR : {3A1E1B7A-C041-4DDC-BF3B-042A0B95B82B}
# 	2:5:Class:CMiSelectRects
# 	2:10:HeaderFile:miselectrects.h
# 	2:8:ImplFile:miselectrects.cpp
# End Section
# Section CCMD_OCR : {F6379198-3B20-461A-B3A9-191945752557}
# 	2:5:Class:CMiSelectableImage
# 	2:10:HeaderFile:miselectableimage.h
# 	2:8:ImplFile:miselectableimage.cpp
# End Section
# Section CCMD_OCR : {01C4414A-D123-4BC7-A1FA-64E376C01655}
# 	2:5:Class:CMiSelectableItem
# 	2:10:HeaderFile:miselectableitem.h
# 	2:8:ImplFile:miselectableitem.cpp
# End Section
# Section CCMD_OCR : {EF347A62-BA21-42E4-94A0-1C0A6D7FDFE7}
# 	2:21:DefaultSinkHeaderFile:midocview.h
# 	2:16:DefaultSinkClass:CMiDocView
# End Section
# Section CCMD_OCR : {7BF80981-BF32-101A-8BBB-00AA00300CAB}
# 	2:5:Class:CPicture
# 	2:10:HeaderFile:picture.h
# 	2:8:ImplFile:picture.cpp
# End Section
# Section CCMD_OCR : {F958524A-8422-4B07-B69E-199F2421ED89}
# 	2:5:Class:CMiDocView
# 	2:10:HeaderFile:midocview.h
# 	2:8:ImplFile:midocview.cpp
# End Section
