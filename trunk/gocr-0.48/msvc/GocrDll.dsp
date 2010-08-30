# Microsoft Developer Studio Project File - Name="GocrDll" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102

CFG=GocrDll - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "GocrDll.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "GocrDll.mak" CFG="GocrDll - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "GocrDll - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "GocrDll - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "GocrDll - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "GOCRDLL_EXPORTS" /Yu"stdafx.h" /FD /c
# ADD CPP /nologo /MT /W3 /GX /O2 /I "..\include" /I "..\src" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "GOCRDLL_EXPORTS" /YX /FD /c
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386

!ELSEIF  "$(CFG)" == "GocrDll - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "GocrDll___Win32_Debug"
# PROP BASE Intermediate_Dir "GocrDll___Win32_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "GocrDll___Win32_Debug"
# PROP Intermediate_Dir "GocrDll___Win32_Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "GOCRDLL_EXPORTS" /Yu"stdafx.h" /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /I "..\include" /I "..\src" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "GOCRDLL_EXPORTS" /YX /FD /GZ /c
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /pdbtype:sept

!ENDIF 

# Begin Target

# Name "GocrDll - Win32 Release"
# Name "GocrDll - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\src\barcode.c
# End Source File
# Begin Source File

SOURCE=..\src\box.c
# End Source File
# Begin Source File

SOURCE=..\src\database.c
# End Source File
# Begin Source File

SOURCE=..\src\detect.c
# End Source File
# Begin Source File

SOURCE=..\src\gocr.c
# End Source File
# Begin Source File

SOURCE=..\src\gocrdll.c
# End Source File
# Begin Source File

SOURCE=..\src\jconv.c
# End Source File
# Begin Source File

SOURCE=..\src\job.c
# End Source File
# Begin Source File

SOURCE=..\src\lines.c
# End Source File
# Begin Source File

SOURCE=..\src\list.c
# End Source File
# Begin Source File

SOURCE=..\src\ocr0.c
# End Source File
# Begin Source File

SOURCE=..\src\ocr0n.c
# End Source File
# Begin Source File

SOURCE=..\src\ocr1.c
# End Source File
# Begin Source File

SOURCE=..\src\otsu.c
# End Source File
# Begin Source File

SOURCE=..\src\output.c
# End Source File
# Begin Source File

SOURCE=..\src\pcx.c
# End Source File
# Begin Source File

SOURCE=..\src\pgm2asc.c
# End Source File
# Begin Source File

SOURCE=..\src\pixel.c
# End Source File
# Begin Source File

SOURCE=..\src\pnm.c
# End Source File
# Begin Source File

SOURCE=..\src\progress.c
# End Source File
# Begin Source File

SOURCE=..\src\remove.c
# End Source File
# Begin Source File

SOURCE=..\src\stdafx.cpp
# End Source File
# Begin Source File

SOURCE=..\src\tga.c
# End Source File
# Begin Source File

SOURCE=..\src\unicode.c
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=..\src\amiga.h
# End Source File
# Begin Source File

SOURCE=..\src\barcode.h
# End Source File
# Begin Source File

SOURCE=..\include\config.h
# End Source File
# Begin Source File

SOURCE=..\src\gocr.h
# End Source File
# Begin Source File

SOURCE=.\GocrDll.h
# End Source File
# Begin Source File

SOURCE=..\src\list.h
# End Source File
# Begin Source File

SOURCE=..\src\ocr0.h
# End Source File
# Begin Source File

SOURCE=..\src\ocr1.h
# End Source File
# Begin Source File

SOURCE=..\src\otsu.h
# End Source File
# Begin Source File

SOURCE=..\src\output.h
# End Source File
# Begin Source File

SOURCE=..\src\pcx.h
# End Source File
# Begin Source File

SOURCE=..\src\pgm2asc.h
# End Source File
# Begin Source File

SOURCE=..\src\pnm.h
# End Source File
# Begin Source File

SOURCE=..\src\progress.h
# End Source File
# Begin Source File

SOURCE=..\src\stdafx.h
# End Source File
# Begin Source File

SOURCE=..\src\tga.h
# End Source File
# Begin Source File

SOURCE=..\src\unicode.h
# End Source File
# Begin Source File

SOURCE=..\include\version.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# Begin Source File

SOURCE=.\ReadMe.txt
# End Source File
# End Target
# End Project
