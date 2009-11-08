#ifndef GLOBAL_H
#define GLOBAL_H

#ifndef	BIF_NEWDIALOGSTYLE
#define		BIF_NEWDIALOGSTYLE		0x0040
#endif

#ifndef	BIF_USENEWUI
#define		BIF_USENEWUI			0x0050
#endif

//	初始化文件夹设定用的回调函数
int CALLBACK _SHBrowseForFolderCallbackProc(HWND hwnd, UINT uMsg, LPARAM lParam, LPARAM lpData)
{	if(uMsg == BFFM_INITIALIZED)
		::SendMessage(hwnd, BFFM_SETSELECTION, TRUE, lpData);
	return	0;
}
bool g_fSelectFolderDlg(CString* lpstrFolder,CString strIniFolder,bool bAvailNewFolder)
{	bool			ret;
	char			lpszPath[MAX_PATH];
	LPMALLOC		lpMalloc;
	BROWSEINFO		sInfo;
	LPITEMIDLIST	lpidlRoot;
	LPITEMIDLIST	lpidlBrowse;
	if(lpstrFolder == NULL)		return	false;
	if(::SHGetMalloc(&lpMalloc) != NOERROR)		return	false;
	ret = false;
	if(strIniFolder != "")
	{	if(strIniFolder.Right(1) == "\\")
			strIniFolder = strIniFolder.Left(strIniFolder.GetLength() - 1);	//删除末尾的"\\"
		
	}
	::SHGetSpecialFolderLocation(NULL, CSIDL_DRIVES, &lpidlRoot);	//取得选定的文件夹名
	::ZeroMemory(&sInfo, sizeof(BROWSEINFO));
	sInfo.pidlRoot		= lpidlRoot;
	sInfo.pszDisplayName = lpszPath;
	sInfo.lpszTitle		= "选择Java源程序目录";
	sInfo.ulFlags		= BIF_RETURNONLYFSDIRS|BIF_EDITBOX ;
	if(bAvailNewFolder == true)
		sInfo.ulFlags	|= BIF_EDITBOX | BIF_NEWDIALOGSTYLE | BIF_USENEWUI;
	sInfo.lpfn			= _SHBrowseForFolderCallbackProc;
	sInfo.lParam		= (LPARAM)strIniFolder.GetBuffer(0);
	
	lpidlBrowse = ::SHBrowseForFolder(&sInfo);			//显示文件夹选择对话框
	
	if(lpidlBrowse != NULL)
	{
		if(::SHGetPathFromIDList(lpidlBrowse,lpszPath))	//取得文件夹名		
		{
			*lpstrFolder = "";
			*lpstrFolder = lpszPath;
/*			
			if(*lpstrFolder != "")
			{
				if(lpstrFolder->Right(1) != "\\")
					*lpstrFolder += "\\";				//在末尾时附加"\\"
			}*/
		}
		
		ret = true;
	}
	
	if(lpidlBrowse != NULL)
		::CoTaskMemFree(lpidlBrowse);
	if(lpidlRoot != NULL)
		::CoTaskMemFree(lpidlRoot);
	
	lpMalloc->Release();
	
	return	ret;
}

#endif