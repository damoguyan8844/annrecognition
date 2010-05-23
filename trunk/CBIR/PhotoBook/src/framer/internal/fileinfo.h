#define _INCLUDE_POSIX_SOURCE
#define _POSIX_SOURCE
#if FOR_UNIX
#include <sys/types.h>
#include <time.h>
#include <sys/stat.h>
#if defined(__NeXT__)
#include <bsd/sys/fcntl.h>
#endif
#if defined(_AIX)
#include <sys/access.h>
#endif
#if !defined(W_OK)
#include <unistd.h>
#endif
extern struct stat tmp_buf;
#define DIRECTORY_SEPARATOR '/'
#define FILE_EXISTS_P(file) (stat(file,&tmp_buf) != -1)
#define FILE_READABLE_P(file) (access(file,R_OK) != -1)
#define DIRECTORY_WRITABLE_P(file) (directory_writable_p(file))
#define FILE_WRITABLE_P(file) ((FILE_EXISTS_P(file)) ? (access(file,W_OK) != -1) : (DIRECTORY_WRITABLE_P(file)))

#endif /* FOR_UNIX */

#if FOR_MAC
#define DIRECTORY_SEPARATOR ':'
#define FILE_EXISTS_P(file) (True)
#define FILE_READABLE_P(file) (True)
#define FILE_WRITABLE_P(file) (True)
#endif

#if FOR_MSDOS
#define DIRECTORY_SEPARATOR '\\'
#include <sys/types.h>
#include <sys/stat.h>
#include <io.h>
#define W_OK 02
#define R_OK 04
extern struct stat tmp_buf;
#define FILE_EXISTS_P(file) (stat(file,&tmp_buf) != -1)
#define FILE_READABLE_P(file) (access(file,R_OK) != -1)
#define DIRECTORY_WRITABLE_P(file) (directory_writable_p(file))
#define FILE_WRITABLE_P(file) \
  ((FILE_EXISTS_P(file)) ? (access(file,W_OK) != -1) : DIRECTORY_WRITABLE_P(file))

#endif /* FOR_MSDOS */

#ifndef FILE_EXISTS_P
#define FILE_EXISTS_P(file) (True)
#define FILE_READABLE_P(file) (True)
#define FILE_WRITABLE_P(file) (True)
#endif

boolean read_only_p(char *file);
boolean directory_writable_p(char *file);
FILE *open_safe_stream(char *filename);
void close_safe_stream(FILE *stream);
void abort_safe_stream(FILE *stream);
void set_search_path(char *pathspec,char **path_set);
char *search_path_for_name(char *name,char **path_set);
extern char *backup_search_path[];
 



