/* Definitions for using Streams and DiskArrays */
#ifndef STREAM_H_INCLUDED
#define STREAM_H_INCLUDED

#include <tpm/tpm.h>

typedef FILE *FileHandle;
FileHandle FileOpen(char *fname, char *mode);
void FileClose(FileHandle h);
FileHandle fdup(FileHandle h);
int getline(char *buffer, int bufsize, FILE *fp);

/* A Stream is an abstraction for anything that supports the open,
 * read, close protocol. All Streams support StreamNext, which reads
 * the next piece of data from the Stream. Some Streams also support
 * StreamRead, which provides random access. A DiskArray is a type of Stream.
 */
typedef int StreamReadFunc(void *source, void *data, int index, void *dest);
typedef int StreamCloseFunc(void *source, void *data);

typedef struct {
  void *source;
  void *data;
  StreamReadFunc *readFunc;
  StreamCloseFunc *closeFunc;
} Stream;

#define StreamRead(s, i, d) (s).readFunc((s).source, (s).data, i, d)
#define StreamNext(s, d) (s).readFunc((s).source, (s).data, -1, d)
#define StreamClose(s) (s).closeFunc((s).source, (s).data)

void DiskArrayOpen(Stream *s, FileHandle h, int num_items, int item_size);

#endif
