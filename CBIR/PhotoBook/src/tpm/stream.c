/* Implementation of some useful stream operations.
 * In particular, implements DiskArrays, which are Streams
 * over packed arrays (e.g. a movie produced from concatenated image files).
 */

#include <tpm/tpm.h>
#include <tpm/stream.h>

/* Prototypes ****************************************************************/

FileHandle FileOpen(char *fname, char *mode);
void FileClose(FileHandle h);
FileHandle fdup(FileHandle h);
int getline(char *buffer, int bufsize, FILE *fp);

void DiskArrayOpen(Stream *s, FileHandle h, int num_items, int item_size);

/* Functions *****************************************************************/

FileHandle FileOpen(char *fname, char *mode)
{
  FileHandle h;

  h = fopen(fname, mode);
  if(!h) {
    fprintf(stderr, "Cannot open file `%s': ", fname);
    perror("");
    fflush(stderr);
    exit(1);
  }
  return h;
}

void FileClose(FileHandle h)
{
  if(fclose(h) == EOF) {
    perror("Cannot close file");
    exit(1);
  }
}

/* Duplicate a file handle. Reading from the second handle before
 * closing the first can produce strange results.
 */
FileHandle fdup(FileHandle h)
{
  return fdopen(dup(fileno(h)), "r");
}

/* A safe way to read lines from a file.
 * Reads up to bufsize characters, terminating at newlines and EOF.
 * Newlines do not show up in the buffer.
 * Returns 0 on EOF.
 */
int getline(char *buffer, int bufsize, FILE *fp)
{
  int i;

  fgets(buffer, bufsize, fp);
  if(feof(fp)) return 0;
  /* Remove trailing whitespace */
  i = strlen(buffer)-1;
  for(i = strlen(buffer)-1; isspace(buffer[i]) && (i >= 0); i--);
  buffer[i+1]=0;
  return 1;
}

typedef struct {
  int current_pos, offset;
  int item_size, num_items;
} *DiskArrayData;

/* Reads data from the specified position in the disk array,
 * avoiding fseeks whenever it can.
 */
static int DiskArrayRead(FileHandle h, DiskArrayData data, 
			 int index, void *dest)
{
  int pos;

  if(index == -1) {
    index = (data->current_pos - data->offset) / data->item_size;
  }
  if(index >= data->num_items) return 0;
  pos = index * data->item_size + data->offset;
  if(data->current_pos != pos) {
    if(fseek(h, pos - data->current_pos, SEEK_CUR) == -1) {
      perror("DiskArrayRead");
      return 0;
    }
  }
  if(!fread(dest, data->item_size, 1, h)) return 0;
  data->current_pos = pos + data->item_size;
  return 1;
}

static int DiskArrayClose(FileHandle h, DiskArrayData data)
{
  int status;

  status = fclose(h);
  free(data);
  if(status == EOF) {
    perror("DiskArrayClose");
    return 0;
  }
  return 1;
}

/* Make s into a disk array of num_items items. 
 * Each item is exactly item_size bytes, with no padding in between.
 * Use StreamRead or StreamNext to read out of the disk array.
 */
void DiskArrayOpen(Stream *s, FileHandle h, int num_items, int item_size)
{
  DiskArrayData data = (DiskArrayData)malloc(sizeof(*data));

  s->source = h;
  data->current_pos = data->offset = ftell(h);
  data->item_size = item_size;
  data->num_items = num_items;
  s->data = data;
  s->readFunc = (StreamReadFunc*)DiskArrayRead;
  s->closeFunc = (StreamCloseFunc*)DiskArrayClose;
}


