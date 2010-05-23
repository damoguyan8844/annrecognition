#include "framer.h"

#define INTERNED_FRAME_NAME(x) \
   if (x != _interned) x=_interned=intern_frame_name(x);
#define BINARY_STRING_SEARCH(result,string,array,accessor,left,right) \
  {int comparison=1,halfway;                                          \
   halfway=(left)+((right)-(left))/2;                                 \
   while (((right) >= (left)) && (comparison != 0))                   \
     {comparison=string_compare(string,((array)[halfway])  accessor);\
      if (comparison < 0) (right) = halfway-1;                        \
      if (comparison > 0) (left)  = halfway+1;                        \
      halfway=(left)+((right)-(left))/2;}                             \
   if (comparison == 0) result=array[halfway]; else result=NULL;}

#define BINARY_STRING_SEARCH_DIRECT(result,string,array,left,right)   \
  {int comparison=1,halfway;                                          \
   halfway=(left)+((right)-(left))/2;                                 \
   while (((right) >= (left)) && (comparison != 0))                   \
     {comparison=string_compare(string,((array)[halfway]));          \
      if (comparison < 0) (right) = halfway-1;                        \
      if (comparison > 0) (left)  = halfway+1;                        \
      halfway=(left)+((right)-(left))/2;}                             \
   if (comparison == 0) result=array[halfway]; else result=NULL;}

char *fread_frame_name(FILE *stream,char *into_string);
Frame_Appendix *add_appendix(Frame f);
Frame_Array *make_frame_array(int size);
void fprint_frame_name(char *string,FILE *stream);
void add_spinoff(Frame spinoff,Frame frame);
void restore_frame_from_file(Frame frame);
int grab_search_bit(void);
void release_search_bit(int bit);
int pprint_elt(generic_stream *stream,Grounding elt,int left,int right,int xpos);
Frame raw_local_probe_annotation(Frame f,char *name);

extern int root_print_depth;
/* The array used for zip coding */
extern Frame_Array *zip_codes;
/* Whether to trap unknown frames (bound to flase when reading from an archive) */
extern boolean trap_unknown_frames, active_image_p;

struct CHAR_BUCKET 
{char key; 
 union {struct INTERVAL {struct CHAR_BUCKET *head, *tail;} branches;
	char *string; int code;} data;
};

struct CHAR_BUCKET *find_bucket(char *string,struct CHAR_BUCKET *beneath);

extern char **frame_name_table;
extern int number_of_frame_names, space_in_frame_name_table;
