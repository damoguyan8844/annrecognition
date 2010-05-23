#include <string.h>
#include <ctype.h>
#include "framer.h"
#include "fraxl.h"
#include <tpm/tpm.h>

/* Globals *******************************************************************/

#define BUFSIZE 32*1024
#define STRSIZE 1024
#define UNSPECIFIED -1
int debug = 0;

typedef enum { IGNORE, TEXT, SYMBOL, SYMBOLS, DATA } type_enum;
char *type_strings[]={ "ignore", "text", "symbol", "symbols", "data", NULL };
typedef void SetFunc(Frame obj_frame, char *field, char *value);

typedef enum { NORMAL, SEARCH, DISPLAY } mod_enum;
char *mod_strings[]={ "", "search", "display", NULL };

typedef enum { NONE, DEFSYMBOLS, DEFLABELS, DEFTREES } def_enum;
char *def_strings[]={ "", "defsymbols", "deflabels", "deftrees", NULL };

struct spec_struct {
  type_enum type;
  char *name, *value;
} *specs;

int num_specs;
Frame db_frame, member_frame, symbol_frame, proto_frame, rs_frame;
Frame data_frame, pdata_frame;
char name_frame[STRSIZE], number_frame[STRSIZE];
int frame_number;
int replace_ann;

/* Prototypes ****************************************************************/

void terminate(void);
void process_name(char *str, FILE *data_fp);
void read_frame(FILE *fp);
void process_vars(char *buffer, Frame obj_frame);
int  process_refs(char *buffer);
void read_spec_file(char *fname);
def_enum parse_defbody(char *str);
mod_enum parse_modifier(char *str);
type_enum parse_type(char *str);
char *parse_range(char *str, int *start, int *end, int *step);
int getline(char *buffer, int bufsize, FILE *fp);

SetFunc set_text, set_sym, set_syms, set_data;
SetFunc *set_func[]={ NULL, set_text, set_sym, set_syms, set_data };

/* Functions *****************************************************************/

void main(int argc, char *argv[])
{
  FILE *index_fp, *data_fp;
  char *prefix, *db_name;
  char str[STRSIZE];

  /* Read args */
  if(!strcmp(argv[argc-1], "-debug")) {
    debug = 1;
    argc--;
  }

  if(argc < 3) {
    printf("%s [framer_file] db_name prefix\n\n", argv[0]);
    printf("Annotates the frames under #/dbs/db_name in the\n");
    printf("FRAMER file framer_file (default is \"radix\").\n");
    printf("prefix.index contains the names of the frames,\n");
    printf("prefix.spec contains the specification of each frame,\n");
    printf("prefix.data contains the annotations for each frame.\n");
    exit(0);
  }
  
  prefix = argv[argc-1];
  db_name = argv[argc-2];
  if(argc > 3) {
    root_filename = argv[1];
  }
  else {
    root_filename = "radix";
  }

  /* Init FRAMER */
  announce_file_ops = debug;
  INITIALIZE_FRAMER();
  announce_file_ops = 1;
  /*
  open_framer_image_file(root_filename);
  */

  /* Set standard frames */
  db_frame = use_annotation(use_annotation(root_frame, "dbs"), db_name);
  /* make the database demand-loaded under a seperate filename */
  set_ground(use_annotation(db_frame, "+filename"), string_to_ground(db_name));
  member_frame = use_annotation(db_frame, "members");
  symbol_frame = use_annotation(db_frame, "symbols");
  proto_frame = use_annotation(db_frame, "proto_member");
  data_frame = use_annotation(db_frame, "member_data");
  pdata_frame = use_annotation(db_frame, "proto_data");
  /* make sure arlotje is loaded so that we can have reactive sets */
  if(!probe_annotation(root_frame, "kits")) {
    load_frame_from_file("kits;arlotje");
  }
  rs_frame = use_annotation(use_annotation(use_annotation(use_annotation(
    root_frame, "kits"), "arlotje"), "prototypes"), "reactive-set");

  /* Read the spec file */
  sprintf(str, "%s.spec", prefix);
  read_spec_file(str);

  /* Open the index file */
  sprintf(str, "%s.index", prefix);
  index_fp = fopen(str, "r");

  if(index_fp) {
    /* Open the data file */
    sprintf(str, "%s.data", prefix);
    data_fp = fopen(str, "r");

    /* Loop the index file */
    printf("Working...(every 100th frame will be listed)\n");
    printf("Press Control-C at any time to abort without changing the database\n");
    frame_number = 0;
    while(fscanf(index_fp, "%s", str) != EOF) {
      process_name(str, data_fp);
    }
    fclose(index_fp);

    if(data_fp) {
      if(fgetc(data_fp) != EOF) {
	printf("Warning: data file too long\n");
      }
      fclose(data_fp);
    }
  }

  /* Save the new FRAMER database */
  backup_everything();
}

void process_name(char *str, FILE *data_fp)
{
  char *p, *q, buffer[BUFSIZE], *bufptr;
  int range_start, range_end, range_step;
  long filepos;

  /* Search for range specifiers */
  bufptr = buffer;
  for(p=str;*p;p++) {
    if(*p == '$') {
      p++;
      if(*p == '(') {
	p = parse_range(p+1, &range_start, &range_end, &range_step);
	if(*p++ != ')') {
	  printf("Missing closing parenthesis in range:\n%s\n", str);
	  terminate();
	}
	if((range_start == UNSPECIFIED) || (range_end == UNSPECIFIED)
	   || (range_start > range_end)) {
	  printf("Invalid range in index file:\n%s\n", str);
	  terminate();
	}
	if(range_step == UNSPECIFIED) range_step = 1;
	if(debug) printf("range: %d-%d-%d\n", 
			 range_start, range_end, range_step);
	/* Assign the same data to every member of the range
	   by saving and then backing up the data_fp */
	if(data_fp) filepos = ftell(data_fp);
	for(;range_start <= range_end;range_start+=range_step) {
	  *bufptr = 0;
	  sprintf(bufptr, "%d%s", range_start, p);
	  if(data_fp) fseek(data_fp, filepos, SEEK_SET);
	  process_name(buffer, data_fp);
	}
	return;
      }
      else 
	*bufptr++ = '$';
    }
    *bufptr++ = *p;
  }
  if(++frame_number % 100 == 0) {
    printf("%s\n", str);
  }
  strcpy(name_frame, str);
  sprintf(number_frame, "%d", frame_number);
  read_frame(data_fp);
}

void read_frame(FILE *fp)
{
  char buffer[BUFSIZE];
  int field, i;
  Frame obj_frame;

  if(debug) {
    if(probe_annotation(member_frame, name_frame))
      printf("\nModifying ");
    else
      printf("\nCreating ");
    printf("frame: %s\n", name_frame);
  }
  obj_frame = use_annotation(member_frame, name_frame);
  set_prototype(obj_frame, proto_frame);
  /*
  set_ground(use_annotation(obj_frame, "_index"), 
	     integer_to_ground(frame_number-1));
  */
  /* make the frame demand-loaded under a seperate filename
  set_ground(use_annotation(obj_frame, "+filename"), 
	     string_to_ground(name_frame));
  */
  for(field=0;field < num_specs;field++) {
    if(debug) printf("  field: %s\n", specs[field].name);
    /* Check of the field has an initializer */
    if(specs[field].value) {
      if(debug) printf("    using initializer\n");
      strcpy(buffer, specs[field].value);
    }
    else {
      /* Read one line from the file */
      if(!fp) {
	printf("Error: No data file\n");
	terminate();
      }
      do {
	if(!getline(buffer, BUFSIZE, fp)) {
	  printf("Data file: premature end of file\n");
	  terminate();
	}
      } while(buffer[0] == '#'); /* skip comments */
    }
    
    if(specs[field].type == IGNORE) {
      if(debug) printf("    ignored `%s'\n", buffer);
      continue;
    }
    
    /* Process any special characters */
    /*if(debug) printf("    unprocessed: `%s'\n", buffer);*/
    do {
      process_vars(buffer, obj_frame);
    } while(process_refs(buffer));
    /*if(debug) printf("    processed: `%s'\n", buffer);*/

    replace_ann = 1;
    set_func[specs[field].type](obj_frame, specs[field].name, buffer);
  }
}

void process_vars(char *buffer, Frame obj_frame)
{
  char *p, *q, *r, *oldbuffer, *var;
  Frame frame;
  Grounding value;
  int range_start, range_end, range_step;

  /* Make a local copy of the buffer */
  oldbuffer = strdup(buffer);

  /* Search for variable references in oldbuffer,
   * writing characters to buffer as we go along.
   */
  for(p=oldbuffer;*p;) {
    if(*p == '$') {
      p++;
      if(*p == '*') {
	q = name_frame;
      }
      else if(*p == '#') {
	q = number_frame;
      }
      else if(*p == '(') {
	/* Read the variable name */
	for(q=p+1;*q && (*q != ')');q++);
	if(!*q) {
	  printf("Error: unterminated variable reference in\n%s\n", oldbuffer);
	  terminate();
	}
	r = var = malloc(q-p);
	for(p++;p!=q;p++) *r++ = *p;
	*r = 0;

	/* Substitute the value under the obj_frame */
	if(!(frame = probe_annotation(obj_frame, var))) {
	  printf("Error: annotation `%s' is not defined for frame `%s'\n",
		 var, name_frame);
	  terminate();
	}
	value = frame_ground(frame);
	if(!STRINGP(value)) {
	  printf("Error: $(%s) does not refer to a string annotation for frame `%s'\n",
		 var, name_frame);
	  terminate();
	}
	q = GSTRING(value);
	if(debug) printf("    $(%s) = `%s'\n", var, q);
	free(var);
      }
      else {
	*buffer++ = *p++;
	continue;
      }

      /* Read the optional range specifier */
      p++;
      if(*p == ':') {
	p = parse_range(p+1, &range_start, &range_end, &range_step);
	if(range_start == UNSPECIFIED) range_start = 0;
	if(range_end == UNSPECIFIED) range_end = strlen(q);
	if(range_end > strlen(q)) range_end = strlen(q);
	if(range_step == UNSPECIFIED) range_step = 1;
	if(range_start > range_end) {
	  printf("Invalid range on `%s':\n%s\n", q, oldbuffer);
	  terminate();
	}
	for(q+=range_start;
	    (range_start <= range_end) && *q;
	    range_start+=range_step)
	  *buffer++ = *q++;
      }
      else {
	*buffer = 0;
	strcat(buffer, q);
	for(;*buffer;buffer++);
      }
      continue;
    }
    *buffer++ = *p++;
  }
  *buffer = 0;
  free(oldbuffer);
}

int  process_refs(char *buffer)
{
  char *p, *q, *r, *oldbuffer, *fname;
  FILE *fp;
  int found;
  int range_start, range_end, range_step, i;

  /* Make a local copy of the buffer */
  oldbuffer = strdup(buffer);

  /* Search for file references in oldbuffer,
   * writing characters to buffer as we go along.
   */
  found = 0;
  for(p=oldbuffer;*p;) {
    if(*p == '`') {
      found = 1;
      /* read the filename */
      for(q=p+1;*q && (*q != '`');q++);
      if(!*q) {
	printf("Error: unterminated file reference in\n%s\n", oldbuffer);
	terminate();
      }
      r = fname = malloc(q-p);
      for(p++;p != q;p++) *r++ = *p;
      *r = 0;
      
      /* open the file */
      if(debug) printf("    file reference: `%s'\n", fname);
      if(!(fp=fopen(fname, "r"))) {
	printf("Cannot open file reference `%s'\n", fname);
	terminate();
      }
      free(fname);

      /* line range */
      if(*++p == ':') {
	p = parse_range(p+1, &range_start, &range_end, &range_step);
	if(range_start == UNSPECIFIED) range_start = 0;
	if(range_step == UNSPECIFIED) range_step = 1;
	if(debug) printf("fileref range: (%d)-(%d)-(%d)\n", 
			 range_start, range_end, range_step);
	for(i=0;i < range_start;i++) {
	  while(fgetc(fp) != '\n')
	    if(feof(fp)) {
	      printf("File reference range exceeds length of file\n");
	      terminate();
	    }
	}
	for(;(range_end == UNSPECIFIED) || (i <= range_end);i++) {
	  /* the (char)EOF is to make iris happy */
	  while((*buffer = fgetc(fp)) != (char)EOF) {
	    buffer++;
	    if(*(buffer-1) == '\n') break;
	  }
	  if(feof(fp)) break;
	}
      }
      else {
	/* read the entire contents of the file */
	while((*buffer = fgetc(fp)) != (char)EOF) buffer++;
      }
      fclose(fp);
    }
    else {
      *buffer++ = *p++;
    }
  }
  *buffer = 0;
  free(oldbuffer);
  return found;
}

char *parse_range(char *str, int *start, int *end, int *step)
{
  char *p, *q, temp[100];
  int size;

  /* Read start */
  p = str;
  for(q=p;*q && isdigit(*q);q++);
  size = q-p;
  memcpy(temp, p, size);
  temp[size] = 0;
  if(size)
    *start = atoi(temp);
  else
    *start = UNSPECIFIED;
  
  /* Read end */
  if(*q != '-') {
    *end = *start;
    *step = 1;
    return q;
  }
  p=q+1;
  for(q=p;*q && isdigit(*q);q++);
  size = q-p;
  memcpy(temp, p, size);
  temp[size] = 0;
  if(size)
    *end = atoi(temp);
  else
    *end = UNSPECIFIED;

  /* Read step */
  if(*q != '-') {
    *step = UNSPECIFIED;
    return q;
  }
  p=q+1;
  for(q=p;*q && isdigit(*q);q++);
  size = q-p;
  memcpy(temp, p, size);
  temp[size] = 0;
  if(size)
    *step = atoi(temp);
  else
    *step = UNSPECIFIED;
  return q;
}

void set_data(Frame obj_frame, char *field, char *value)
{
  Frame ann_frame;

  if(debug) printf("    data: `%s'\n", value);
  use_annotation(pdata_frame, field);
  ann_frame = use_annotation(data_frame, frame_name(obj_frame));
  set_prototype(ann_frame, pdata_frame);
  ann_frame = use_annotation(ann_frame, field);
  set_ground(ann_frame, string_to_ground(value));
}

#if 0
void set_vector(Frame obj_frame, char *field, char *str)
{
  Frame ann_frame;
  Grounding vector;
  char *number;
  int vecsize, i;

  /* Find number of components */
  vecsize = 0;
  for(number=strtok(str," \n");*number;number=strtok(NULL," \n")) {
    number[strlen(number)] = ' ';
    vecsize++;
  }

  /* Fill in the vector */
  GVMAKE(vector, vecsize);
  if(debug) printf("    vector (%d):", vecsize);
  for(number=strtok(str," \n"),i=0;*number;number=strtok(NULL," \n"),i++) {
    if(debug) printf(" %lf", atof(number));
    GVSET(vector, i, float_to_ground(atof(number)));
  }
  if(debug) printf("\n");

  use_annotation(proto_frame, field);
  ann_frame = use_annotation(obj_frame, field);
  set_ground(ann_frame, vector);
}
#endif

void set_text(Frame obj_frame, char *field, char *text)
{
  Frame ann_frame;

  if(debug) printf("    text: `%s'\n", text);
  use_annotation(proto_frame, field);
  ann_frame = use_annotation(obj_frame, field);
  set_ground(ann_frame, string_to_ground(text));
}

void set_syms(Frame obj_frame, char *field, char *symbol_list)
{
  char *symbol;

  symbol = strtok(symbol_list, " \n");
  if(!symbol) {
    /* Force the annotation to be created even if the symbol_list is empty */
    if(debug) printf("    symbols: <empty>\n");
    set_prototype(use_annotation(obj_frame, field), rs_frame);
    return;
  }
  replace_ann = 0;
  for(;symbol;symbol=strtok(NULL, " \n")) {
    set_sym(obj_frame, field, symbol);
  }
}

Frame create_symbol(char *category, char *symbol)
{
  Frame sym_frame, frame;
  char str[100];

  if(debug) printf("    adding new symbol `%s/%s'\n", category, symbol);

  sym_frame = use_annotation(symbol_frame, category);
  /* Check if the symbol frame exists yet */
  if(frame = probe_annotation(sym_frame, symbol)) return frame;

  sym_frame = use_annotation(sym_frame, symbol);
  frame = use_annotation(sym_frame, "set");
  set_prototype(frame, rs_frame);
  /* Make the symbol set tell
   * the frame when a value is added.
   */
  sprintf(str, "(value \"%s\" home)", category);
  put_value(frame, "implies", parse_ground_from_string(str));
  
  /* Add the symbol to the prototype */
  if(!(frame = probe_annotation(proto_frame, category))) {
    frame = use_annotation(proto_frame, category);
    /* Make the annotation a reactive set */
    set_prototype(frame, rs_frame);
    /* Set up the implies relation to make the frame tell
     * the symbol set when a value is added.
     */
    put_value(frame, "implies", 
	      parse_ground_from_string("(value \"set\" home)"));
  }

  return sym_frame;
}

void set_sym(Frame obj_frame, char *field, char *symbol)
{
  Frame sym_frame, ann_frame;

  if(debug) printf("    symbol: %s\n", symbol);

  sym_frame = create_symbol(field, symbol);

  /* Now add the value to the annotation set */
  if(replace_ann) {
    retract_value(obj_frame, field, 
		  get_value(obj_frame, field));
  }
  put_value(obj_frame, field, frame_to_ground(sym_frame));
}

void read_spec_file(char *fname)
{
  FILE *fp;
  int i,j;
  char buffer[BUFSIZE], *token, *field_name, *field_value, *str;
  type_enum type;
  mod_enum modifier;
  def_enum defbody;
  Frame field_frame;
  
  /* Open spec file */
  if(!(fp=fopen(fname, "r"))) {
    printf("Cannot open spec file `%s'\n", fname);
    exit(1);
  }

  if(debug) printf("Reading spec file...\n");

  /* Read the first line (as a description) */
  getline(buffer, BUFSIZE, fp);
  set_ground(db_frame, string_to_ground(buffer));

  /* Read the specs */
  num_specs = 0;
  specs = (struct spec_struct *)malloc(0);
  for(i=0;;) {
    /* read entire line */
    if(!getline(buffer, BUFSIZE, fp)) break;

    /* is it a comment? */
    if(buffer[0] == '#') continue;

    /* read first token (field type or annotation type) */
    token = strtok(buffer, " ");
    if(!token) continue;

    modifier = defbody = 0;
    
    defbody = parse_defbody(token);
    if(defbody) {
      if(debug) printf("(defbody)%s ", token);
    }
    else {
      modifier = parse_modifier(token);
      if(modifier) {
	if(debug) printf("(mod)%s ", token);
      }
      else {
	if(debug) printf("(type)%s ", token);
	type = parse_type(token);
      }
    }

    if(defbody <= DEFSYMBOLS) {
      /* read annotation name */
      token = strtok(NULL, " ");
      if(!token) {
	printf("Spec file: missing annotation name\n");
	exit(1);
      }
      if(debug) printf("(name)%s ", token);
      field_name = token;
    }

    if(modifier) {
      str = modifier==SEARCH?"metrics":"views";
      if(debug) printf("Putting %s under #^/%s\n", field_name, str);
      field_frame = use_annotation(db_frame, str);
      set_ground(field_frame, string_to_ground(field_name));
      field_frame = use_annotation(field_frame, field_name);

      /* Read lines until "end" */
      for(;;) {
	if(!getline(buffer, BUFSIZE, fp)) {
	  printf("Spec file: unexpected EOF while searching for \"end\"\n");
	  exit(1);
	}
	if(!strcmp(buffer, "end")) break;
	/* Read annotation name */
	token = strtok(buffer, " ");
	if(!token) continue;
	/* Read value (text) */
	field_value = strtok(NULL, "");
	if(debug) printf("  %s %s\n", token, field_value);
	set_ground(use_annotation(field_frame, token), 
		   string_to_ground(field_value));
      }
      continue;
    }
    else if(defbody == DEFSYMBOLS) {
      field_name = strdup(field_name);
      if(debug) printf("Creating symbols under category `%s':\n",
		       field_name);
      /* Read lines until "end" */
      for(;;) {
	if(!getline(buffer, BUFSIZE, fp)) {
	  printf("Spec file: unexpected EOF while searching for \"end\"\n");
	  exit(1);
	}
	if(!strcmp(buffer, "end")) break;
	/* Use whole line, excluding leading whitespace */
	for(field_value = buffer; isspace(*field_value); field_value++);
	if(debug) printf("  %s\n", field_value);
	create_symbol(field_name, field_value);
      }
      free(field_name);
      continue;
    }
    else if(defbody) {
      str = defbody==DEFLABELS?"labels":"trees";
      if(debug) printf("Defining #^/labeling/%s\n", str);
      field_frame = use_annotation(db_frame, "labeling");
      field_frame = use_annotation(field_frame, str);
      /* Read lines until "end" */
      for(;;) {
	if(!getline(buffer, BUFSIZE, fp)) {
	  printf("Spec file: unexpected EOF while searching for \"end\"\n");
	  exit(1);
	}
	if(!strcmp(buffer, "end")) break;
	/* Use whole line, excluding leading whitespace */
	for(field_value = buffer; isspace(*field_value); field_value++);
	if(debug) printf("  %s\n", field_value);
	use_annotation(field_frame, field_value);
      }
      continue;
    }
    else {
      /* read optional initializer */
      token = strtok(NULL, "");
      if(token) {
	if(*token != '=') {
	  printf("Spec file: expected `=' before initializer `%s'\n", token);
	  exit(1);
	}
	field_value = token+1;
	if(debug) printf("(value)%s", field_value);
      }
      else {
	field_value = NULL;
      }
      if(debug) printf("\n");
    }

    /* Allocate memory for the new spec */
    num_specs++;
    specs = (struct spec_struct *)
      realloc(specs, sizeof(struct spec_struct)*num_specs);
    if(!specs) {
      printf("Cannot allocate %d spec structures\n", num_specs);
      exit(1);
    }
    specs[i].type = type;
    specs[i].name = strdup(field_name);
    if(field_value)
      specs[i].value = strdup(field_value);
    else
      specs[i].value = NULL;
    i++;
  }

  if(debug) printf("%d annotations per frame\n", num_specs);
  fclose(fp);
}

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

def_enum parse_defbody(char *str)
{
  int i;

  for(i=1;def_strings[i];i++) {
    if(!strcmp(def_strings[i],str)) {
      return i;
    }
  }
  return 0;
}

mod_enum parse_modifier(char *str)
{
  int i;

  for(i=1;mod_strings[i];i++) {
    if(!strcmp(mod_strings[i],str)) {
      return i;
    }
  }
  return 0;
}

type_enum parse_type(char *str)
{
  int i;

  for(i=0;type_strings[i];i++) {
    if(!strcmp(type_strings[i], str)) {
      return(i);
    }
  }
  printf("Unknown field type: %s\n", str);
  printf("Known types:");
  for(i=0;type_strings[i];i++) {
    printf(" %s", type_strings[i]);
  }
  printf("\n");
  exit(1);
}

void terminate(void)
{
  printf("Database unchanged.\n");
  exit(1);
}
