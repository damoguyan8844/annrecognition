/* Photobook main operations: 
 *   startup/shutdown
 *   set/get database, metric, members
 *   set filter, query
 */

#include "photobook.h"
#include "class_member.h" /* used by AddMember */
#include <tpm/stream.h>
#include <assert.h>
#include <type/parse_value.h>

/* Globals *******************************************************************/

int debug = 1;
static HashTable string_table; /* Quarked strings */
static HashTable type_table; /* Quarked types */
static ParseFunc *stringParse; /* parse function used by QuarkParse */
#define MemberFree Ph_ObjFree

/* Prototypes ****************************************************************/

Ph_Handle Ph_Startup(void);
void Ph_Shutdown(Ph_Handle phandle);
Type Ph_TypeQuark(char *type_s);
char *Ph_StringQuark(char *s);

List Ph_GetDatabases(void);
int Ph_SetDatabase(Ph_Handle phandle, char *db_name);
char *Ph_GetDatabase(Ph_Handle phandle);

List Ph_GetSymbols(Ph_Handle phandle);
List Ph_GetSubSymbols(Ph_Handle phandle, char *symbol);
List Ph_GetDBTrees(Ph_Handle phandle);
List Ph_GetDBLabels(Ph_Handle phandle);

List Ph_GetDBMetrics(Ph_Handle phandle);
List Ph_GetDBViews(Ph_Handle phandle);
Ph_Object Ph_SetMetric(Ph_Handle phandle, char *name);
Ph_Object Ph_GetMetric(Ph_Handle phandle);
Ph_Object Ph_SetView(Ph_Handle phandle, char *name);
Ph_Object Ph_GetView(Ph_Handle phandle);

Ph_Object PhLoadMetric(Ph_Handle phandle, char *name);
Ph_Object PhLoadView(Ph_Handle phandle, char *name);

int Ph_SetFilter(Ph_Handle phandle, char *filter);
int Ph_SetQuery(Ph_Handle phandle, List query_list);
void Ph_Shuffle(Ph_Handle phandle);

Ph_Member *Ph_GetMembers(Ph_Handle phandle, int *length_return);
List Ph_ListMembers(Ph_Handle phandle);
Ph_Member *Ph_GetWorkingSet(Ph_Handle phandle, int *length_return);
List Ph_ListWorkingSet(Ph_Handle phandle);
Ph_Member Ph_MemberWithName(Ph_Handle phandle, char *name);
Ph_Member Ph_AddMember(Ph_Handle phandle, char *name);
Ph_Image Ph_MemImage(Ph_Member member);

int Ph_SaveWS(Ph_Handle phandle, char *file);
int Ph_LoadWS(Ph_Handle phandle, char *file);

/* Functions *****************************************************************/

/* Initializes FRAMER */
static void FramerStartup(Ph_Handle phandle)
{
  announce_file_ops = debug;
  root_filename = phandle->framer_file;
  if(debug) fprintf(stderr, "root_filename: %s\n", root_filename);
  WITH_HANDLING
    INITIALIZE_FRAMER();
  ON_EXCEPTION
    /* no framer file */
    fprintf(stderr, "The framer file `%s' was not found.\n", root_filename);
    fprintf(stderr, "Check the contents of your FRAMER_SEARCH_PATH variable.\n");
    exit(1);
  END_HANDLING
  if(phandle->framer_image_mode) {
    WITH_HANDLING
      open_framer_image_file(root_filename);
      phandle->framer_image_mode = 1;
      debugprint("Running FRAMER in image mode\n");
    ON_EXCEPTION
      phandle->framer_image_mode = 0;
      debugprint("Running FRAMER in normal mode\n");
    END_HANDLING
  }
  /* make sure arlotje is loaded so that we can have reactive sets */
  if(!probe_annotation(root_frame, "kits")) {
    load_frame_from_file("kits;arlotje");
  }
  phandle->rs_frame = 
    subframe(root_frame, "kits", "arlotje", "prototypes", "reactive-set", "");
}

/* Returns a list of the frame names under <frame>. */
static List CollectNames(Frame frame)
{
  List names;
  if(!frame) return NULL;
  names = ListCreate(NULL);
  {DO_ANNOTATIONS(subframe, frame) {
    ListAddInOrder(names, frame_name(subframe), (CmpFunc*)strcmp);
  }}
  return names;
}

/* Scans #/dbs for the names of databases 
 * and creates a list of their names.
 */
List Ph_GetDatabases(void)
{
  return CollectNames(subframe(root_frame, "dbs", ""));
}

static int QuarkParse(Type t, char *s, void **data)
{
  int status = stringParse(t, s, data);
  if(!status) return status;
  s = **(char***)data;
  **(char***)data = Ph_StringQuark(s);
  if(s != **(char***)data) free(s);
  return status;
}

/* non-phandle specific initializations */
static void PhInit(void)
{
  UnparseFunc *upf;
  /* must be static so that it hangs around */
  static ParseTable quarkParse = { "quark", { QuarkParse, NULL } };

  TypeTableCreate();
  TypeParseTpm();
  TypeDefineTpm();
  type_table = HashTableCreate(HASH_STRING, (FreeFunc*)TypeFree);

  /* get and remember the parseFunc for strings in a global */
  stringParse = TypeParseData(TypeClassGet("string"))->parseFunc;
  /* quarks use the string unparse */
  upf = TypeParseData(TypeClassGet("string"))->unparseFunc;
  quarkParse.data.unparseFunc = upf;
  /* register the quark parse/unparse functions.
   * must be done before registering the actual type.
   */
  TypeParseDefine(&quarkParse);
  /* register a quark type for quarked strings */
  TypeClassDefine("quark", sizeof(char*), NULL, sizeof(char*), NULL, 0, 0);

  MetricClasses = ListCreate(NULL);
  ViewClasses = ListCreate(NULL);
  GlobalClasses = ListCreate(NULL);
  PhClassInit();
}

/* Initializes Photobook and returns a handle.
 * Must be the first Photobook function called.
 */
Ph_Handle Ph_Startup(void)
{
  Ph_Handle phandle = (Ph_Handle)malloc(sizeof(*phandle));
  
  phandle->data_dir = getenv("PHOTOBOOK_DATA_DIR");
  if(!phandle->data_dir) phandle->data_dir = "./data";
  
  phandle->framer_file = "radix";
  FramerStartup(phandle);
  phandle->db_name = NULL;

  PhInit();

  return phandle;
}

/* Frees the storage relevant to the current database */
static void PhCloseDatabase(Ph_Handle phandle)
{
  int i;
  ListFree(phandle->metrics);
  ListFree(phandle->views);

  for(i=0;i<phandle->total_members;i++) {
    MemberFree(phandle->total_set[i]);
  }
  free(phandle->total_set);
  free(phandle->working_set);
  PhFFCacheFree();
  /* free the string table after every db change */
  HashTableFree(string_table);
  /* FRAMER weirdness requires us to delete frames in order
   * to reclaim storage.
   */
  if(!phandle->framer_image_mode) delete_annotation(phandle->db_frame);
}

/* Sets the working_set to the total_set */
static void PhUseEverything(Ph_Handle phandle)
{
  phandle->ws_members = phandle->total_members;
  if(phandle->working_set) free(phandle->working_set);
  phandle->working_set = Allocate(phandle->ws_members, Ph_Member);
  memcpy(phandle->working_set, phandle->total_set,
	 phandle->total_members * sizeof(Ph_Member));
}

/* Create a new member with name <name> and return it. 
 * Adds the member to phandle->total_set, but not working_set.
 * Returns NULL if the member has no FRAMER entry.
 */
Ph_Member Ph_AddMember(Ph_Handle phandle, char *name)
{
  Ph_Member m;
  struct MemberData *data;
  Frame frame;

  frame = probe_annotation(phandle->member_frame, name);
  if(!frame) return NULL;

  m = Ph_ObjCreate(phandle, &Member, NULL);
  data = (struct MemberData *)m->data;
  data->index = phandle->total_members;
  data->frame = frame;
  Ph_ObjName(m) = frame_name(data->frame); /* persistent string */

  /* add transient data fields */
  frame = subframe(phandle->db_frame, "member_data", name, "");
  if(frame) {
    DO_ANNOTATIONS(f, frame) {
      Type type = Ph_TypeQuark(GSTRING(frame_ground(f)));
      Ph_ObjAddField(m, frame_name(f), type, NULL, NULL, NULL, 0, 0);
    }
  }

  phandle->total_members++;
  phandle->total_set = 
    (Ph_Member*)realloc(phandle->total_set,
			phandle->total_members*sizeof(Ph_Member));
  phandle->total_set[phandle->total_members-1] = m;
  return m;
}

/* Set or change the current database to db_name.
 * Closes the current database, if any.
 * Returns an error code if there is no such database.
 */
int Ph_SetDatabase(Ph_Handle phandle, char *db_name)
{
  char str[1000];
  FileHandle fp;
  int i;

  /* close down any existing database */
  if(phandle->db_name) PhCloseDatabase(phandle);

  phandle->db_frame = 
    probe_annotation(use_annotation(root_frame, "dbs"), db_name);
  if(!phandle->db_frame) {
    WITH_HANDLING
      /* try to load it in, in case we deleted it */
      phandle->db_frame = load_frame_from_file(db_name);
    ON_EXCEPTION
      return PH_ERROR;
    END_HANDLING
  }
  phandle->member_frame = subframe(phandle->db_frame, "members", "");
  phandle->symbol_frame = subframe(phandle->db_frame, "symbols", "");
  phandle->proto_frame = subframe(phandle->db_frame, "proto_member", "");

  /* Addressing will be relative to the member_frame */
  read_root = phandle->member_frame;

  /* recreate the global string table */
  string_table = HashTableCreate(HASH_STRING, NULL);
  phandle->metrics = ListCreate((FreeFunc*)Ph_ObjFree);
  phandle->cur_metric = NULL;
  phandle->views = ListCreate((FreeFunc*)Ph_ObjFree);
  phandle->cur_view = NULL;
  /* frame names are persistent, so we'll point to that */
  phandle->db_name = frame_name(phandle->db_frame);

  /* read the index file and create total_set */
  if(debug) fprintf(stderr, "creating members\n");
  sprintf(str, "%s/%s/index", phandle->data_dir, db_name);
  fp = FileOpen(str, "r");
  phandle->total_members = 0;
  phandle->total_set = (Ph_Member*)malloc(1);
  for(;;) {
    getline(str, 1000, fp);
    if(feof(fp)) break;
    if(!Ph_AddMember(phandle, str)) {
      if(debug) fprintf(stderr, "missing member: %s\n", str);
    }
  }
  FileClose(fp);

  /* init working_set to total_set */
  phandle->working_set = NULL;
  PhUseEverything(phandle);
  return PH_OK;
}

/* Returns the current database name */
char *Ph_GetDatabase(Ph_Handle phandle)
{
  return phandle->db_name;
}

List Ph_GetSymbols(Ph_Handle phandle)
{
  return CollectNames(phandle->symbol_frame);
}

List Ph_GetSubSymbols(Ph_Handle phandle, char *symbol)
{
  return CollectNames(subframe(phandle->symbol_frame, symbol, ""));
}

List Ph_GetDBTrees(Ph_Handle phandle)
{
  return CollectNames(subframe(phandle->db_frame, "labeling", "trees", ""));
}

List Ph_GetDBLabels(Ph_Handle phandle)
{
  return CollectNames(subframe(phandle->db_frame, "labeling", "labels", ""));
}

/* Returns the names of the metrics defined for the current database */
List Ph_GetDBMetrics(Ph_Handle phandle)
{
  return CollectNames(probe_annotation(phandle->db_frame, "metrics"));
}

/* Returns the names of the views defined for the current database */
List Ph_GetDBViews(Ph_Handle phandle)
{
  return CollectNames(probe_annotation(phandle->db_frame, "views"));
}

static void FrameInit(Ph_Object obj, Frame frame)
{
  ObjField *of;
  /* Initialize the super object first */
  if(obj->super) FrameInit(obj->super, frame);

  /* loop the fields of the object so that it is
   * initialized in field order.
   */
  for(of=obj->fields;of->name;of++) {
    char *value;
    Frame f = probe_annotation(frame, of->name);
    if(!f) continue;
    value = PhFrameValue(f);
    if(!value) {
      fprintf(stderr, "Missing value for %s in %s\n", 
	      frame_name(f), Ph_ObjName(obj));
      continue;
    }
    if(Ph_ObjSetString(obj, frame_name(f), value) == PH_ERROR) {
      fprintf(stderr, "Bad value `%s' for field `%s' of %s\n",
	      value, frame_name(f), Ph_ObjName(obj));
    }
  }
}

static Ph_Object NewMV(Ph_Handle phandle, List classes, char *fr, char *name)
{
  Ph_Object obj;
  char *class_name, *value;
  Frame frame;
  ObjClass class;
    
  frame = subframe(phandle->db_frame, fr, name, "");
  if(frame) {
    class_name = PhFrameValue(probe_annotation(frame, "class"));
  }
  else class_name = NULL;
  /* quark the name */
  name = Ph_StringQuark(name);
  if(!class_name) class_name = name;

  class = PhClassFind(classes, class_name);
  if(!class) return NULL;
  obj = Ph_ObjCreate(phandle, class, name);
  if(!obj) return NULL;

  /* initialize with the frame */
  if(frame) FrameInit(obj, frame);
  return obj;
}

Ph_Object PhLoadMetric(Ph_Handle phandle, char *name)
{
  Ph_Object obj;

  if(!name) {
    /* find default metric */
    Frame frame = probe_annotation(phandle->db_frame, "metrics");
    if(!frame) return NULL;
    name = PhFrameValue(frame);
    /* is it really an annotation? */
    if(name && !probe_annotation(frame, name)) name = NULL;
    if(!name) {
      DO_ANNOTATIONS(f, frame) {
	/* choose the first metric */
	name = frame_name(f);
	break;
      }
      if(!name) return NULL;
    }
  }

  obj = PhObjFind(phandle->metrics, name);
  /* not used yet, so create the object */
  if(!obj) {
    obj = NewMV(phandle, MetricClasses, "metrics", name);
    if(!obj) {
      fprintf(stderr, "SetMetric failed on %s\n", name);
      return NULL;
    }
    ListAddFront(phandle->metrics, obj);
  }
  return obj;
}

/* Sets or changes the current search metric, and returns the metric object.
 * The database must be set. If there is no metric class named name,
 * returns NULL. Note that the operation will take place even if the database
 * does not know about the metric. If name is NULL, the default metric for
 * the database will be chosen, if any.
 */
Ph_Object Ph_SetMetric(Ph_Handle phandle, char *name)
{
  Ph_Object obj = PhLoadMetric(phandle, name);
  if(obj) phandle->cur_metric = obj;
  return obj;
}

/* Returns the current metric, as an object. */
Ph_Object Ph_GetMetric(Ph_Handle phandle)
{
  return phandle->cur_metric;
}

Ph_Object PhLoadView(Ph_Handle phandle, char *name)
{
  Ph_Object obj;

  if(!name) {
    /* find default view */
    Frame frame = probe_annotation(phandle->db_frame, "views");
    if(!frame) return NULL;
    name = PhFrameValue(frame);
    /* is it really an annotation? */
    if(name && !probe_annotation(frame, name)) name = NULL;
    if(!name) {
      /* no explicit default */
      DO_ANNOTATIONS(f, frame) {
	/* choose the first view */
	name = frame_name(f);
	break;
      }
      if(!name) return NULL;
    }
  }

  obj = PhObjFind(phandle->views, name);
  /* not used yet, so create the object */
  if(!obj) {
    obj = NewMV(phandle, ViewClasses, "views", name);
    if(!obj) {
      fprintf(stderr, "SetView failed on %s\n", name);
      return NULL;
    }
    ListAddFront(phandle->views, obj);
  }

  /* add the view as a transient field to all members */
  {
    int i;
    Type type;
    char str[100];
    sprintf(str, "view/%s", name);
    type = Ph_TypeQuark("unknown"); /* it's a Ph_Image */
    for(i=0;i<phandle->total_members;i++) {
      /* don't cache, can load */
      Ph_ObjAddField(phandle->total_set[i], str, type,
		     obj, "image", NULL, 1, 0);
    }
  }
  return obj;
}

/* Sets or changes the current view mode, and returns the view object.
 * The database must be set. If there is no view class named name,
 * returns NULL. The operation will take place even if the database does not
 * know about the view. If name is NULL, the default view for the database
 * will be chosen, if any.
 */
Ph_Object Ph_SetView(Ph_Handle phandle, char *name)
{
  Ph_Object obj = PhLoadView(phandle, name);
  if(obj) phandle->cur_view = obj;
  return obj;
}

/* Returns the current metric, as an object. */
Ph_Object Ph_GetView(Ph_Handle phandle)
{
  return phandle->cur_view;
}

/* Returns the member of the total_set whose name is name.
 * Returns NULL if there is no such member.
 */
Ph_Member Ph_MemberWithName(Ph_Handle phandle, char *name)
{
  int i;
  for(i=0;i<phandle->total_members;i++)
    if(!strcmp(Ph_MemName(phandle->total_set[i]), 
	       name)) return phandle->total_set[i];
  return NULL;
}

/* Filters the working_set according to the commands in filter.
 * Returns an error code and sets phandle->error_string on parse error.
 * Filter of NULL or "" indicates no filter, i.e. every member is placed
 * in the working set.
 */
int Ph_SetFilter(Ph_Handle phandle, char *filter)
{
  int index;
  Grounding result_set;
  char *parsed;

  if(!filter) filter = "";
  parsed = ParseFilter(phandle, filter);
  if(!parsed) return PH_ERROR;

  if(!parsed[0]) {
    free(parsed);
    /* empty string means no filter */
    PhUseEverything(phandle);
    return PH_OK;
  }
  if(phandle->working_set) free(phandle->working_set);
  if(debug) fprintf(stderr, "parsed: `%s'\n", parsed);
  result_set = EVAL(parsed);
  free(parsed);
  
  /* convert the result_set into a working_set */
  /* FRAMER wierdness: result set of one element is sometimes returned
   * as a frame.
   */
  if(FRAMEP(result_set)) {
    phandle->working_set = Allocate(1, Ph_Member);
    index = 0;
    if(phandle->working_set[0] = 
       Ph_MemberWithName(phandle, frame_name(GFRAME(result_set)))) index++;
    phandle->ws_members = index;
  }
  else if(ND_GROUND_P(result_set)) {
    phandle->working_set = Allocate(ND_SIZE(result_set), Ph_Member);
    index = 0;
    {DO_RESULTS(member, result_set) {
      if(phandle->working_set[index] = 
	 Ph_MemberWithName(phandle, frame_name(GFRAME(member))))
	index++;
    }}
    phandle->ws_members = index;
  }
  else {
    phandle->working_set = Allocate(1, Ph_Member);
    phandle->ws_members = 0;
  }
  return PH_OK;
}

/* used by SortWS */
static int CompareDist(Ph_Member *a, Ph_Member *b)
{
  return (Ph_MemDistance(*a) > Ph_MemDistance(*b)) ? 1 : -1;
}

/* Reorders the working_set by ascending member distance */
static void SortWS(Ph_Handle phandle)
{
  qsort((void*)phandle->working_set, phandle->ws_members,
	sizeof(Ph_Member), (CmpFunc*)CompareDist);
}

/* Performs a similarity query based on the current metric.
 * Reorders the working_set so that most similar is first.
 * The query_list is a list of the names of the members which are
 * the basis of the query (i.e. the examples).
 * Currently, only the first element of the list is used.
 * If query_list is NULL or empty, does nothing.
 * If there is no active metric, does nothing.
 */
int Ph_SetQuery(Ph_Handle phandle, List query_list)
{
  PhDistFunc *distFunc;
  Ph_Member query;

  if(!query_list || ListEmpty(query_list)) return PH_OK;
  if(!phandle->cur_metric) return PH_OK;
  /* Compute member distances */
  distFunc = (PhDistFunc*)PhObjFunc(phandle->cur_metric, "distance");
  assert(distFunc);
  query = Ph_MemberWithName(phandle, ListFront(query_list)->data);
  if(!query) return PH_ERROR;
  distFunc(phandle->cur_metric, query,
	   phandle->working_set, phandle->ws_members);

  SortWS(phandle);
  return PH_OK;
}

/* Shuffle the order of the working set by assigning random distances
 * and sorting.
 */
void Ph_Shuffle(Ph_Handle phandle)
{
  int i;
  for(i=0;i<phandle->ws_members;i++) {
    Ph_MemDistance(phandle->working_set[i]) = RandReal();
  }
  SortWS(phandle);
}

/* Returns all of the database members, i.e. total_set, in db order */
Ph_Member *Ph_GetMembers(Ph_Handle phandle, int *length_return)
{
  *length_return = phandle->total_members;
  return phandle->total_set;
}

/* Returns all of the database members, i.e. total_set, in db order */
List Ph_ListMembers(Ph_Handle phandle)
{
  return ListFromPtrArray(phandle->total_set, phandle->total_members);
}

/* Returns the working set in its current order */
Ph_Member *Ph_GetWorkingSet(Ph_Handle phandle, int *length_return)
{
  *length_return = phandle->ws_members;
  return phandle->working_set;
}

/* Returns the working set in its current order */
List Ph_ListWorkingSet(Ph_Handle phandle)
{
  return ListFromPtrArray(phandle->working_set, phandle->ws_members);
}

/* Frees the storage used by an instance of Photobook.
 * phandle is freed and should no longer be used.
 */
void Ph_Shutdown(Ph_Handle phandle)
{
  PhCloseDatabase(phandle);

  HashTableFree(type_table);
  ListFree(MetricClasses);
  ListFree(ViewClasses);
  ListFree(GlobalClasses);
  TypeTableFree();
  TypeParseFree();

  free(phandle);
  if(debug) {
    MEM_BLOCKS();
    MEM_STATUS();
  }
}

/* Returns a quarked version of s, i.e. returns identical pointer values
 * for all equivalent strings.
 * (This naming convention is in keeping with the quantum mechanical theory 
 * that all equivalent quarks are identical.)
 */
char *Ph_StringQuark(char *s)
{
  int found;
  HashEntry *e = HashTableAddEntry(string_table, s, &found);
  return HashEntryKey(e);
}

/* Returns the quarked version of TypeParse(type_s) */
Type Ph_TypeQuark(char *type_s)
{
  int found;
  HashEntry *e;
  Type type;
  e = HashTableFindEntry(type_table, type_s);
  if(e) return HashEntryValue(e);
  type = TypeParse(type_s);
  if(TypeHasUnknowns(type)) return type;
  e = HashTableAddEntry(type_table, type_s, &found);
  return HashEntryValue(e) = type;
}

Ph_Image Ph_MemImage(Ph_Member member)
{
  PhImageFunc *func = (PhImageFunc*)PhObjFunc(member->phandle->cur_view, 
					      "image");
  assert(func);
  return func(member->phandle->cur_view, member);
}

/* Saves the working set to file.
 * Returns an error code if the file could not be opened.
 */
int Ph_SaveWS(Ph_Handle phandle, char *file)
{
  int i;
  FILE *fp;
  fp = fopen(file, "w");
  if(!fp) return PH_ERROR;

  for(i=0;i<phandle->ws_members;i++) {
    fprintf(fp, "%s %g\n", 
	    Ph_MemName(phandle->working_set[i]),
	    Ph_MemDistance(phandle->working_set[i]));
  }
  fclose(fp);
}

/* Reads the working set from file.
 * Returns an error code if the file could not be opened.
 */
int Ph_LoadWS(Ph_Handle phandle, char *file)
{
#define LINE_MAX 100
  char line[LINE_MAX], *name, *dist;
  FILE *fp;
  List members;
  Ph_Member member;

  fp = fopen(file, "r");
  if(!fp) return PH_ERROR;
  
  members = ListCreate(NULL);
  for(;;) {
    if(!getline(line, LINE_MAX, fp)) break;
    /* read the member name */
    name = strtok(line, " ");
    if(!name) continue;
    member = Ph_MemberWithName(phandle, name);
    if(!member) {
      fprintf(stderr, "Bad member name `%s'\n", name);
      continue;
    }
    /* read the distance value */
    dist = strtok(NULL, " ");
    if(dist) Ph_MemDistance(member) = atof(dist);
    else Ph_MemDistance(member) = NOTADISTANCE;
    ListAddRear(members, member);
  }
  fclose(fp);
  if(phandle->working_set) free(phandle->working_set);
  phandle->working_set = ListToPtrArray(members, &phandle->ws_members);
  ListFree(members);
  return PH_OK;
}

