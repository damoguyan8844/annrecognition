/* Ph_MemGetAnn: Reading the annotations of a member */

#include "photobook.h"

/* Prototypes ****************************************************************/

char *Ph_MemGetAnn(Ph_Member m, char *field);
void Ph_MemSetAnn(Ph_Member m, char *field,
		  char *value, int symbol_f);
char *PhFrameValue(Frame frame);
List PhFrameResults(Frame frame);
List Ph_MemTextAnns(Ph_Member m);
List Ph_MemSymbolAnns(Ph_Member m);

/* Functions *****************************************************************/

static char *GroundValue(Grounding value)
{
  if(STRINGP(value)) return GSTRING(value);
  if(FRAMEP(value)) return frame_name(GFRAME(value));
  return NULL;
}

char *PhFrameValue(Frame frame)
{
  Grounding value;
  if(!frame) return NULL;
  value = frame_ground(frame);
  return GroundValue(value);
}

List PhFrameResults(Frame frame)
{
  Grounding value;
  List results;
  if(!frame) return NULL;
  value = frame_ground(frame);
  results = ListCreate(NULL);
  if(RESULT_SETP(value)) {
    {DO_RESULTS(gnd, value) {
      ListAddRear(results, GroundValue(gnd));
    }}
  }
  else {
    ListAddRear(results, GroundValue(value));
  }
  return results;
}

/* Returns the annotation field of m in string form.
 * If the annotation is a frame, returns its name.
 * Caller does not free the returned string.
 * Returns NULL if m does not have field or 
 * if the field is not a string or frame.
 */
char *Ph_MemGetAnn(Ph_Member m, char *field)
{
  return PhFrameValue(probe_annotation(Ph_MemFrame(m), field));
}

/* Adds a new symbolic annotation set for "category/symbol" to the database, 
 * if one doesn't exist already.
 */
static Frame PhAddSymbol(Ph_Handle phandle, char *category, char *symbol)
{
  Frame sym_frame, frame;
  char str[100];

  sym_frame = use_annotation(phandle->symbol_frame, category);
  /* Check if the symbol frame exists yet */
  if(frame = probe_annotation(sym_frame, symbol)) return frame;

  sym_frame = use_annotation(sym_frame, symbol);
  frame = use_annotation(sym_frame, "set");
  set_prototype(frame, phandle->rs_frame);
  /* Make the symbol set tell
   * the frame when a value is added.
   */
  sprintf(str, "(value \"%s\" home)", category);
  put_value(frame, "implies", parse_ground_from_string(str));
  
  /* Add the symbol to the prototype */
  if(!(frame = probe_annotation(phandle->proto_frame, category))) {
    frame = use_annotation(phandle->proto_frame, category);
    /* Make the annotation a reactive set */
    set_prototype(frame, phandle->rs_frame);
    /* Set up the implies relation to make the frame tell
     * the symbol set when a value is added.
     */
    put_value(frame, "implies", 
	      parse_ground_from_string("(value \"set\" home)"));
  }

  return sym_frame;
}

/* Adds an annotation for m's field.
 * If symbol_f is true, value is added as a symbolic annotation.
 * If symbol_f is false, adds value as a text annotation.
 */
void Ph_MemSetAnn(Ph_Member m, char *field,
		  char *value, int symbol_f)
{
  Frame f, sym_frame;

  /* enter the field in the proto_frame, if it isn't there already */
  use_annotation(m->phandle->proto_frame, field);

  /* f = the annotation frame of the member */
  f = use_annotation(Ph_MemFrame(m), field);
  if(!symbol_f) {
    set_ground(f, string_to_ground(value));
    return;
  }
  /* create a symbol */
  sym_frame = PhAddSymbol(m->phandle, field, value);
  /* erase the current symbol */
  retract_value(Ph_MemFrame(m), field, get_value(Ph_MemFrame(m), field));
  /* put in the new symbol */
  put_value(Ph_MemFrame(m), field, frame_to_ground(sym_frame));
}

List Ph_MemTextAnns(Ph_Member m)
{
  List result = ListCreate(NULL);
  {DO_ANNOTATIONS(f, Ph_MemFrame(m)) {
    if(STRINGP(frame_ground(f))) ListAddRear(result, frame_name(f));
  }}
  return result;
}

List Ph_MemSymbolAnns(Ph_Member m)
{
  List result = ListCreate(NULL);
  {DO_ANNOTATIONS(f, Ph_MemFrame(m)) {
    if(FRAMEP(frame_ground(f))) ListAddRear(result, frame_name(f));
  }}
  return result;
}
