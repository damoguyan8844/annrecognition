#include "type.h"

/* Globals *******************************************************************/

#define SkipSpace(p) for(;*p && isspace(*p);(p)++)
#define SkipNumber(p) for(;*p && isdigit(*p);(p)++)
#define SkipWord(p) for(;*p && !isspace(*p) && !strchr("[](),", *p); (p)++)

#define AddIntParam(t, type) ListAddRear(t->i_params.value.l, type)
#define AddTypeParam(t, type) ListAddRear(t->t_params.value.l, type)
#define TypeTNInt(t) ListSize((t)->i_params.value.l)
#define TypeTNType(t) ListSize((t)->t_params.value.l)

/* Prototypes ****************************************************************/

Type TypeParse(char *s);
void TypeUnparse(FILE *fp, Type t);

/* Private */
static Type TypeParsePart(char **s);
static void TypeTFree(TypeT t);

/* Functions *****************************************************************/

/* Returns the Type corresponding to a string.
 * Syntax is 
 *   class[integer params](type params)
 * e.g.
 *   int
 *   list(int)
 *   array[2](int)
 *   list(array[2](int))
 * Parentheses may be omitted from the last type param:
 *   list int
 *   array[2] int
 *   list array[2] int
 *   struct(int) float
 * A number x by itself is shorthand for array[x], e.g.
 *   2 int
 *   list 2 int
 * Unknowns are specified by preceding a name with a question mark:
 *   array[?x] int    (array of unknown length)
 *   list ?y          (list of unknown subtype)
 *   ?z               (unknown type)
 */
Type TypeParse(char *s)
{
  char *p = s;
  Type t;

  SkipSpace(p);
  t = TypeParsePart(&p);
  if(!t) {
    TypeError("empty type string");
  }
  if(*p) {
    fprintf(stderr, "TypeParse warning: extra characters after type\n");
  }
  return t;
}

static int IsNumber(char *s)
{
  for(;*s;s++) if(!isdigit(*s)) return 0;
  return 1;
}

static List TypeListCreate(void)
{
  return ListCreate((FreeFunc*)TypeFree);
}

/* requires: *s has no leading whitespace */
static Type TypeParsePart(char **s)
{
  Type type;
  TypeT t;
  char *word,*p,*q;
  int i;

  /* read the first word */
  p = *s;
  SkipWord(*s);
  i = *s - p;
  word = Allocate(i + 1, char);
  memcpy(word, p, i);
  word[i] = '\0';
  SkipSpace(*s);

  if(!*word || (*word == ')')) { free(word); return NULL; }

  /* unknown? */
  if(*word == '?') {
    type = TypeCreate(TYPE_UNKNOWN, strdup(word+1));
    free(word);
    return type;
  }

  t = Allocate(1, struct TypeTStruct);
  t->i_params.tag = TYPE_LIST;
  t->i_params.value.l = TypeListCreate();
  t->t_params.tag = TYPE_LIST;
  t->t_params.value.l = TypeListCreate();

  /* convert shorthand; "100" -> "array[100]" */
  if(IsNumber(word)) {
    t->class = TypeClassGet("array");
    sscanf(word, "%d", &i);
    AddIntParam(t, TypeCreate(TYPE_INT, i));
  }
  else {
    t->class = TypeClassGet(word);
  }
  free(word);

  /* any numerical parameters? */
  if(**s == '[') {
    for(;;) {
      (*s)++;
      SkipSpace(*s);
      if(**s == '?') {
	p = ++(*s);
	SkipWord(*s);
	/* copy the word in [p .. *s] to q */
	i = *s - p;
	q = Allocate(i+1,char);
	memcpy(q,p,i);
	q[i] = '\0';
	AddIntParam(t, TypeCreate(TYPE_UNKNOWN, q));
      }
      else if(isdigit(**s)) {
	sscanf(*s, "%d", &i);
	AddIntParam(t, TypeCreate(TYPE_INT, i));
	SkipNumber(*s);
      }
      else {
	TypeError("Bad numerical parameter for type `%s'", t->class->name);
      }
      SkipSpace(*s);
      if(**s == ']') break;
      if(**s != ',') {
	TypeError("Bad char `%c'; comma or ] expected in numerical list for `%s'", **s, t->class->name);
      }
    }
    (*s)++;
    SkipSpace(*s);
  }
  if(t->class->num_i_params != VARIABLE_PARAMS) {
    if(t->class->num_i_params < TypeTNInt(t)) {
      TypeError("Too many int parameters (%d) for type `%s'; expected %d", 
		TypeTNInt(t), t->class->name, t->class->num_i_params);
    }
    if(t->class->num_i_params > TypeTNInt(t)) {
      TypeError("Too few int parameters (%d) for type `%s'; expected %d", 
		TypeTNInt(t), t->class->name, t->class->num_i_params);
    }
  }

  /* any type parameters? */
  if(**s == '(') {
    for(;;) {
      (*s)++;
      SkipSpace(*s);
      type = TypeParsePart(s);
      if(type == NULL) {
	TypeError("Bad type parameter for `%s'", t->class->name);
      }
      AddTypeParam(t, type);
      SkipSpace(*s);
      if(**s == ')') break;
      if(**s != ',') {
	TypeError("Comma or ) expected in type list for `%s'", t->class->name);
      }
    }
    (*s)++;
    SkipSpace(*s);
  }

  /* shorthand; read extra params after the ')' */
  type = TypeParsePart(s);
  if(type) AddTypeParam(t, type);

  /* check number of parameters */
  if(t->class->num_t_params != VARIABLE_PARAMS) {
    if(t->class->num_t_params < TypeTNType(t)) {
      TypeError("Too many type parameters (%d) for type `%s'; expected %d", 
		TypeTNType(t), t->class->name, t->class->num_t_params);
    }
    if(t->class->num_t_params > TypeTNType(t)) {
      TypeError("Too few type parameters (%d) for type `%s'; expected %d", 
		TypeTNType(t), t->class->name, t->class->num_t_params);
    }
  }

  /* compute size */
  type = TypeCreate(TYPE_TYPE, t);
  TypeComputeSize(type);

  return type;
}

/* Returns the string representation of type, s.t. 
 * TypeParse(TypeUnparse(type)) == type.
 * Uses shorthand whenever possible.
 */
void TypeUnparse(FILE *fp, Type type)
{
  int i;

  if(type->tag == TYPE_UNKNOWN) {
    fprintf(fp, "?%s", TypeUName(type));
    return;
  }
  if(type->tag == TYPE_INT) {
    fprintf(fp, "%d", type->value.i);
    return;
  }
  if(type->tag == TYPE_LIST) {
    ListNode p;
    ListIterate(p, type->value.l) {
      TypeUnparse(fp, (Type)p->data);
      if(p->next) fprintf(fp, ", ");
    }
    return;
  }
  /* tag == TYPE_TYPE */

  /* shorthand; "array[3]" -> "3" */
  if(!strcmp(TypeClass(type), "array") && TypeNInt(type) == 1
     && !TypeIntU(type,0)) {
    fprintf(fp, "%d", TypeInt(type,0));
  }
  else {
    fprintf(fp, "%s", TypeClass(type));
    if(TypeNInt(type)) {
      fprintf(fp, "[");
      TypeUnparse(fp, &type->value.t->i_params);
      fprintf(fp, "]");
    }
  }

  /* special case; don't alter "struct(a,b)" */
  if(!strcmp(TypeClass(type), "struct")) {
    fprintf(fp, "(");
    TypeUnparse(fp, &type->value.t->t_params);
    fprintf(fp, ")");
    return;
  }

  if(TypeNType(type) > 1) {
    fprintf(fp, "(");
    for(i=0;i<TypeNType(type)-1;i++) {
      TypeUnparse(fp, TypeType(type,i));
      if(i < TypeNType(type)-2) fprintf(fp, ", ");
    }
    fprintf(fp, ")");
  }
  /* shorthand; put last t_param outside of parens */
  if(TypeNType(type)) {
    fprintf(fp, " ");
    i = TypeNType(type)-1;
    TypeUnparse(fp, TypeType(type,i));
  }
}
