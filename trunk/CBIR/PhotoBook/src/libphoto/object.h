/* Definitions for Photobook object (Ph_Object) structure */
#include <type/type.h>

typedef struct {
  char *name;
  char *type_s;
} ObjClassField;

struct ObjectStruct;
typedef void ObjFieldCallback(struct ObjectStruct *obj, 
			      char *field, void *userData);
typedef struct {
  ObjFieldCallback *callback;
  void *userData;
} ObjFieldCB;

typedef void GenericFunc();
typedef struct {
  char *name;
  GenericFunc *func;
} ObjClassFunc;

typedef struct ObjClassStruct {
  char *name;
  int size;
  ObjClassField *fields;
  ObjClassFunc *funcs;
  struct ObjClassStruct *super;
} *ObjClass;

typedef struct {
  char *name;
  void *data; /* pointer into obj->data */
  Type type;
  List callbacks;
  char shared_type;
} ObjField;

/* Transient field */
/* This structure is designed to look like an ObjField */
typedef struct {
  char *name;
  void *data;                    /* pointer to cached data, if any */
  Type type;
  List callbacks;
  char shared_type;
  char cache_flag;               /* 1 = cached, 2 = don't cache */
  char use_disk;                 /* 0 = don't load */
  struct ObjectStruct *producer; /* producer object, if any */
  char *p_field;                 /* function field of producer object */
  void *p_data;                  /* argument to producer function */
} ObjTField;

struct Ph_HandleStruct;
typedef struct ObjectStruct {
  char *name;
  ObjClass class;
  void *data;
  ObjField *fields;
  struct ObjectStruct *super;
  struct Ph_HandleStruct *phandle;
  List trans_fields; /* List of ObjTField */
  int cache_used, cache_limit;
} *Ph_Object;

/* function types */

typedef void PhObjConstructor(Ph_Object self);
typedef void PhObjDestructor(Ph_Object self);

#define Ph_ObjName(o) (o)->name
#define Ph_ObjClass(o) (o)->class->name

/* object.c ******************************************************************/

Ph_Object Ph_ObjCreate(struct Ph_HandleStruct *phandle, 
		       ObjClass class, char *name);
void Ph_ObjFree(Ph_Object obj);
ObjClass PhClassFind(List class_list, char *class_name);
Ph_Object PhObjFind(List obj_list, char *name);
ObjField *Ph_ObjField(Ph_Object obj, char *field);

int Ph_ObjSetString(Ph_Object obj, char *field, char *value_s);
int Ph_ObjSet(Ph_Object obj, char *field, void *value);
int Ph_ObjGetString(Ph_Object obj, char *field, char **value);
int Ph_ObjGet(Ph_Object obj, char *field, void *value);

int Ph_ObjWatch(Ph_Object obj, char *field, 
		ObjFieldCallback *func, void *userData);
void Ph_ObjAddField(Ph_Object obj, char *field, Type type,
		    Ph_Object producer, char *p_field, void *p_data,
		    char dont_cache, char dont_load);

ObjField *PhObjField(Ph_Object obj, char *field);
ObjTField *PhObjTField(Ph_Object obj, char *field);
GenericFunc *PhObjFunc(Ph_Object obj, char *field);
