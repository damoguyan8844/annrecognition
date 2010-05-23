#include "photobook.h"

void watchProc(Ph_Object obj, char *field, void *userData)
{
  char *value;
  Ph_ObjGetString(obj, field, &value);
  printf("%s%s%s\n", field, (char*)userData, value);
  free(value);
}

/* usage: test-obj <database> <object>
 */
void main(int argc, char *argv[])
{
  char *db_name = "textures";
  char *obj_name = "view/image";
  Ph_Handle phandle;
  char field[100], value[100], *getvalue;
  Ph_Object obj;
  int i;
  
  if(argc > 1) db_name = argv[1];
  if(argc > 2) obj_name = argv[2];

  phandle = Ph_Startup();
  if(Ph_SetDatabase(phandle, db_name) == PH_ERROR) {
    fprintf(stderr, "Could not set database `%s'\n", db_name);
    exit(1);
  }

  obj = PhLookupObject(phandle, obj_name);
  if(!obj) {
    printf("No object `%s'\n", obj_name);
    exit(1);
  }
  for(i=0;obj->fields[i].name;i++) {
    if(Ph_ObjWatch(obj, obj->fields[i].name, 
		   watchProc, " is now ") == PH_ERROR) {
      printf("Error setting watch\n");
    }
  }

  for(;;) {
    printf("\nField > ");
    gets(field);
    if(!strcmp(field, "")) break;
    printf("Value > ");
    gets(value);
    if(!value[0]) {
      if(Ph_ObjGetString(obj, field, &getvalue) == PH_ERROR) {
	printf("An error occurred\n");
      }
      else {
	printf("%s\n", getvalue);
	free(getvalue);
      }
    }
    else if(Ph_ObjSetString(obj, field, value) == PH_ERROR) {
      printf("An error occurred\n");
    }
  }
  Ph_Shutdown(phandle);
}
