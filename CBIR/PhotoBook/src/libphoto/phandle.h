/* Definition of the Photobook handle "Ph_Handle" structure */

typedef struct Ph_HandleStruct {
  /* Global */
  char *data_dir;

  /* Database */
  char *db_name;
  List metrics, views; /* list of Ph_Object */
  Ph_Object cur_metric, cur_view;

  /* Members */
  Ph_Member *total_set;
  Ph_Member *working_set;
  int total_members, ws_members;

  /* Errors */
  char error_string[1000];

  /* FRAMER */
  char *framer_file;
  int framer_image_mode;
  Frame db_frame, member_frame, symbol_frame, rs_frame, proto_frame;
} *Ph_Handle;

#define Ph_NumMembers(phandle) (phandle)->total_members
#define Ph_WSMembers(phandle) (phandle)->ws_members
