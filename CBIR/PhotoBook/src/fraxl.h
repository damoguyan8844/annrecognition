Grounding add_to_ground AP((Frame frame,Grounding ground));
/* This adds <ground> to the ground of <frame> */

Grounding remove_from_ground AP((Frame frame,Grounding ground));
/* This removes <ground> from the ground of <frame> */

void read_eval_print(FILE *in,FILE *out,void (*iter_fn)(),void (*error_fn)());

Grounding in_list AP((Grounding elt,Grounding lst));
Grounding reverse_list AP((Grounding l2));
/* int count(Grounding elt,Grounding lst,int so_far); */
Grounding remove_from_list AP((Grounding elt,Grounding lst));
Grounding remove_from_list_once AP((Grounding elt,Grounding lst));
Grounding pair_car AP((Grounding pair));
Grounding pair_cdr AP((Grounding pair));
Grounding pair_cadr AP((Grounding pair));
Grounding listref AP((Grounding pair,Grounding index));
Grounding listcdrref AP((Grounding pair,Grounding index));
Grounding mapcar_primitive AP((Grounding fcn,Grounding list));
Grounding all_elements AP((Grounding pair));

Grounding generic_less_than AP((Grounding x,Grounding y));
Grounding generic_greater_than AP((Grounding x,Grounding y));
Grounding generic_plus AP((Grounding x,Grounding y));
Grounding generic_minus AP((Grounding x,Grounding y));
Grounding generic_times AP((Grounding x,Grounding y));
Grounding generic_divide AP((Grounding x,Grounding y));

Grounding gvref AP((Grounding vector,Grounding frame));
Grounding substring_primitive AP((Grounding string,Grounding start,Grounding end));
Grounding find_substring_primitive AP((Grounding string,Grounding in_string));
Grounding intern_primitive AP((Grounding string));
Grounding symbol_name_primitive AP((Grounding sybol));

Grounding clear_frame_prototype AP((Frame frame));
Grounding clear_frame_ground AP((Frame frame));
Grounding fraxl_frame_home AP((Frame frame));
Grounding fraxl_frame_name AP((Frame frame));
Grounding fraxl_frame_annotations AP((Frame frame));
Grounding fraxl_frame_spinoffs AP((Frame frame));
Grounding fraxl_use_annotation AP((Frame frame,Grounding name));
Grounding fraxl_make_annotation AP((Frame frame,Grounding name));
Grounding fraxl_make_unique_annotation AP((Frame frame,Grounding name));
Grounding fraxl_make_ephemeral_annotation AP((Frame frame,Grounding name));
Grounding fraxl_make_alias AP((Frame frame,Grounding name,Frame of_frame));
Grounding fraxl_move_frame AP((Frame frame,Frame new_home,Grounding new_name));
Grounding fraxl_probe_annotation AP((Frame frame,Grounding name));
Grounding fraxl_local_probe_annotation AP((Frame frame,Grounding name));
Grounding fraxl_soft_probe_annotation AP((Frame frame,Grounding name));
Grounding fraxl_delete_annotation AP((Frame frame));
Grounding fraxl_inherits_frame AP((Frame frame,Grounding name));
Grounding describe_frame_primitive AP((Frame frame));
Grounding parse_frame_path_primitive AP((Grounding under_frame));

Grounding make_table AP((Grounding creator));
Grounding string_cell AP((Grounding column,Grounding string));
Grounding ground_cell AP((Grounding column,Grounding string));
Grounding compound_cell AP((Grounding column,Grounding string));

Grounding backup_root_to_file AP((Grounding string));
Grounding set_backup_path_primitive AP((Grounding path));
Grounding load_frame_file AP((Grounding string));
Grounding touch AP((Frame frame));

Grounding eval_pair AP((Grounding expr,Grounding env));
Grounding lookup_variable AP((Grounding expr,Grounding env));
Grounding fraxl_eval AP((Grounding from));
extern Grounding apply0 AP((Grounding fcn));
extern Grounding apply1 AP((Grounding fcn,Grounding arg1));
extern Grounding apply2 AP((Grounding fcn,Grounding arg1,Grounding arg2));
extern Grounding apply3 AP((Grounding fcn,Grounding arg1,Grounding arg2,Grounding arg3));
extern int procedure_arity AP((Grounding fcn));
extern Grounding procedure_arguments AP((Grounding fcn));
extern Frame get_handlers AP((Frame to));
extern Grounding get_handler AP((Frame to,char *message));
extern Frame get_handler_frame AP((Frame to,char *message));
extern Grounding send0 AP((Frame to,char *message));
extern Grounding send1 AP((Frame to,char *message,Grounding arg1));
extern Grounding send2 AP((Frame to,char *message,Grounding arg1,Grounding arg2));

/* This macro handles the first level of eval dispatching;
   it either returns its argument or calls eval_pair or lookup_variable. */
#define eval(x,env) \
   ((TYPEP(x,pair_ground) ? \
    ((x == empty_list) ? (USED_GROUND(x)) : (eval_pair(x,env))) : \
     ((TYPEP(x,symbol_ground)) ? (lookup_variable(x,env)) \
      : (USED_GROUND(x)))))

Grounding framep(Grounding x);
Grounding stringp(Grounding x);
Grounding symbolp(Grounding x);

Grounding listp(Grounding x);
Grounding nullp(Grounding x);
Grounding vectorp(Grounding x);

Grounding functionp(Grounding x);
Grounding primitivep(Grounding x);

Grounding fixnump(Grounding x);
Grounding floatp(Grounding x);

Grounding numberp(Grounding x);
Grounding rationalp(Grounding x);
Grounding integerp(Grounding x);

Frame local_probe_annotation(Frame frame,char *name);

Grounding print_ground_to_stdout(Grounding x);
Grounding display_ground_to_stdout(Grounding x);
Grounding parse_ground_from_stdin(void);
Grounding print_string_ground_to_stdout(Grounding x);
Grounding newline_to_stdout(void);
Grounding print_result_to_stdout(Grounding x);
Grounding parse_ground_from_string(char *string);
char *print_ground_to_string(Grounding ground);
void ground_error(exception ex,char *herald,Grounding irritant);

Grounding binding_output_to_string(Grounding thunk);
Grounding binding_output_to_file(Grounding file,Grounding thunk);
Grounding binding_input_to_string(Grounding string,Grounding thunk);
Grounding binding_input_to_file(Grounding file,Grounding thunk);
Grounding file_as_string(Grounding filename);

Grounding declare_lexpr(Grounding (*fcn)(),char *name);
Grounding declare_function(Grounding (*fcn)(),char *string,int arity,
			   Ground_Type type0,Ground_Type type1,
			   Ground_Type type2,Ground_Type type3);

#if defined(__cplusplus)  /* Wed Nov 24 16:14:14 1993  Maia Engeli */
Grounding declare_big_function(Grounding (*fcn)(...),char *string,int arity,
			       Ground_Type type0,Ground_Type type1,
			       Ground_Type type2,Ground_Type type3,       
			       Ground_Type type4,Ground_Type type5,
			       Ground_Type type6,Ground_Type type7,
			       Ground_Type type8,Ground_Type type9);
Grounding declare_unary_function(Grounding (*fcn)(Grounding),char *string,
				 Ground_Type type0);
Grounding declare_binary_function(Grounding (*fcn)(Grounding,Grounding),char *string,
				  Ground_Type type0,Ground_Type type1);
#else  /* !__cplusplus */
Grounding declare_big_function(Grounding (*fcn)(),char *string,int arity,
			       Ground_Type type0,Ground_Type type1,
			       Ground_Type type2,Ground_Type type3,		       
			       Ground_Type type4,Ground_Type type5,
			       Ground_Type type6,Ground_Type type7,
			       Ground_Type type8,Ground_Type type9);
Grounding declare_unary_function(Grounding (*fcn)(),char *string,Ground_Type type0);

Grounding declare_binary_function(Grounding (*fcn)(),char *string,
				  Ground_Type type0,Ground_Type type1);
#endif  /* __cplusplus */

#define frame_fcn(x) ((Grounding (*)()) x)


/* ARLOtje functions */

Grounding get_elements(Frame frame);
Grounding add_element(Frame frame,Grounding value);
Grounding remove_element(Frame frame,Grounding value);
Grounding has_element(Frame frame,Grounding value);

Grounding get_value(Frame unit,char *slot);
Grounding put_value(Frame unit,char *slot,Grounding value);
Grounding retract_value(Frame unit,char *slot,Grounding value);
Grounding test_value(Frame unit,char *slot,Grounding value);

Grounding get_value_primitive(Frame unit,Grounding slot);
Grounding put_value_primitive(Frame unit,Grounding slot,Grounding value);
Grounding retract_value_primitive(Frame unit,Grounding slot,Grounding value);
Grounding test_value_primitive(Frame unit,Grounding slot,Grounding value);

Grounding with_indentation(Grounding indent,Grounding thunk);
Grounding with_table(Grounding indent,Grounding thunk);
Grounding with_table_entry(Grounding column,Grounding thunk);
Grounding value_column(Grounding with);
Grounding labelled_value_column(Grounding label,Grounding with);

extern Grounding define_symbol;


/* Declaring primitives (macro support) */

#define declare_fcn0(c_name,fraxl_name) \
  declare_big_function(c_name,fraxl_name,0,\
                       any_ground,any_ground,any_ground,any_ground,\
                       any_ground,any_ground,any_ground,any_ground,\
                       any_ground,any_ground);
#define declare_fcn1(c_name,fraxl_name,type1) \
  declare_big_function(c_name,fraxl_name,1,\
                       type1,any_ground,any_ground,any_ground,\
                       any_ground,any_ground,any_ground,any_ground,\
                       any_ground,any_ground);
#define declare_fcn2(c_name,fraxl_name,type1,type2) \
  declare_big_function(c_name,fraxl_name,2,\
                       type1,type2,any_ground,any_ground,\
                       any_ground,any_ground,any_ground,any_ground,\
                       any_ground,any_ground);
#define declare_fcn3(c_name,fraxl_name,type1,type2,type3) \
  declare_big_function(c_name,fraxl_name,3,\
                       type1,type2,type3,any_ground,\
                       any_ground,any_ground,any_ground,any_ground,\
                       any_ground,any_ground);
#define declare_fcn4(c_name,fraxl_name,type1,type2,type3,type4) \
  declare_big_function(c_name,fraxl_name,4,\
                       type1,type2,type3,type4,\
                       any_ground,any_ground,any_ground,any_ground,\
                       any_ground,any_ground);
#define declare_fcn5(c_name,fraxl_name,type1,type2,type3,type4,type5) \
  declare_big_function(c_name,fraxl_name,5,\
                       type1,type2,type3,type4,\
                       type5,any_ground,any_ground,any_ground,\
                       any_ground,any_ground);
#define declare_fcn6(c_name,fraxl_name,type1,type2,type3,type4,type5,type6) \
  declare_big_function(c_name,fraxl_name,6,\
                       type1,type2,type3,type4,\
                       type5,type6,any_ground,any_ground,\
                       any_ground,any_ground);
#define declare_fcn7(c_name,fraxl_name,type1,type2,type3,type4,type5,type6,type7) \
  declare_big_function(c_name,fraxl_name,7,\
                       type1,type2,type3,type4,\
                       type5,type6,type7,any_ground,\
                       any_ground,any_ground);
#define declare_fcn8(c_name,fraxl_name,type1,type2,type3,type4,type5,type6,type7,type8) \
  declare_big_function(c_name,fraxl_name,8,\
                       type1,type2,type3,type4,\
                       type5,type6,type7,type8,\
                       any_ground,any_ground);
#define declare_fcn9(c_name,fraxl_name,type1,type2,type3,type4,type5,type6,type7,type8,type9) \
  declare_big_function(c_name,fraxl_name,9,\
                       type1,type2,type3,type4,\
                       type5,type6,type7,type8,\
                       type9,any_ground);
#define declare_fcn10(c_name,fraxl_name,type1,type2,type3,type4,type5,type6,type7,type8,type9,type10) \
  declare_big_function(c_name,fraxl_name,10,\
                       type1,type2,type3,type4,\
                       type5,type6,type7,type8,\
                       type9,type10);


/* Initializing FRAMER */

Grounding run_init(Grounding expr);
void init_framer_functions(void);
void init_description_functions(void);
void init_arlotje(void);
void init_mnemosyne(void);

#ifndef CUSTOMIZE_FRAXL
#define CUSTOMIZE_FRAXL()
#endif

#if (NETWORKING)
void init_network_functions(void);
#define INITIALIZE_FRAXL() \
  init_framer_functions(); \
  init_description_functions(); \
  init_arlotje(); \
  init_mnemosyne(); \
  init_network_functions(); \
  CUSTOMIZE_FRAXL(); \
  run_init((Grounding) probe_annotation(use_annotation(root_frame,"system"),"inits"))
#else
#define INITIALIZE_FRAXL() \
  init_framer_functions(); \
  init_description_functions(); \
  init_arlotje(); \
  init_mnemosyne(); \
  CUSTOMIZE_FRAXL(); \
  run_init((Grounding) probe_annotation(use_annotation(root_frame,"system"),"inits"))
#endif /* (NETWORKING) */

#undef INITIALIZE_FRAMER
#define INITIALIZE_FRAMER() \
  INITIALIZE_JUST_FRAMER(); \
  INITIALIZE_FRAXL();



