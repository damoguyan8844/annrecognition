/* C Mode */

#ifndef DEBUG_EVAL
#define DEBUG_EVAL 1
#endif
#if DEBUG_EVAL
#define traced_apply d_apply
#define traced_eval fraxl_eval
#else
#define deterministic_apply d_apply
#define straight_eval fraxl_eval
#endif

#ifndef CLK_TCK
#ifdef CLOCKS_PER_SEC
#define CLK_TCK CLOCKS_PER_SEC
#endif
#endif


/* Macros for the evaluator */

/* This macro handles forms evaluated for side effect; it just calls
    side_eval_pair and doesn't bother looking up variables. */
#define side_eval(x,env) \
    if ((TYPEP(x,pair_ground)) && (x != empty_list)) side_eval_pair(x,env)
/* This is like eval but does a tail call (a throw to the current stack frame)
   if its okay to tail call. */
#define tail_eval(x,env) \
  ((TYPEP(x,pair_ground) ? \
    ((x == empty_list) ? (USED_GROUND(x)) \
    : ((okay_to_tail_call) ? (tail_eval_pair(x,env)) : eval_pair(x,env))) \
    : ((TYPEP(x,symbol_ground)) ? (lookup_variable(x,env)) \
       : (USED_GROUND(x)))))

/* This does a tail apply when appropriate */
#define tail_apply(rail) \
  ((okay_to_tail_call) ? (nd_tail_apply(rail)) : (nd_apply(rail)))
#define tail_dapply(rail) \
  ((okay_to_tail_call) ? (d_tail_apply(rail)) : (d_apply(rail)))


/* Evaluator variable declarations */

/* These are all special forms in the evaluator. */
extern Grounding lambda_symbol, define_symbol, let_star_symbol, else_symbol;
/* These are various symbols declared elsewhere (actually, in the reader). */
extern Grounding t_symbol, quote_symbol, backquote_symbol, unquote_symbol;

extern exception Eval_Failure, Type_Error, Arity_Error, Not_A_Function,
  Unbound_Variable, Non_Message, Unhandled_Message, Syntax_Error,
  FRAXL_Error, FRAXL_Exception;
extern void (*crisis_handler)();
void call_fraxl_debugger(exception ex);
void ground_error(exception ex,char *herald,Grounding irritant);


/* Evaluator data structures */

/* Flags for the evaluator */
extern boolean break_snips, check_compound_arity, trace_allocation, debugging;

/* Used in the read/eval/print loop */
extern long ground_memory_in_use;

/* Used to control tail recursion, okay_to_tail_call is set by the interpreter,
   use_tail_recursion set by the user. */
extern boolean okay_to_tail_call, use_tail_recursion;
/* This is used to do a tail call */
extern exception Non_Local_Exit;

/* Maintaining the stack */
struct STACK_FRAME { int depth; boolean looping, stepping;
		     Grounding expr, env, choices, choice; 
		     struct STACK_FRAME *previous; };
extern struct STACK_FRAME *current_stack_frame, *throw_to;
extern Grounding throw_value;
extern int max_stack_limit;


/* These macros iterate over expressions, skipping comments */
#define DO_FORMS(x,expr) \
    Grounding _tmp, x; boolean abort=False;  \
    _tmp=(expr); \
    while ((NOT(abort)) && \
	   (((TYPEP(_tmp,pair_ground)) && (_tmp != empty_list)) ? \
	    (x=GCAR(_tmp), _tmp=GCDR(_tmp)) : 0))
/* This iterates over all but the last expression in a list of expressions
   and leaves that expression in <rest> */
#define DO_SIDE_EFFECT_FORMS(x,expr,rest) \
    Grounding _tmp, x;   \
    _tmp=(expr); \
    if ((GCDR(_tmp)) == empty_list) \
      final=GCAR(_tmp); \
    else while (((TYPEP(_tmp,pair_ground)) && (_tmp != empty_list)) ? \
                 (x=GCAR(_tmp), _tmp=GCDR(_tmp)) : 0)                  \
            if (_tmp == empty_list) final=x; else \

/* This places the next expression in <expr> into <form> and `advances'
   <expr> to point past it. */
#define POP_FORM(into,expr) \
    while ((TYPEP(expr,pair_ground)) && (expr != empty_list) && \
	   (TYPEP(GCAR(expr),comment_ground))) expr=GCDR(expr); \
    if ((NOT(TYPEP(expr,pair_ground))) || (expr == empty_list)) \
      ground_error(Syntax_Error,"should be a list of expressions: ",expr); \
    else if (expr != empty_list) {into=GCAR(expr); expr=GCDR(expr);} \
    else into=NULL;

#define KEYWORD_P(sym) ((the(symbol,sym)->dispatch) != NULL)
#define MACRO_P(sym) ((the(symbol,sym)->dispatch) == expand_macro)


/* Inline consing (these are consing macros used by the evaluator) */

/* Head consing */
#define HCONS(X,Y,INTO) \
  {Grounding _tmp; INITIALIZE_NEW_GROUND(_tmp,pair_ground); \
   GCAR(_tmp)=X; GCDR(_tmp)=Y; USE_GROUND(_tmp); INTO=_tmp;}

/* Tail consing */
#define TCONS(X,INTO) \
  {Grounding _ground; INITIALIZE_NEW_GROUND(_ground,pair_ground); \
   GCAR(_ground)=X; GCDR(_ground)=(*INTO); \
   USE_GROUND(_ground); *INTO=_ground; INTO=(&(GCDR(_ground)));}

struct RESULT_ACCUMULATOR
{Grounding cheap[4]; Grounding *accumulator, *ptr, *limit; int size;};

#define WITH_RESULT_ACCUMULATOR(r) \
   struct RESULT_ACCUMULATOR *r, _r_acc; r=&_r_acc; \
   _r_acc.ptr=_r_acc.accumulator=_r_acc.cheap; \
   _r_acc.size=4; _r_acc.limit=_r_acc.ptr+4;


/* Evaluator definitions */

Grounding eval_pair(Grounding expr,Grounding env);
void side_eval_pair(Grounding expr,Grounding env);
Grounding tail_eval_pair(Grounding expr,Grounding env);
Grounding nd_tail_apply(Grounding rail);
Grounding d_tail_apply(Grounding rail);
Grounding nd_apply(Grounding rail);
Grounding d_apply(Grounding rail);
Grounding nd_applier(Grounding nd_rail,Grounding d_rail);
Grounding extend_results(Grounding results,Grounding by);
Grounding declare_keyword(char *name,Grounding (*fcn)());
Grounding declare_macro(char *name,Grounding expander);

int fraxl_debugger(char *herald,struct STACK_FRAME *frame);
void print_stack(struct STACK_FRAME *stack,generic_stream *stream,
		 Grounding focus,int width,int stack_window);

void add_binding(Grounding var,Grounding val,Grounding env);

void output_trace(char *header,Grounding expr);
void output_trace2(char *header,Grounding expr,char *header2,Grounding expr2);
void setup_combination(struct STACK_FRAME *stack);

void accumulate_result(Grounding result,struct RESULT_ACCUMULATOR *ra);
Grounding resolve_accumulator(struct RESULT_ACCUMULATOR *ra);

Grounding list_to_result_set(Grounding list);
Grounding finalize_result_list(Grounding list);

