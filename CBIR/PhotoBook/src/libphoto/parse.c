/* Module: command parser */

#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdarg.h>
#include "photobook.h"

/* Globals *******************************************************************/

#define NUM_OPS 4
static char *op_list[NUM_OPS] = {
  "and", "or", "not", "except"
};
static char *sub_list[NUM_OPS] = {
  "intersect", "amb", 
  "difference (frame-annotations #^/)", "difference"
};

/* Local phandle (ugly) */
static Ph_Handle phandle;
static char errorFlag;

/* Prototypes ****************************************************************/

char *ParseFilter(Ph_Handle phandle, char *filter);

/* Private */
static char *parse_phrase(char *phrase);
static List count_words(char *string);
static char *word_to_framer_speak(char *word);
static void parse_error(char *str, ...);
static void strip_outer_parens(char *word);
static int unary_op(char *word);
static int any_op(char *word);
static char *append_word(char *s, char *w);

/* Functions *****************************************************************/

char *ParseFilter(Ph_Handle phandle_in, char *filter)
{
  char *phrase, *result;

  phandle = phandle_in;
  /* clear any existing error */
  errorFlag = 0;

  phrase = strdup(filter);
  result = parse_phrase(phrase);
  free(phrase);
  return result;
}

/* Translate a user input string into a fraxl string.
 * return = fraxl string
 * Caller must free the returned string.
 * Modifies phrase.
 */
static char *parse_phrase(char *phrase)
{
  char *result, *first_word, *str;
  int num_words;
  List wordlist;
  ListNode ptr;

  /* Break the phrase into words */
  wordlist = count_words(phrase);
  num_words = ListSize(wordlist);
  result = NULL;

  if(!errorFlag) {
    if(!num_words) {
      result = strdup("");
    }
    else {
      first_word = (char*)ListFront(wordlist)->data;
      /* Non-operator: translate the first word and return. */
      if(!any_op(first_word)) {
	if(num_words == 1) {
	  result = word_to_framer_speak(first_word);
	}
	else {
	  parse_error("Unrecognized operator \"%s\".", first_word);
	}
      }
      /* Is an operator application */
      else {
	if(unary_op(first_word)) {
	  /* Enforce unary operator constraints */
	  if(num_words != 2) {
	    parse_error("The operator \"%s\" needs one argument.", first_word);
	    goto Lexit;
	  }
	}
	else if (num_words < 3) {
	  /* Enforce binary operator constraints */
	  parse_error("The operator \"%s\" needs at least two arguments.", 
		      first_word);
	  goto Lexit;
	}
	/* translate the operator */
	str = word_to_framer_speak(first_word);
	if(!str) goto Lexit;
	result = malloc(strlen(str) + 2);
	sprintf(result, "(%s", str);
	free(str);
	/* call parse_phrase recursively on the operands */
	ListIterate(ptr, wordlist) {
	  if(ptr == ListFront(wordlist)) continue;
	  str = parse_phrase((char*)ptr->data);
	  if(!str) { free(result); result = NULL; goto Lexit; }
	  result = realloc(result, strlen(result)+strlen(str)+1);
	  strcat(result, str);
	  free(str);
	}
	result = realloc(result, strlen(result) + 2);
	strcat(result, ")");
      }
    }
  }

 Lexit:
  ListFree(wordlist);
  return(result);
}

/* Break phrase up into words. Words may be delimited by whitespace or
 * parentheses. For example, "(and (adult)(male))" has three words.
 * Returns list of strings.
 * Modifies phrase.
 */
static List count_words(char *phrase)
{
  int paren_count;
  char *p, *q;
  List wordlist;

  /* Strip the outermost parentheses */
  strip_outer_parens(phrase);
  
  /* If an error has occurred, return the empty allocated space */
  wordlist = ListCreate((FreeFunc*)GenericFree);
  if(errorFlag) return wordlist;
  
  /* Iterate through the string */
  for(p=phrase;*p;p++) {
    if(isspace(*p)) continue;
    if(*p == '(') {
      /* Open paren: All characters until the closing paren are part of the
       * word. Be careful about nested parentheses!
       */
      paren_count = 0;
      for(q=p+1;*q;q++) {
	if (*q == '(')
	  paren_count++;
	else if(*q == ')') {
	  if(paren_count == 0) {
	    break;
	  }
	  else paren_count--;
	}
      }
      if(!*q) {
	parse_error("Missing closing parenthesis.");
	goto done;
      }
      q++;
    }
    else if(*p == '[') {
      /* Open brace: read until closing brace. No nesting allowed.
       * Braces are stripped.
       */
      for(q=++p;*q && (*q != ']');q++);
      if(!*q) {
	parse_error("Missing closing brace.");
	goto done;
      }
    }
    else if(*p == '"') {
      /* Open quote: read until closing quote. No nesting allowed.
       * Quotes are not stripped.
       */
      for(q=p+1;*q && (*q != '"');q++);
      if(!*q) {
	parse_error("Missing closing quote.");
	goto done;
      }
      q++;
    }
    else {
      /* Easy case: Just read characters until whitespace. */
      for(q = p;*q && !isspace(*q);q++);
    }
    if(*q) {
      *q = 0;
      ListAddRear(wordlist, strdup(p));
      p = q;
    }
    else {
      ListAddRear(wordlist, strdup(p));
      break;
    }
  }
  
 done:
  return(wordlist);
}

/* Translate a simple word into fraxl notation.
 * Returned string must be freed by caller.
 * Returns NULL on parse error.
 */
static char *word_to_framer_speak(char *phrase)
{
  int i;
  char *result;
  char *p, *str;

  /* Allocate result string */
  result = malloc(1000);
  result[0] = 0;

  /* Check for bad characters, namely the # " ( ) { } signs. */
  for(i=0;phrase[i];i++) {
    if(strchr("#{}", phrase[i])) {
      parse_error("Invalid character `%c'", phrase[i]);
      goto Lerror;
    }
    if(strchr("()", phrase[i])) {
      parse_error("Badly placed `%c'", phrase[i]);
      goto Lerror;
    }
    if((phrase[i] == '"') && i) {
      if(!phrase[i+1] && (*phrase == '"')) continue;
      parse_error("Badly placed quotes");
      goto Lerror;
    }
  }

  /* Operators are mapped according to sub_list */
  if(i = any_op(phrase)) {
    strcpy(result, sub_list[i-1]);
    return result;
  }

  /* Is it a text request? (starts with quote) */
  if(*phrase == '"') {
    /* Find ending quote and terminate the string. 
     * Also converts to lower case.
     */
    if(debug) printf("Text request: %s\n", phrase);
    for(p = ++phrase;*p && (*p != '"');p++) *p = tolower(*p);
    *p = 0;
    /* Search all frames which have string annotations,
     * and form the result set of those which contain the given string.
     * Note that the match may occur in any textual annotation.
     * The result set can get long, so we use realloc().
     */
    strcpy(result, "(amb");
    for(i=0;i<phandle->total_members;i++) {
      DO_ANNOTATIONS(ann, Ph_MemFrame(phandle->total_set[i])) {
	Grounding value = frame_ground(ann);
	if(!STRINGP(value)) continue;
	/* Convert the value to lower case */
	str = strdup(GSTRING(value));
	for(p = str;*p;p++) *p = tolower(*p);
	/* Is the request contained in str? */
	if(strstr(str, phrase)) {
	  p = Ph_MemName(phandle->total_set[i]);
	  result = realloc(result, strlen(result) + strlen(p) + 5);
	  sprintf(result, "%s #^/%s", result, p);
	  /* Don't continue with this frame */
	  free(str);
	  break;
	}
	free(str);
      }
    }
    result = realloc(result, strlen(result) + 2);
    strcat(result, ")");
    return result;
  }

  /* Find delimiting '/' character */
  for(i=0;(phrase[i] != '/') && phrase[i];i++);
  if(!phrase[i]) {
    /* No slash; assume they are requesting a specific image */
    if(debug) printf("Frame request: `%s'\n", phrase);
    if(probe_annotation(phandle->member_frame, phrase)) {
      sprintf(result, "{#^/%s}", phrase);
    }
    else {
      debugprint("Nonexistent frame!\n");
      strcpy(result, "{}");
    }
    return result;
  }

  /* Otherwise use the result set of the symbol frame */
  phrase[i++] = 0;
  if(subframe(phandle->symbol_frame, phrase, &phrase[i], "")) {
    strcpy(result, "(frame-ground #^^/symbols/");
    append_word(result, phrase);
    strcat(result, "/");
    append_word(result, &phrase[i]);
    strcat(result, "/set)");
  }
  else {
    int m;
    /* look in the string annotations */
    strcpy(result, "(amb");
    for(m=0;m<phandle->total_members;m++) {
      Grounding value;
      Frame ann = probe_annotation(Ph_MemFrame(phandle->total_set[m]), phrase);
      if(!ann) continue;
      value = frame_ground(ann);
      if(!STRINGP(value)) continue;
      if(!strcmp(GSTRING(value), &phrase[i])) {
	p = Ph_MemName(phandle->total_set[m]);
	result = realloc(result, strlen(result) + strlen(p) + 5);
	sprintf(result, "%s #^/%s", result, p);
      }
    }
    result = realloc(result, strlen(result) + 2);
    strcat(result, ")");
    return result;
/*
    parse_error("Bad category/item: %s/%s", phrase, &phrase[i]);
    goto Lerror;
*/
  }
  return result;
 Lerror:
  free(result);
  return NULL;
}

/* Print a message into the error_string and set the error_flag. */
static void parse_error( char *str, ...)
{
  va_list args;

  va_start(args, str);
  errorFlag = 1;
  vsprintf(phandle->error_string,str,args);
  if(debug) printf("Parse_error: '%s'\n", phandle->error_string);
  va_end(args);
}

/* Remove all outermost parenthesis, if any. */
static void strip_outer_parens(char *phrase)
{
  int i;

  /* Skip leading spaces */
  while(isspace(*phrase)) phrase++;

  /* If a parenthesis is the first character, */
  if (*phrase == '(') {
    /* remove it and its matching close paren. */
    *phrase = ' ';
    /* this algorithm assumes that no text follows the last closing paren */
    /* Walk backwards from the end of the string, skipping spaces */
    i = strlen(phrase) - 1;
    while((isspace(phrase[i])) && (i > 0))
      i--;
    if((i > 0) && (phrase[i] == ')')) {
      phrase[i] = '\0';
      /* Now strip again, recursively */
      strip_outer_parens(phrase);
    }
    else {
      parse_error("Missing closing parenthesis.");
    }
  }
}
    
/* Returns true iff word is a unary operator. */
static int unary_op(char *word)
{
  return(!strcasecmp(word, "not"));
}

/* Check if word is an operator.
 * Returns index into op_list, otherwise zero.
 */
static int any_op(char *word)
{
  int i = 0;

  while(i < NUM_OPS) {
    if(!strcasecmp(op_list[i], word))
      return(i+1);
    i++;
  }
  return(0);
}

/* Appends w to s, escaping spaces */
static char *append_word(char *s, char *w)
{
  char *p;

  for(p=s;*p;p++);
  while(*w) {
    if(isspace(*w)) *p++ = '\\';
    *p++ = *w++;
  }
  *p = '\0';
  return s;
}
