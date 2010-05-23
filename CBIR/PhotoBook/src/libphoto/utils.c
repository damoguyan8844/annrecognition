#include <framer.h>
#include <fraxl.h>
#include <stdarg.h>

/* Applies the use_annotation command repeatedly to traverse a Frame tree.
 * root = starting Frame
 * Arguments following must be strings, and specify the annotations to be
 * applied, in order. The list must be terminated with an empty string.
 */
Frame subframe(Frame root, ...)
{
  va_list args;
  char *name;

  va_start(args, root);
  while(root) {
    name = va_arg(args, char*);
    if(!*name) break;
    root = probe_annotation(root, name);
  }
  return root;
}
