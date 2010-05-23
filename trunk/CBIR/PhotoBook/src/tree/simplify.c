/* Simplifies a tree by removing any nodes with exactly one child.
 */

#include <tpm/stream.h>
#include <tpm/gtree.h>

int main(int argc, char *argv[])
{
  Tree tree;

  tree = TreeRead(stdin, RealRead);
  if(!tree) exit(0);
  TreeSimplify(tree, GenericFree);
  TreeWrite(stdout, tree, RealWrite);
  exit(0);
}
