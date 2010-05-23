/* Generic tree implementation with arbitrary branching, ordered children,
 * and multiple roots.
 * A tree node has a data value and links to its parent, sibling, and 
 * first child. The second child is tree->child->sibling, etc. Any of these
 * links may be NULL. 
 * If tree->parent == NULL, then tree is a root node.
 * If tree->sibling == NULL, then tree is the rightmost child of its parent,
 * or rightmost root node.
 * If tree->child == NULL, then tree is a leaf.
 */

#include <tpm/list.h>
#include <tpm/gtree.h>

/* Prototypes ****************************************************************/

Tree TreeCreate(void *data);
void TreeFree(Tree tree, FreeFunc *fp);
Tree TreeRead(FILE *fp, ReadFunc *rf);
void TreeWrite(FILE *fp, Tree tree, WriteFunc *pf);
void TreeAddSibling(Tree tree, Tree subtree);
Tree TreeAddChild(Tree tree, Tree subtree);
Tree TreeAddLeftChild(Tree tree, Tree subtree);
Tree TreeAddChildUnique(Tree tree, Tree subtree);
Tree TreeRemoveChild(Tree tree);
Tree TreeAppend(Tree a, Tree b);
List TreeLeafList(Tree tree);
int TreeDepth(Tree subtree);
int TreeHeight(Tree tree);
int TreeSize(Tree tree);
int TreeCountLeaves(Tree tree);
int TreeCountNonLeaves(Tree tree);
Tree TreeCommonAncestor(Tree *a, Tree *b);
Tree TreeFirstLeaf(Tree tree);
Tree TreeCopy(Tree tree, CopyFunc *cf);
int TreeSimplify(Tree tree, FreeFunc *fp);

/* Functions *****************************************************************/

/* Returns a new tree node with the given data.
 * The node will have NULL for all links.
 */
/* E.g., t = TreeCreate(RealCreate(2.2)); */
Tree TreeCreate(void *data)
{
  Tree tree;
  
  tree = (Tree)malloc(sizeof(struct TreeStruct));
  if(!tree) {
    fprintf(stderr, "TreeCreate: Cannot allocate memory\n");
    fflush(stderr);
    return tree;
  }
  tree->child = tree->sibling = tree->parent = NULL;
  tree->data = data;
  return tree;
}

/* Frees the storage used by tree.
 * If fp != NULL, fp(data) is used to free the node data.
 * Otherwise node data is not freed.
 */
void TreeFree(Tree tree, FreeFunc *fp)
{
  Tree temp;

  while(tree) {
    TreeFree(tree->child, fp);
    temp = tree;
    tree = tree->sibling;
    if(fp) fp(temp->data);
    free(temp);
  }
}

/* Used by ReadTree */
static int nextchar(FILE *fp)
{
  int ch;

  while((ch = fgetc(fp)) != EOF) if(!isspace(ch)) break;
  return ch;
}

/* Read a tree in ASCII format from fp.
 * If rf != NULL, uses rf(fp) to read node data.
 * Otherwise the data is ignored.
      An example tree file is:
        3
        1
        : 2 0
        2
        : 2 1
      which specifies (in postfix notation) the tree
             1
            / \
           0   2
          / \
         3   1
      The lines without colons denote leaves; colons are followed by the 
      number of nodes to join and the value of the parent node.
 */
Tree TreeRead(FILE *fp, ReadFunc *rf)
{
  Tree tree, t1, t2;
  int ch;
  List node_stack;
  void *data;
  
  node_stack = ListCreate(NULL);

  while((ch = nextchar(fp)) != EOF) {
    if(ch == ':') {
      /* pop nodes off of the stack and merge them into a tree */
      fscanf(fp, "%d", &ch);
      if(rf) data = rf(fp);
      else data = NULL;
      tree = TreeCreate(data);

      while(ch--) {
	ListRemoveFront(node_stack, &data);
	if(!data) fprintf(stderr, "TreeRead: stack underflow\n");
	TreeAddLeftChild(tree, (Tree)data);
      }
      ListAddFront(node_stack, tree);
    }
    else {
      /* push new node onto the stack */
      if(rf) {
	ungetc(ch, fp);
	data = rf(fp);
      }
      else {
	data = NULL;
      }
      ListAddFront(node_stack, TreeCreate(data));
    }
  }

  /* make remaining nodes siblings */
  ListRemoveRear(node_stack, (void**)&tree);
  for(t1=tree;ListSize(node_stack);t1=t2) {
    ListRemoveRear(node_stack, (void**)&t2);
    t1->sibling = t2;
  }
  ListFree(node_stack);
  return tree;
}

static void _TreeWrite(FILE *fp, Tree tree, WriteFunc *pf)
{
  Tree t;
  int count;

  if(tree->child) {
    for(t = tree->child, count = 0; t; t = t->sibling, count++)
      _TreeWrite(fp, t, pf);
    fprintf(fp, ": %d ", count);
  }
  pf(fp, tree->data);
  fprintf(fp, "\n");
  fflush(fp);
}

/* Write a tree to fp in ASCII format.
 * If pf != NULL, uses pf(fp, data) to write node data.
 * Otherwise node data is not written.
 */
void TreeWrite(FILE *fp, Tree tree, WriteFunc *pf)
{
  for(;tree;tree=tree->sibling)
    _TreeWrite(fp, tree, pf);
}

/* Make subtree the last (rightmost) sibling of tree */
void TreeAddSibling(Tree tree, Tree subtree)
{
  Tree *t;
  assert(tree);
  if(!subtree) return;
  for(t = &tree->sibling; *t; t = &(*t)->sibling);
  *t = subtree;
  for(t = &subtree; *t; t = &(*t)->sibling) {
    (*t)->parent = tree->parent;
  }
}

/* Make subtree the last (rightmost) child of tree */
Tree TreeAddChild(Tree tree, Tree subtree)
{
  Tree *t;
  assert(tree);
  if(!subtree) return tree;
  for(t = &tree->child; *t; t = &(*t)->sibling);
  *t = subtree;
  for(t = &subtree; *t; t = &(*t)->sibling) {
    (*t)->parent = tree;
  }
  return tree;
}

/* Make subtree the first (leftmost) child of tree */
Tree TreeAddLeftChild(Tree tree, Tree subtree)
{
  Tree *t;
  assert(tree);
  if(!subtree) return tree;
  subtree->parent = tree;
  for(t = &subtree->sibling; *t; t = &(*t)->sibling) {
    (*t)->parent = tree;
  }
  *t = tree->child;
  tree->child = subtree;
  return tree;
}

/* Same as TreeAddChild, but does nothing 
 * if subtree is already a child of tree.
 */
Tree TreeAddChildUnique(Tree tree, Tree subtree)
{
  Tree *t;
  assert(tree);
  if(!subtree) return tree;
  for(t = &tree->child; *t; t = &(*t)->sibling)
    if(*t == subtree) return tree;
  *t = subtree;
  for(t = &subtree; *t; t = &(*t)->sibling) {
    (*t)->parent = tree;
  }
  return tree;
}

/* Dissociates tree from its parent.
 * No storage is freed.
 */
Tree TreeRemoveChild(Tree tree)
{
  Tree *t;
  assert(tree);
  for(t = &tree->parent->child; *t != tree; t = &(*t)->sibling);
  *t = tree->sibling;
  tree->parent = tree->sibling = NULL;
  return tree;
}

/* Append two trees to form a multi-rooted tree (a forest) */
Tree TreeAppend(Tree a, Tree b)
{
  Tree *t;

  if(!a) return b;
  if(!b) return a;
  for(t = &a->sibling; *t; t = &(*t)->sibling);
  *t = b;
  return a;
}

/* Form a list of the leaf data in tree, corresponding to a
 * left to right (postorder) traversal.
 */
List TreeLeafList(Tree tree)
{
  List list;

  assert(tree);
  list = ListCreate(NULL);
  if(TreeLeaf(tree)) {
    ListAddFront(list, tree->data);
  }
  else {
    {IterateChildren(tree, child) {
      ListAppend(list, TreeLeafList(child));
    }}
  }
  return list;
}

/* Returns the depth of subtree in its tree.
 * If subtree is the root (i.e. parent == NULL), depth = 0.
 */
int TreeDepth(Tree subtree)
{
  int depth;
  assert(subtree);
  for(depth = 0, subtree = subtree->parent; 
      subtree; depth++, subtree = subtree->parent);
  return depth;
}

/* Returns the maximum distance from tree to a leaf */
int TreeHeight(Tree tree)
{
  int h1, h2;
  if(!tree) return 0;
  h1 = TreeHeight(tree->child)+1;
  h2 = TreeHeight(tree->sibling);
  return max(h1, h2);
}

/* Returns the total number of nodes in tree */
int TreeSize(Tree tree)
{
  if(!tree) return 0;
  return TreeSize(tree->child) + TreeSize(tree->sibling) + 1;
}

/* Returns the number of leaf nodes in tree */
int TreeCountLeaves(Tree tree)
{
  int count;
  if(!tree) return 0;
  if(tree->child) count = TreeCountLeaves(tree->child);
  else count = 1;
  return count + TreeCountLeaves(tree->sibling);
}

/* Returns the number of interior nodes in tree */
int TreeCountNonLeaves(Tree tree)
{
  int count;
  if(!tree) return 0;
  if(tree->child) count = TreeCountNonLeaves(tree->child) + 1;
  else count = 0;
  return count + TreeCountNonLeaves(tree->sibling);
}

/* Returns the common ancestor of *a and *b.
 * If there is none, NULL is returned and 
 * post *a and post *b point to the roots of the respective trees 
 * of pre *a and pre *b.
 * Otherwise, post *a points to the subtree under the common ancestor 
 * containing pre *a, and same for *b.
 */
Tree TreeCommonAncestor(Tree *a, Tree *b)
{
  int a_depth, b_depth;

  a_depth = TreeDepth(*a);
  b_depth = TreeDepth(*b);
  for(;a_depth > b_depth;a_depth--,*a=(*a)->parent);
  for(;b_depth > a_depth;b_depth--,*b=(*b)->parent);
  for(;a_depth;a_depth--) {
    if((*a)->parent == (*b)->parent) return (*a)->parent;
    *a = (*a)->parent;
    *b = (*b)->parent;
  }
  return NULL;
}

/* Returns the leftmost leaf in tree. */
Tree TreeFirstLeaf(Tree tree)
{
  if(TreeLeaf(tree)) return tree;
  else return TreeFirstLeaf(tree->child);
}

/* Returns a copy of tree, using cf, if non-NULL, to copy node data. */
Tree TreeCopy(Tree tree, CopyFunc *cf)
{
  Tree result;
  if(!tree) return NULL;
  result = TreeCreate(cf ? cf(tree->data) : tree->data);
  result->child = TreeCopy(tree->child, cf);
  result->sibling = TreeCopy(tree->sibling, cf);
  return result;
}

/* Removes single arity nodes from tree. 
 * If fp != NULL, fp(data) is used to free the node data.
 * Otherwise node data is not freed.
 */
int TreeSimplify(Tree tree, FreeFunc *fp)
{
  int count;
  if(!tree) return 0;
  count = TreeSimplify(tree->child, fp);
  /* any siblings? */
  if(!tree->sibling && tree->parent) {
    /* Clobber parent's data */
    if(fp) fp(tree->parent->data);
    tree->parent->data = tree->data;
    /* Put children under parent */
    tree->parent->child = NULL;
    TreeAddChild(tree->parent, tree->child);
    free(tree);
    return count+1;
  }
  else {
    /* loop siblings */
    for(tree=tree->sibling; tree; tree=tree->sibling) {
      count += TreeSimplify(tree->child, fp);
    }
    return count;
  }
}
