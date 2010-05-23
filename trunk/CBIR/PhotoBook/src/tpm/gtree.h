/* Definitions for generic trees */
#ifndef GTREE_H_INCLUDED
#define GTREE_H_INCLUDED

#include <tpm/list.h>

/* This is intended to be a transparent data structure */
typedef struct TreeStruct {
  void *data;
  struct TreeStruct *child, *sibling, *parent;
} *Tree;

/* A typical use of this iterator is 
 * {IterateChildren(tree, child) {
 *   Print(child->data);
 * }}
 */
#define IterateChildren(t,c) Tree c;for(c=(t)->child;c;c=c->sibling)
/* TRUE if t is a leaf */
#define TreeLeaf(t) ((t)->child == NULL)

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

#endif
