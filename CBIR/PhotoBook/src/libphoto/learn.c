#include "photobook.h"
#include <tpm/gtree.h>
#include <tpm/stream.h>
#include <assert.h>
#include <math.h>

/* Globals *******************************************************************/

#define REDUCE 1
#define HURT_SINGLETONS 0
#define SIMPLE_BIAS 1

typedef struct NodeDataStruct {
  int *pos, *neg;
  double value;
  int high, low;
  float *score, *bias;
} NodeData;

typedef struct LeafStruct {
  int index;
  Tree node;
} Leaf;

typedef struct PosExStruct {
  Ph_Member member;
  int mark;
} PosEx;

typedef struct CoverStruct {
  int tree, label;
  Tree node;
  float score;
} Cover;

static char Optimistic = 0;
static int numLabels = 0, numTrees, numLeaves, maxLabels;
static int curTree, *numNodes;
static List *posEx; /* List of PosEx for each label */
static List *covers; /* List of Cover for each label */
static Leaf ***leaves; /* array of Leaf for each member in each tree */
static NodeData **nodes; /* array of NodeData for each tree */
static Tree *Hierarchies;
static char *Changed;
static int **memIndex; /* member index for each leaf of each tree */
static float errprob, **prob_matrix; /* label probability for each member */
static float nu;
static int numBiases, maxBiases = 2+112+56;
static int *labelBias;
static int *treeEnable;
int UpdateGlobalBias(void);

#define NodeBiasN(n, i) ((NodeData*)(n)->data)->bias[i]
#define NodeBias(n, label) ((NodeData*)(n)->data)->bias[labelBias[label]]
#define NodeScore(n, label) ((NodeData*)(n)->data)->score[label]
#define NodePos(n, label) ((NodeData*)(n)->data)->pos[label]
#define NodeNeg(n, label) ((NodeData*)(n)->data)->neg[label]
#define NodeLow(n) ((NodeData*)(n)->data)->low
#define NodeHigh(n) ((NodeData*)(n)->data)->high
#define NodeTotal(n) (NodeHigh(n) - NodeLow(n) + 1)
#define CoverPos(c) NodePos((c)->node, (c)->label)
#define CoverLow(c) NodeLow((c)->node)
#define CoverHigh(c) NodeHigh((c)->node)
#define CoverTotal(c) NodeTotal((c)->node)

#define IsCovered(i, cover) \
    ((leaves[cover->tree][i]->index >= CoverLow(cover)) && \
     (leaves[cover->tree][i]->index <= CoverHigh(cover)))
#define ExCovered(ex, cover) IsCovered(Ph_MemIndex(ex->member), cover)

#define SCORE_UNKNOWN -1
#define BIAS_UNKNOWN 1e-5

/* Prototypes ****************************************************************/

char *Ph_MemLabels(Ph_Member member, char *result);
char *Ph_MemWithLabel(int label, char *flags);
void LearnInit(Ph_Handle phandle, List trees, int labels);
void LearnFree(void);
void LearnPosEx(Ph_Member member, int label);
void LearnNegEx(Ph_Member member, int label);
void LearnUpdate(void);
void AddLabel(void);
void LearnOptimistic(char flag);
float *Ph_MemLabelProb(Ph_Member member);
void LearnErrProb(float prob);

void LearnReadBias(char *fname);
void LearnSaveBias(char *fname);
int LearnEnabled(int tree);
void LearnEnable(int tree, int flag);
int LearnNumCovers(int label);

/* private */
static void CollectCovers(int label);

/* Functions *****************************************************************/

void LearnOptimistic(char flag)
{
  Optimistic = flag;
}

int LearnEnabled(int tree)
{
  if((tree < 0) || (tree >= numTrees)) return 0;
  return treeEnable[tree];
}

void LearnEnable(int tree, int flag)
{
  int i;
  if((tree < 0) || (tree >= numTrees)) return;
  treeEnable[tree] = flag;
  /* Force everything to be updated */
  for(i=0;i<numLabels;i++) Changed[i] = 1;
}

void LearnNu(float f)
{
  nu = f;
}

static void NormalizeBias(int bias)
{
  int tree, i;
  float sum = 0.0;
  for(tree=0;tree<numTrees;tree++) {
    for(i=0;i<numNodes[tree];i++) {
      sum += nodes[tree][i].bias[bias];
    }
  }
  for(tree=0;tree<numTrees;tree++) {
    for(i=0;i<numNodes[tree];i++) {
      nodes[tree][i].bias[bias] /= sum;
    }
  }
}

static void ReadBias(FILE *fp, int tree, int bias)
{
  NodeData *node;
  float val;
  int n;
  for(;;) {
    if(fscanf(fp, "%d %f", &n, &val) < 2) break;
    node = &nodes[tree][n];
    node->bias[bias] = val;
  }
}

/*
 * File format is:
 * *
 * : tree
 * node1 value
 * node2 value
 * nodes...
 * : tree
 * another tree...
 * *
 * another bias...
 */
void LearnReadBias(char *fname)
{
  char ch;
  int tree, label;
  FILE *bias_fp = fopen(fname, "r");
  if(!bias_fp) {
    fprintf(stderr, "Cannot open `%s'\n", fname);
    perror("");
    return;
  }
  for(;;) {
    while(fgetc(bias_fp) != '*') if(feof(bias_fp)) break;
    if(feof(bias_fp)) break;
    for(;;) {
      for(;;) {
	ch = fgetc(bias_fp);
	if(ch == EOF) break;
	if(ch == ':') break;
	if(ch == '*') break;
      }
      if(ch != ':') { ungetc(ch, bias_fp); break; }
      if(fscanf(bias_fp, "%d", &tree) < 1) break;
      if(tree >= numTrees) {
	fprintf(stderr, "Input bias contains too many trees; ignoring tree %d\n",
		tree);
	continue;
      }
      ReadBias(bias_fp, tree, numBiases);
    }
    NormalizeBias(numBiases);
    numBiases++;
    if(numBiases == maxBiases) {
      fprintf(stderr, "bias limit reached.\n", maxBiases);
      break;
    }
  }
  fprintf(stderr, "%d biases total\n", numBiases);
  fclose(bias_fp);
}

/* used by LearnSaveBias */
static void WriteBias(FILE *fp, int tree, int label)
{
  int i,n;
  float value;
  
  for(n=0;n<numNodes[tree];n++) {
    NodeData *node = &nodes[tree][n];
#if SIMPLE_BIAS
    /* output the max for this node over all labels */
    value = 0;
    for(i=0;i<maxLabels;i++) {
      if(node->score[i] == SCORE_UNKNOWN) continue;
      if(node->score[i]*node->bias[i] > value) 
	value = node->score[i]*node->bias[labelBias[i]];
    }
#else
    if(node->score[label] == SCORE_UNKNOWN) continue;
    value = node->score[label] * node->bias[labelBias[label]];
#endif
    if(value > 1e-40) fprintf(fp, "%d %g\n", n, value);
  }
}

void LearnSaveBias(char *fname)
{
  int label, tree;
  FILE *bias_fp = fopen(fname, "w");
  for(label=0;label<numLabels;label++) {
    fprintf(bias_fp, "*\n");
    for(tree=0;tree<numTrees;tree++) {
      fprintf(bias_fp, ": %d\n", tree);
      WriteBias(bias_fp, tree, label); 
    }
#if SIMPLE_BIAS
    break;
#endif
  }
  fclose(bias_fp);
}

void LearnErrProb(float prob)
{
  errprob = prob;
}

static void ComputeLabelProb(int label)
{
#define mat(i,j,k) mat[(i)*numLeaves*numPos + (j)*numPos + (k)]
  float *mat;
  int numPos;
  PosEx *ex;
  int i,j, pos;
  Cover *cover;
  float prob, total;

  /* Make a matrix of pos/tree/member */
  numPos = ListSize(posEx[label]);
  /* tree index 0 will hold the total for all trees */
  mat = Allocate((numTrees+1) * numLeaves * numPos, float);

  /* Fill in the matrix */
  pos = 0;
  {ListIter(p, ex, posEx[label]) {
    /* clear the member probabilities for this pos */
    for(i=0;i<numTrees+1;i++) {
      for(j=0;j<numLeaves;j++) {
	mat(i,j,pos) = 0.0;
      }
    }
    total = 0.0;
    /* loop all trees */
    for(i=0;i<numTrees;i++) {
      Tree node = leaves[i][Ph_MemIndex(ex->member)]->node;
      /* walk up the tree, scoring the nodes */
      for(;node;node=node->parent) {
	prob = NodeScore(node, label)*NodeBias(node, label);
	total += prob;
	/* loop the leaves under this node */
	for(j=NodeLow(node);j<=NodeHigh(node);j++) {
	  mat(0, memIndex[i][j], pos) += prob;
	  mat(i+1, memIndex[i][j], pos) += prob;
	}
      }
    }

    /* normalize the results */
    for(i=0;i<numTrees+1;i++) {
      for(j=0;j<numLeaves;j++) {
	mat(i,j,pos) /= total;
      }
    }
    pos++;
  }}

  /* sort and tally the probabilities for each posEx */
  for(i=0;i<1;i++) {
    for(j=0;j<numLeaves;j++) {
      /* sort ascending; smallest is first */
      qsort(&mat(i,j,0), numPos, sizeof(float), (CmpFunc*)FloatCmp);
      total = 0.0;
      for(pos=0;pos<numPos;pos++) {
	total = total * errprob + mat(i,j,pos);
      }
      total *= (1-errprob);
      prob_matrix[label][j] = total;
    }
  }
  free(mat);
}

/* Returns a vector of probabilities; one prob for each label */
float *Ph_MemLabelProb(Ph_Member member)
{
  int i;
  float *result = Allocate(numLabels, float);
  for(i=0;i<numLabels;i++) {
    result[i] = prob_matrix[i][Ph_MemIndex(member)];
  }
  return result;
}

int LearnNumCovers(int label)
{
  return ListSize(covers[label]);
}

void ShowStats(int label)
{
  int pos = 0, children = 0, ncovers;
  Cover *cover;
  ListIter(p, cover, covers[label]) {
    pos += CoverPos(cover);
    children += CoverTotal(cover);
  }
  ncovers = ListSize(covers[label]);
  printf("%d %d %d\n", ncovers, pos, children);
}

/* Returns an array of label tree indices for member.
 * If result != NULL, puts the array in result.
 * Otherwise, caller must free the returned array.
 * If the member is not covered by a particular label, the value will
 * be zero at that position in the array.
 * Otherwise, the value will be 1+the tree which produced the cover.
 */
char *Ph_MemLabels(Ph_Member member, char *result)
{
  Cover *cover;
  int i;
  if(!result) result = Allocate(numLabels, char);
  for(i=0;i<numLabels;i++) {
    result[i] = 0;
    if(Ph_MemIndex(member) < numLeaves) {
      ListIter(p, cover, covers[i]) {
	/* is it a fake cover? */
	if(!cover->node) {
	  /* tree holds the covered member */
	  if(cover->tree == Ph_MemIndex(member)) {
	    result[i] = -1;
	    break;
	  }
	  continue;
	}
	if(IsCovered(Ph_MemIndex(member), cover)) {
	  result[i] = cover->tree+1;
	  break;
	}
      }
    }
  }
  return result;
}

char *Ph_MemWithLabel(int label, char *flags)
{
  Cover *cover;
  int i;
  if(flags == NULL) flags = Allocate(numLeaves, char);
  for(i=0;i<numLeaves;i++) flags[i] = 0;
  {ListIter(p, cover, covers[label]) {
    for(i=CoverLow(cover);i<=CoverHigh(cover);i++) {
      flags[memIndex[cover->tree][i]] = cover->tree+1;
    }
  }}
  return flags;
}

static void *NodeDataCreate(FileHandle fp)
{
  NodeData *data;
  int i;

  /* expand the nodes array */
  numNodes[curTree]++;
  nodes[curTree] = (NodeData*)realloc(nodes[curTree], 
				      numNodes[curTree]*sizeof(NodeData));
  data = &nodes[curTree][numNodes[curTree]-1];

  data->pos = (int*)calloc(maxLabels, sizeof(int));
  data->neg = (int*)calloc(maxLabels, sizeof(int));
  data->score = Allocate(maxLabels, float);
  for(i=0;i<maxLabels;i++) data->score[i] = SCORE_UNKNOWN;
  data->bias = Allocate(maxBiases, float);
  for(i=0;i<maxBiases;i++) data->bias[i] = BIAS_UNKNOWN;
  fscanf(fp, "%lf", &data->value);
  return data;
}

static void NodeDataFree(NodeData *p)
{
  free(p->bias);
  free(p->score);
  free(p->pos);
  free(p->neg);
  /* don't free p itself; it is part of the nodes array */
}

/* Makes the data fields of tree nodes point into the nodes array */
static void PatchPointers(int tree, Tree t, int *i)
{
  {IterateChildren(t, child) {
    PatchPointers(tree, child, i);
  }}
  t->data = &nodes[tree][(*i)++];
}

static void LoopLeaves(int tree, Tree t, Leaf **leafs, int *i)
{
  if(TreeLeaf(t)) {
    NodeData *node = t->data;
    int mem_index = (int)node->value-1;
    NodeHigh(t) = NodeLow(t) =
      leafs[mem_index]->index = *i;
    leafs[mem_index]->node = t;
    memIndex[tree][*i] = mem_index;
    (*i)++;
    return;
  }
  {IterateChildren(t, child) {
    LoopLeaves(tree, child, leafs, i);
  }}
}

static void ComputeHighLow(Tree tree)
{
  if(!TreeLeaf(tree)) {
    Tree child = tree->child;
    ComputeHighLow(child);
    NodeLow(tree) = NodeLow(child);
    if(child->sibling) {
      do {
	child = child->sibling;
	ComputeHighLow(child);
      } while(child->sibling);
    }
    NodeHigh(tree) = NodeHigh(child);
  }
}

static void InitialBias(Tree tree)
{
  int i;
  if(Optimistic) {
    /* slight bias toward bigger nodes */
    NodeBiasN(tree, 0) = 1 + 0.01*(float)(NodeTotal(tree)-1)/numLeaves;
  }
  else if(HURT_SINGLETONS && (NodeTotal(tree) == 1)) {
    /* node with a single child are disfavored */
    NodeBiasN(tree, 0) = 0.9;
  }
  else {
    /* make a slight bias toward smaller nodes */
    NodeBiasN(tree, 0) = 1 - 0.01*(float)(NodeTotal(tree)-1)/numLeaves;
  }
  {IterateChildren(tree, child) {
    InitialBias(child);
  }}
}

void LearnInit(Ph_Handle phandle, List trees, int labels)
{
  char *tree_name;
  int i;
  FileHandle fp;

  if(debug) fprintf(stderr, "loading trees\n");
  numTrees = ListSize(trees);
  Hierarchies = Allocate(numTrees, Tree);
  numLeaves = Ph_NumMembers(phandle);
  leaves = Allocate(numTrees, Leaf**);
  memIndex = Allocate(numTrees, int*);
  maxLabels = labels;
  nodes = Allocate(numTrees, NodeData*);
  numNodes = Allocate(numTrees, int);
  treeEnable = Allocate(numTrees, int);
  for(i=0;i<numTrees;i++) {
    /* dummy allocation */
    nodes[i] = Allocate(1, NodeData);
    numNodes[i] = 0;
    treeEnable[i] = 1;
  }
  numBiases = 1;
  curTree = 0;
  {ListIter(p, tree_name, trees) {
    /* read in the tree */
    if(debug) fprintf(stderr, "  %s\n", tree_name);
    fp = fopen(tree_name, "r");
    if(!fp) {
      char str[100];
      sprintf(str, "%s/%s/%s", phandle->data_dir, phandle->db_name,
	      tree_name);
      fp = FileOpen(str, "r");
    }
    Hierarchies[curTree] = TreeRead(fp, NodeDataCreate);
    FileClose(fp);
    i = 0;
    PatchPointers(curTree, Hierarchies[curTree], &i);

    leaves[curTree] = Allocate(numLeaves, Leaf*);
    for(i=0;i<numLeaves;i++) {
      leaves[curTree][i] = Allocate(1, Leaf);
    }
    memIndex[curTree] = Allocate(numLeaves, int);
    i = 0;
    LoopLeaves(curTree, Hierarchies[curTree], leaves[curTree], &i);
#if REDUCE
    /* the leaves array beyond numLeaves will be lost (not even freed) */
    if(i < numLeaves) {
      fprintf(stderr, "reducing numLeaves to %d\n", numLeaves = i);
    }
#endif
    ComputeHighLow(Hierarchies[curTree]);
    InitialBias(Hierarchies[curTree]);
    curTree++;
  }}
  NormalizeBias(0);

  /* dummy allocations */
  numLabels = 0;
  covers = Allocate(1, List);
  posEx = Allocate(1, List);
  Changed = Allocate(1, char);
  prob_matrix = Allocate(1, float*);
  prob_matrix[0] = Allocate(1, float);
  errprob = 0.1;
  labelBias = Allocate(1, int);
}

void LearnFree(void)
{
  int i,j;

  MEM_VERIFY_ALL();
  for(i=0;i<numTrees;i++) {
    TreeFree(Hierarchies[i], (FreeFunc*)NodeDataFree);
  }
  free(Hierarchies);

  for(i=0;i<numLabels;i++) {
    ListFree(covers[i]);
    ListFree(posEx[i]);
  }
  free(covers);
  free(posEx);

  for(i=0;i<numTrees;i++) {
    for(j=0;j<numLeaves;j++) {
      free(leaves[i][j]);
    }
    free(leaves[i]);
    free(memIndex[i]);
    free(nodes[i]);
  }
  free(leaves);
  free(nodes);
  free(numNodes);
  free(memIndex);
  free(Changed);
  free(prob_matrix[0]);
  free(prob_matrix);
  free(labelBias);
  free(treeEnable);
}

static void PosExFree(PosEx *ex)
{
  free(ex);
}

static void CoverFree(Cover *cover)
{
  free(cover);
}

void AddLabel(void)
{
  int i;
  if(numLabels == maxLabels) {
    fprintf(stderr, 
	    "Reached %d label maximum. Sorry, no more labels.\n", maxLabels);
    return;
  }
  numLabels++;
  covers = (List*)realloc(covers, numLabels*sizeof(List));
  /* frees the covers */
  covers[numLabels-1] = ListCreate((FreeFunc*)CoverFree);
  posEx = (List*)realloc(posEx, numLabels*sizeof(List));
  posEx[numLabels-1] = ListCreate((FreeFunc*)PosExFree);
  Changed = (char*)realloc(Changed, numLabels*sizeof(char));
  for(i=0;i<numLabels;i++) {
    Changed[i] = 0;
  }
  /* realloc and clear the prob_matrix */
  prob_matrix = (float**)realloc(prob_matrix, numLabels*sizeof(float*));
  prob_matrix[0] = (float*)realloc(prob_matrix[0], 
			   numLabels * numLeaves * sizeof(float));
  for(i=0;i<numLabels;i++) {
    int j;
    prob_matrix[i] = &prob_matrix[0][i*numLeaves];
    for(j=0;j<numLeaves;j++)
      prob_matrix[i][j] = 0.0;
  }
  labelBias = (int*)realloc(labelBias, numLabels*sizeof(int));
  labelBias[numLabels-1] = 0;
}

static Cover *NewCover(int tree, int label, Tree node)
{
  Cover *cover = Allocate(1, struct CoverStruct);

  cover->node = node;
  cover->tree = tree;
  cover->label = label;
  return cover;
}

void LearnNegEx(Ph_Member member, int label)
{
  int tree;
  Tree node;

  /* loop trees */
  for(tree = 0;tree < numTrees;tree++) {
    /* update the node scores */
    node = leaves[tree][Ph_MemIndex(member)]->node;
    /* check for contradiction; pos example becoming neg */
    if(NodePos(node, label)) {
      Tree t = node;
      /* rescore the tree */
      while(t) {
	NodePos(t, label)--;
	t = t->parent;
      }
      /* remove from the pos list */
      {PosEx *ex;ListIter(p, ex, posEx[label]) {
	if(ex->member == member) {
	  ListRemoveValue(posEx[label], ex, NULL);
	  break;
	}
      }}
    }
    while(node) {
      NodeNeg(node, label)++;
      NodeScore(node, label) = 0.0; /* neg nodes lose */
      node = node->parent;
    }
  }

  Changed[label] = 1;
}

void LearnPosEx(Ph_Member member, int label)
{
  PosEx *ex = Allocate(1, struct PosExStruct);
  int tree;
  Tree node;

  ex->member = member;
  ListAddFront(posEx[label], ex);

  /* loop trees */
  for(tree = 0;tree < numTrees;tree++) {
    /* update the node scores */
    node = leaves[tree][Ph_MemIndex(member)]->node;
    /* check for contradiction; neg example becoming pos */
    if(NodeNeg(node, label)) {
      Tree t = node;
      /* rescore the tree */
      while(t) {
	NodeNeg(t, label)--;
	/* what happens to NodeScore? */
	t = t->parent;
      }
    }
    while(node) {
      NodePos(node, label)++;
      /* Since this overwrites the score, we want to make sure that
       * there is no negative influence here.
       */
      if(!NodeNeg(node, label)) {
	NodeScore(node, label) = NodePos(node, label)+1;
      }
      node = node->parent;
    }
  }

  Changed[label] = 1;
}

static int CoverCmp(Cover *c1, Cover *c2)
{
  return c1->node != c2->node;
}

static float ScoreBias(int tree, int label, int bias)
{
  int n;
  float score = 0.0;

  for(n=0;n<numNodes[tree];n++) {
    NodeData *node = &nodes[tree][n];
    if(node->score[label] == SCORE_UNKNOWN) continue;
    score += node->score[label] * node->bias[bias];
  }
  return score;
}

float UpdateBias(int label, int *bias_return)
{
  int bias, tree;
  float score, maximum;

  *bias_return = 0;
  maximum = 0;
  if(numBiases == 1) return 1.0;
  /* finds the best bias over all trees for the given label */
  for(bias = 0;bias < numBiases;bias++) {
    score = 0.0;
    for(tree=0;tree < numTrees;tree++) {
      score += ScoreBias(tree, label, bias);
    }
#if 0
    fprintf(stderr, "  label %d bias %d has score %g\n",
	    label, bias, score);
#endif
    if(score > maximum) { maximum = score; *bias_return = bias; }
  }
  fprintf(stderr, "label %d has bias %d with score %g\n", 
	  label, *bias_return, maximum);
  labelBias[label] = *bias_return;
  return maximum;
}

int UpdateGlobalBias(void)
{
  int best, tree, label;
  float maximum;

  maximum = UpdateBias(0, &best);
#if 1
  for(label=1;label<numLabels;label++) {
    int bias;
    float score = UpdateBias(label, &bias);
    if(score > maximum) { maximum = score; best = bias; }
  }
#endif
  fprintf(stderr, "best bias %d has score %g\n", best, maximum);
  return best;
}

static void CollectCovers(int label)
{
  PosEx *ex;
  Cover *best_cover;
  ListNode q;
  int tree, index;
  Tree node;

  ListFree(covers[label]);
  covers[label] = ListCreate((FreeFunc*)CoverFree);
  
  if(debug) printf("label %d:\n", label);

  /* unmark positive examples */
  {ListIter(p, ex, posEx[label]) {
    ex->mark = 0;
  }}

  /* loop positive examples */
  {ListIter(p, ex, posEx[label]) {
    if(ex->mark) continue;
    index = Ph_MemIndex(ex->member);

    /* find best cover for the member */
    best_cover = NewCover(0, label, NULL);
    best_cover->score = 0;
    node = (Tree)1; /* to detect if all trees are disabled */
    /* loop trees */
    for(tree=0;tree<numTrees;tree++) {
      if(!treeEnable[tree]) continue;
#if 0
      printf("  tree: %d\n", tree);
#endif 
      /* start at leaf node and work up */
      for(node = leaves[tree][index]->node;
	  node; node=node->parent) {
	/* score this node */
	float score;
	/* if there is an unknown score here, something is wrong. */
	assert(NodeScore(node, label) != SCORE_UNKNOWN);
	score = NodeScore(node, label)*NodeBias(node, label);
#if 0
	printf("    %g\n", score);
#endif
	if(score == 0.0) break;
	/* is it better than best so far? */
	if(best_cover->score < score) {
	  best_cover->tree = tree;
	  best_cover->node = node;
	  best_cover->score = score;
	}
      }
    }
    if(node == (Tree)1) {
      /* all trees are disabled. just label the example itself. */
      best_cover->tree = index;
      ListAddFront(covers[label], best_cover);
      continue;
    }
    if(best_cover->score == 0) {
      fprintf(stderr, "label %d for %d was disallowed by bias %d\n",
	      label, index, labelBias[label]);
      if(node) {
	fprintf(stderr, "  score: %g\tbias: %g\n", 
		NodeScore(node, label), NodeBias(node, label));
      }
      CoverFree(best_cover);
      continue;
    }
    if(debug) printf("cover %d: %d %d pos %d score %f bias %f\n", 
		     best_cover->tree, 
		     CoverLow(best_cover), CoverHigh(best_cover), 
		     CoverPos(best_cover), best_cover->score,
		     NodeBias(best_cover->node, label)
		     );
    ListAddFront(covers[label], best_cover);

    /* mark all pos ex under the cover */
    for(q=p;q;q=q->next) {
      ex = (PosEx*)q->data;
      if(ExCovered(ex, best_cover)) {
	ex->mark = 1;
      }
    }
  }}
}

void LearnUpdate(void)
{
  int i;
  for(i=0;i<numLabels;i++) {
    if(Changed[i]) {
      CollectCovers(i);
/*
      ComputeLabelProb(i);
*/
      Changed[i] = 0;
    }
  }
}
