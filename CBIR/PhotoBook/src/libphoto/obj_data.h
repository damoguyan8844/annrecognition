/* Data structures for each metric */

/* useful macro for metric distance functions */
#define PhGetField(obj, f, v) \
  if(Ph_ObjGet(obj, f, &v) == PH_ERROR) {                 \
    fprintf(stderr, "Error getting %s field for %s\n",    \
	    f, Ph_ObjName(obj));                          \
    return;                                               \
  }


struct EuclideanData {
  int vector_size, from, to;
  double *weights;
  char *field;
/* private */
  Type type;
};

struct MahalData {
  int vector_size;
  char *coeff_field, *icovar_field;
  char *mask;
};

struct DiverData {
  char *covar_field;
};

#include <tpm/gtree.h>
struct HierData {
  char *tree_file;
  Tree tree;
};

struct WoldData {
  int num_peaks, nbr_size;
  char *peaks;
  char *alt_metric_name, *orien_type, *orien_label, *tamura_vector;
/* private */
  Matrix nbr;
  Ph_Object alt_metric;
  PhDistFunc *alt_distance;
};

struct PeaksData {
  int num_peaks, nbr_size;
  char *peaks;
/* private */
  Matrix nbr;
};

struct RankComboData {
  int num_metrics;
  char **metric_names;
  char *weights;
/* private */
  Ph_Object *metrics;
  PhDistFunc **distfuncs;
};

struct CombinationData {
  int num_metrics;
  char **metric_names;
  double *factors, *weights;
/* private */
  Ph_Object *metrics;
  PhDistFunc **distfuncs;
};

struct VSpaceData {
  int rows, cols;
  char *corr_field, *basis_field;
};

struct MinData {
  char *field;
};

struct TswData {
  char *field;
  float cutoff;
  int keep, levels;
};

struct ViewData {
  int height, width, channels;
};

struct ViewImageData {
  char *field;
};

struct ViewBarData {
  char *vector_field;
  int spacing;
  double maximum, minimum;
  char color[3];
};

struct ViewStretchData {
  char *field;
};

struct ViewLabelProbData {
  int label;
};

struct ViewChannelData {
  char *field;
  int channel;
};

struct ViewZoomData {
  char *field;
  int zfact;
};

struct ViewTswData {
  char *field;
  double maximum;
};

struct ViewPeaksData {
  char *peaks;
};
