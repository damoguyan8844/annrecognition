#include "photobook.h"
#include "obj_data.h"

#include "metric/class_euclidean.h"
#include "metric/class_mahalanobis.h"
#include "metric/class_divergence.h"
#include "metric/class_hier.h"
#include "metric/class_wold.h"
#include "metric/class_combination.h"
#include "metric/class_vspace.h"
#include "metric/class_min.h"
#include "metric/class_tsw.h"
#include "metric/class_peaks.h"
#include "metric/class_rank_combo.h"

#include "view/class_view.h"
#include "view/class_image.h"
#include "view/class_bar.h"
#include "view/class_stretch.h"
#include "view/class_labelprob.h"
#include "view/class_channel.h"
#include "view/class_zoom.h"
#include "view/class_tsw_tree.h"
#include "view/class_view_peaks.h"

/* Globals *******************************************************************/

List MetricClasses, ViewClasses, GlobalClasses; /* List of ObjClass */

/* Prototypes ****************************************************************/

List Ph_GetMetrics(void);
List Ph_GetViews(void);
void PhClassInit(void);
Ph_Object PhLookupObject(Ph_Handle phandle, char *name);

/* Functions *****************************************************************/

void PhClassInit(void)
{
  ListAddRear(MetricClasses, &Euclidean);
  ListAddRear(MetricClasses, &Mahalanobis);
  ListAddRear(MetricClasses, &Divergence);
  ListAddRear(MetricClasses, &Hierarchical);
  ListAddRear(MetricClasses, &Wold);
  ListAddRear(MetricClasses, &Combination);
  ListAddRear(MetricClasses, &VSpace);
  ListAddRear(MetricClasses, &MinClass);
  ListAddRear(MetricClasses, &Tsw);
  ListAddRear(MetricClasses, &Peaks);
  ListAddRear(MetricClasses, &RankCombo);

  /* "View" view is omitted since it cannot be used (abstract base class) */
  ListAddRear(ViewClasses, &ViewImage);
  ListAddRear(ViewClasses, &ViewBar);
  ListAddRear(ViewClasses, &ViewStretch);
  ListAddRear(ViewClasses, &ViewLabelProb);
  ListAddRear(ViewClasses, &ViewChannel);
  ListAddRear(ViewClasses, &ViewZoom);
  ListAddRear(ViewClasses, &ViewTsw);
  ListAddRear(ViewClasses, &ViewPeaks);
}

static List ClassNames(List classes)
{
  List names = ListCreate(NULL);
  ObjClass class;
  ListIter(p, class, classes) {
    ListAddRear(names, class->name);
  }
  return names;
}

List Ph_GetMetrics(void)
{
  return ClassNames(MetricClasses);
}

List Ph_GetViews(void)
{
  return ClassNames(ViewClasses);
}

Ph_Object PhLookupObject(Ph_Handle phandle, char *name)
{
  ObjClass class;
  Ph_Object obj;
  int i;
  for(i=0;name[i];i++) {
    if(name[i] == '/') {
      /* try all namespaces */
      if(!strncmp(name, "metric", i)) {
	return PhLoadMetric(phandle, &name[i+1]);
      }
      if(!strncmp(name, "view", i)) {
	return PhLoadView(phandle, &name[i+1]);
      }
      break;
    }
  }
  /* use the global namespace */
  class = PhClassFind(GlobalClasses, name);
  if(!class) return NULL;
  return Ph_ObjCreate(phandle, class, name);
}
