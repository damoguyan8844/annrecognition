#include "photobook.h"

/* Functions ****************************************************************/

void main(int argc, char *argv[])
{
  Ph_Handle phandle;
  int j, k;
  FILE *fp, *fpd, *fpi;
  char *ws_file, *filter;
  Ph_Member class_image, test_image;
  Ph_Member *working_set;
  int ws_members;
  List classes;
  int threshold;
  char str[100];

  if(!strcmp(argv[argc-1], "-debug")) {
    debug = 1;
    argc--;
  }
  else 
    debug = 0;

  if(argc < 5) {
    printf("Usage:\n\n%s db_name metric threshold index [filter]\n\n",
	   argv[0]);
    printf("To compute the query results on the database <db_name>\n");
    printf("using <metric> to compute distances.\n");
    printf("Only the first <threshold> matches are listed.\n");
    printf("Frame names come from file named <index>.\n");
    printf("\nIndexes are written to the file `<metric>.results'.\n");
    printf("Distances are written to the file `<metric>.results.dist'.\n");
    exit(1);
  }

  phandle = Ph_Startup();
  if(Ph_SetDatabase(phandle, argv[1]) == PH_ERROR) {
    fprintf(stderr, "Unknown database `%s'\n", argv[1]);
    exit(1);
  }
  if(Ph_SetMetric(phandle, argv[2]) == NULL) {
    fprintf(stderr, "Unknown metric `%s'\n", argv[2]);
    exit(1);
  }

  /* Read threshold, index */
  threshold = atoi(argv[3]);
  ws_file = argv[4];

  if(argc > 5) filter = argv[5];
  else filter = "";

  /* Open output file */
  sprintf(str, "%s.results", argv[2]);
  fp = fopen(str, "w");
  sprintf(str, "%s.results.dist", argv[2]);
  fpd = fopen(str, "w");

  /* Filter the working_set */
  if(filter[0]) printf("filter: %s\n", filter);
  if(Ph_SetFilter(phandle, filter) == PH_ERROR) {
    fprintf(stderr, "Bad filter\n");
    exit(1);
  }

  if(Ph_LoadWS(phandle, ws_file) == PH_ERROR) {
    fprintf(stderr, "Cannot open `%s'", ws_file);
    perror("");
    exit(1);
  }
  classes = Ph_ListWorkingSet(phandle);
  printf("%d members\n", ListSize(classes));

  /* Loop class_image over all frames */
  {ListIter(p, class_image, classes) {
    List query;
    printf(": %s\n", Ph_MemName(class_image));
    
    query = ListCreate(NULL);
    ListAddRear(query, Ph_MemName(class_image));
    Ph_SetQuery(phandle, query);
    ListFree(query);
    working_set = Ph_GetWorkingSet(phandle, &ws_members);

    /* Loop up to threshold value */
    for(j=0;j < threshold;j++) {
      if(j >= ws_members) break;
      test_image = working_set[j];
      fprintf(fp, "%d ", Ph_MemIndex(test_image));
      fprintf(fpd, "%g ", Ph_MemDistance(test_image));
    }
    fprintf(fp, "\n");
    fflush(fp);
    fprintf(fpd, "\n");
    fflush(fpd);
  }}

  /* Close file */
  fclose(fp);
  fclose(fpd);
  exit(0);
}
