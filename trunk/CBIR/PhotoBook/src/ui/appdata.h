/* Colors */
#define NUM_COLORS 4
enum { COLOR_PHOTOBG, COLOR_TEXT, COLOR_UNPICKED, COLOR_PICKED };

/* AppData definition */
typedef struct AppDataStruct {
  XtAppContext app_context;
  Display *display;
  Screen *screen;
  Window root;
  int depth;
  Visual *visual;
  Colormap colormap;
  GC gc;
  int *color_table, *gamma_table;
  XColor color[NUM_COLORS];
  double gamma;
  Font text_font;

  Widget shell, main_window, form, left_pane, right_pane;
  Widget view_menu, metric_menu;
  TShell view_shell, metric_shell;
  TShell symbol_shell, labeling_shell;
  TShell hooks_shell, glabel_shell;
  Widget count_label, page_label, com_text;
  Widget pop_shell, pop_label;
  Widget imt;
  XtpmIP imp;

  int cache_size;
  Pixmap no_image_pixmap;
  int num_rows, num_cols;
  int im_width, im_height, im_channels, pix_width, pix_height;
  int text_height, text_ascent;
  int text_lines, y_pad;
  Ph_Member *members;
  int num_members;
  char *selected;
  char *filter;

  List labels;
  List glabels;
  List colorList;
  List trees;

  /* pixmap text flags */
  char show_dist, mask_dist, show_name, show_label, show_tree;
  char *show_annotation;

  List garbage;
} *AppData;
