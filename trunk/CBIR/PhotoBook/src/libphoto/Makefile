INCLUDE = -I.. -I.
CFLAGS = -O $(INCLUDE)
RANLIB = ranlib

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

METRICS = metric/euclidean.o metric/mahalanobis.o metric/divergence.o metric/hier.o metric/wold.o metric/combination.o metric/vspace.o metric/min.o metric/tsw.o metric/peaks.o metric/rank_combo.o
VIEWS = view/view.o view/image.o view/bar.o view/stretch.o view/labelprob.o view/channel.o view/zoom.o view/tsw_tree.o view/view_peaks.o
UTILS = utils.o ph_image.o matrix.o
MAIN = photobook.o load_field.o memann.o class_table.o object.o parse.o learn.o
OBJS = $(METRICS) $(VIEWS) $(UTILS) $(MAIN)

libphoto.a: $(OBJS)
	ar rv $@ $?
	$(RANLIB) $@
