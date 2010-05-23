INCLUDE = -I.. -I../libphoto
#CFLAGS = -g -DMEMCHECK $(INCLUDE)
CFLAGS = -O $(INCLUDE)
RANLIB = ranlib

UTILS = cache.o config_shell.o tshell.o widgets.o
SUBS = labeling.o symbols.o load.o pix_text.o
MAIN = main.o setup.o left_pane.o right_pane.o config.o
OBJS = $(MAIN) $(SUBS) $(UTILS)
MYLIBS = ../libphoto/libphoto.a ../xtpm/libxtpm.a ../type/libtype.a ../tpm/libtpm.a ../framer/libframer.a ../framer/libfraxl.a 
LIBS = $(MYLIBS) -L/usr/X11/lib -lXmu -lXm -lXt -lX11 -lm

photobook: $(OBJS) $(MYLIBS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LIBS)
