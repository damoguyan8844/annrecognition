INCLUDE = -I..
CFLAGS = -O $(INCLUDE)
RANLIB = ranlib

FRAMER_OBJS = inform.o crises.o streams.o patches.o frames.o allocate.o archive.o files.o image.o notwork.o grounds.o printers.o parsers.o
FRAXL_OBJS = prims.o numeric.o eval.o evaldefs.o arlotje.o match.o network.o describe.o forms.o

all: libframer.a libfraxl.a

libframer.a: $(FRAMER_OBJS)
	ar rv $@ $?
	$(RANLIB) $@

libfraxl.a: $(FRAXL_OBJS)
	ar rv $@ $?
	$(RANLIB) $@
