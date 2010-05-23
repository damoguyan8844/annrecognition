# need this to resolve <xtpm/xtpm.h> style references
INCLUDE = -I..
#CFLAGS = -g -DMEMCHECK $(INCLUDE)
CFLAGS = -O $(INCLUDE)
RANLIB = ranlib
OBJS = colors.o converters.o im_table.o multilist.o photo_table.o xtpm.o xcheck.o

libxtpm.a: $(OBJS)
	ar rv $@ $?
	$(RANLIB) $@
