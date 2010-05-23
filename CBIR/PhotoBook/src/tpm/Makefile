# need this to resolve <tpm/list.h>-style references
INCLUDE = -I..
#CFLAGS = -g -DMEMCHECK $(INCLUDE)
CFLAGS = -O $(INCLUDE)
RANLIB = ranlib
OBJS = binary.o gtree.o hash.o list.o tpm.o quark.o stream.o partition.o

libtpm.a: $(OBJS)
	ar rv $@ $?
	$(RANLIB) $@
