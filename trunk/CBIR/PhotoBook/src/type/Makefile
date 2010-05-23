INCLUDE = -I..
CFLAGS = -O $(INCLUDE)
RANLIB = ranlib

OBJS = type.o table.o quark.o parse.o parse_value.o tpm_parse.o tpm_types.o match.o

libtype.a: $(OBJS)
	ar rv $@ $?
	$(RANLIB) $@
