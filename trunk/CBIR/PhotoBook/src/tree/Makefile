INCLUDE = -I..
LIBS = -L../tpm
CFLAGS = -O $(INCLUDE) $(LIBS)

all: nnhierarchy simplify

nnhierarchy: nnhierarchy.c
	$(CC) $(CFLAGS) -o $@ $? -ltpm
simplify: simplify.c
	$(CC) $(CFLAGS) -o $@ $? -ltpm
