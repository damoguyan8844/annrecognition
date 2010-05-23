INCLUDE = -I.. -I../libphoto
CFLAGS = -O $(INCLUDE)

FRLIBS = ../framer/libframer.a ../framer/libfraxl.a
LIBS = ../libphoto/libphoto.a ../type/libtype.a ../tpm/libtpm.a $(FRLIBS) -lm

all: annotate results convert_to_binary concat

annotate: annotate.c
	$(CC) $(CFLAGS) -o $@ $? $(FRLIBS)
results: results.c
	$(CC) $(CFLAGS) -o $@ $? $(LIBS)
convert_to_binary: convert_to_binary.c
	$(CC) $(CFLAGS) -o $@ $? ../tpm/libtpm.a
concat: concat.c
	$(CC) $(CFLAGS) -o $@ $? ../tpm/libtpm.a
