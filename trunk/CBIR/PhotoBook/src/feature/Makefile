INCLUDE = -I.. -I../matrix
LIBS = -L../matrix
CFLAGS = -O $(INCLUDE) $(LIBS)

all: rgb2gray rgb2ohta histo

rgb2gray: rgb2gray.c
	$(CC) $(CFLAGS) -o $@ $? -lmatrix
rgb2ohta: rgb2ohta.c
	$(CC) $(CFLAGS) -o $@ $? -lmatrix
mr_sar: mr_sar.c
	$(CC) $(CFLAGS) -o $@ $? -lmatrix -llapack -lblas -lf2c -lm
histo: histo.c
	$(CC) $(CFLAGS) -o $@ $? -lmatrix -lm
