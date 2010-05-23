CFLAGS = -O -I..

OBJS = basic.o math.o lapack.o

libmatrix.a: $(OBJS)
	ar rv $@ $?
test: test.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ test.c $(OBJS) -L/v/lib/$(MACHTYPE) -llapack -lblas -lf2c -lm
