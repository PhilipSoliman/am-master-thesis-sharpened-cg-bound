# CPython module homework
LIBTARGETS= custom_cg.so
LIBOBJS=$(LIBTARGETS:=.o)
PYTHONCFLAGS=$(shell python3-config --cflags)
PYTHONLDFLAGS=$(shell python3-config --ldflags)

# flags
CFLAGS=-I. -Wall -fPIC #$(PYTHONCFLAGS)
LDFLAGS=-shared #$(PYTHONLDFLAGS)
# include explicit location of lapacke
LAPACKFLAGS=-lm -llapacke -llapack -lblas -ltmglib -I/../../mingw64/include -L/../../mingw64/lib

# compiler
CC=gcc

lib: $(LIBTARGETS) 

# shared lib compilation
%.o : %.cpp
	$(CC) $(CFLAGS) -c -o $@ $< $(LAPACKFLAGS)

%.so : %.o
	$(CC) $(CFLAGS)  $< -o $@ $(LDFLAGS) $(LAPACKFLAGS)

$(LIBOBJS) : Makefile

clean:
	rm -f core *~ *.o $(LIBTARGETS) vgcore.*
