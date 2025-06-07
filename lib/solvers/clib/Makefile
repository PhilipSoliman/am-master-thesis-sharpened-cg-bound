LIBTARGETS=custom_cg.so
LIBOBJS=$(LIBTARGETS:=.o)

# flags
CFLAGS=-I. -Wall -fPIC #$(PYTHONCFLAGS)
LDFLAGS=-shared #$(PYTHONLDFLAGS)
MATHFLAG=-lm

# compiler
CC=gcc

lib: $(LIBTARGETS) 

# shared lib compilation
%.o : %.cpp
	$(CC) $(CFLAGS) -c -o $@ $< $(MATHFLAG)

%.so : %.o
	$(CC) $(CFLAGS)  $< -o $@ $(LDFLAGS) $(MATHFLAG)

$(LIBOBJS) : Makefile

clean:
	rm -f core *~ *.o $(LIBTARGETS) vgcore.*
