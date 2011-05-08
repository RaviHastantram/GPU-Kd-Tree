#
#  Makefile for fltk applications
#

LOCAL = /usr/local
USR  = /usr/

CC = nvcc

INCLUDE = -I$(LOCAL)/include
LIBDIR = -L$(LOCAL)/lib
LIBS  =

CFLAGS =  -m32 

.SUFFIXES: .o .cpp .cxx

.o: 
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

.cpp.o: 
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

ALL.O = main.o geom.o util.o rply.o

gpukd: $(ALL.O)
	$(CC) $(CFLAGS) -o $@ $(ALL.O) $(INCLUDE) $(LIBDIR) $(LIBS)

clean:
	rm -f $(ALL.O)

clean_all:
	rm -f $(ALL.O) ray

