#
#  Makefile for fltk applications
#

LOCAL = /usr/local
USR  = /usr/

CC = nvcc

INCLUDE = -I$(LOCAL)/include
LIBDIR = -L$(LOCAL)/lib
LIBS  = 

CFLAGS = -DDEBUG -arch sm_11 -g -G

.SUFFIXES: .o .cpp .cxx .cu

.o: 
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

.cpp.o: 
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

.cu.o:
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

ALL.O = rply.o geom.o util.o gpuNode.o gpuBuilder.o gpuTriangleList.o main.o

gpukd: $(ALL.O)
	$(CC) $(CFLAGS) -o $@ $(ALL.O) $(INCLUDE) $(LIBDIR) $(LIBS)

clean:
	rm -f $(ALL.O)

clean_all:
	rm -f $(ALL.O) ray

