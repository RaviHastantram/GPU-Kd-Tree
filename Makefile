#
#  Makefile for fltk applications
#

LOCAL = /usr/local
USR  = /usr/
TACC = /opt/apps/cuda/3.2/cuda/lib/

CC = nvcc

INCLUDE = -I$(LOCAL)/include -I./thrust -I$(TACC_CUDA_INC)
LIBDIR = -L$(LOCAL)/lib
LIBS  = -L$(TACC_CUDA_LIB) -L$(TACC) -lcudart

CFLAGS = -arch compute_11 -g -G -m64

.SUFFIXES: .o .cpp .cxx .cu

.o: 
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

.cpp.o: 
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

.cu.o:
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

ALL.O = main.o geom.o util.o rply.o gpuBuilder.o gpuTriangleList.o gpuNode.o

gpukd: $(ALL.O)
	$(CC) $(CFLAGS) -o $@ $(ALL.O) $(INCLUDE) $(LIBDIR) $(LIBS)

clean:
	rm -f $(ALL.O)

clean_all:
	rm -f $(ALL.O) ray

