#
# This makefile should be run via "make fc_dnn/src/run_dnn" from the root
# directory
#
OPENCV_INCLUDE=$(OPENCV_PREFIX)/include
OPENCV_LIB=$(OPENCV_PREFIX)/lib

GCC_LIB=/usr/lib/gcc/x86_64-linux-gnu/4.9

#CFLAGS += -fcilkplus -m64 -ffast-math -mfma 
CFLAGS += -std=c++11
CFLAGS += -fcilkplus -m64 -march=native -ffast-math -mfma -funroll-loops -flto

#CFLAGS += -D_REENTRANT
CFLAGS += -Wall -Winline
CFLAGS += -I$(OPENCV_INCLUDE)

#CFLAGS += --param inline-unit-growth=1000
#CFLAGS += -mrtm

ifdef DEBUG
	CFLAGS += -O0 -g3
else
	CFLAGS += -DNDEBUG
	CFLAGS += -O3 -Ofast
endif


LDFLAGS += -L$(GCC_LIB) -L$(OPENCV_LIB) -L$(CILKPLUS_PREFIX)/lib64 -lcilkrts 
#LDFLAGS +=-lopencv_imgcodecs
LDFLAGS +=-lopencv_core -lopencv_imgproc -lopencv_highgui

BINS = run_dnn

.PHONY:	all clean

all: $(BINS)

common.o: common.c
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

conv.o: conv.c
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

pool.o: pool.c
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

softmax.o: softmax.c
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

interweave.o: interweave.c
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

tensor_decomp.o: tensor_decomp.c
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

tensor_conv.o: tensor_conv.c
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

dnn.o: dnn.c
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

run.o: run.c
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<    	

run_dnn: common.o conv.o pool.o softmax.o interweave.o tensor_decomp.o tensor_conv.o dnn.o run.o
	$(LD) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(BINS) *.o
