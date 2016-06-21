
# Haswell
#CC := /home/amatveev/Pipeline/cilkplus-install/bin/g++
#LD := /home/amatveev/Pipeline/cilkplus-install/bin/g++

# 72-core
CC := /home/armafire/tools/cilkplus-install/bin/g++
LD := /home/armafire/tools/cilkplus-install/bin/g++

# gcc-trunk
#CC := /mnt/disk1/armafire/tools/gcc-trunk-install/bin/g++
#LD := /mnt/disk1/armafire/tools/gcc-trunk-install/bin/g++

# Haswell
#OPENCV_INCLUDE := /home/amatveev/Pipeline/tools/OpenCV/opencv-2.4-install/include/
#OPENCV_LIB := /home/amatveev/Pipeline/tools/OpenCV/opencv-2.4-install/lib/

# 72-core CM1
#OPENCV_INCLUDE=/home/armafire/tools/opencv-install/include/
#OPENCV_LIB=/home/armafire/tools/opencv-install/lib/

# 72-core CM2
#OPENCV_INCLUDE=/home/armafire/tools/opencv-2.4.11-install/include
#OPENCV_LIB=/home/armafire/tools/opencv-2.4.11-install/lib
OPENCV_INCLUDE=/home/armafire/tools/opencv-3-install-test/include/
OPENCV_LIB=/home/armafire/tools/opencv-3-install-test/lib/

#CFLAGS += -fcilkplus -m64 -ffast-math -mfma 
CFLAGS += -std=c++11 -fcilkplus -m64 -march=native -ffast-math -mfma -funroll-loops -flto

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


LDFLAGS += -L$(OPENCV_LIB) -lcilkrts -lopencv_imgcodecs -lopencv_core -lopencv_imgproc 
#LDFLAGS += -L$(OPENCV_LIB) -lcilkrts -lopencv_highgui -lopencv_core -lopencv_imgproc 

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
