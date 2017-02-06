CC := /home/armafire/tools/cilkplus-install/bin/g++
LD := /home/armafire/tools/cilkplus-install/bin/g++

OPENCV_INCLUDE=/home/armafire/tools/opencv-3-install-test/include
OPENCV_LIB=/home/armafire/tools/opencv-3-install-test/lib

CFLAGS += -std=c++11 -fcilkplus -m64 -march=native -ffast-math -mfma 
CFLAGS += -Wall 
CFLAGS += -I$(OPENCV_INCLUDE)

ifdef DEBUG
	CFLAGS += -O0 -g3
else
	CFLAGS += -DNDEBUG
	CFLAGS += -O3 
	#-Ofast
endif

LDFLAGS += -L$(OPENCV_LIB) -lcilkrts -lopencv_core -lopencv_imgcodecs -lboost_program_options -lboost_system -lboost_filesystem -lboost_regex

BINS = combine relabel

.PHONY:	all clean

all: $(BINS)

combine.o: combine.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

relabel.o: relabel.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

combine: combine.o
	$(LD) -o $@ $^ $(LDFLAGS)

relabel: relabel.o
	$(LD) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(BINS) *.o

