# Run like this:
#
# CILKPLUS_PREFIX=<path to CILK install> \
# OPENCV_PREFIX=<path to OpenCV install> make
#
CFLAGS= -O3
CILK=$(CILKPLUS_PREFIX)/bin/g++
CILKFLAGS= -w -g -O3 -m64 -fcilkplus -std=c++11
#
# Include locations
#
CILKFLAGS+=-I$(HDF5_PREFIX)/include
CILKFLAGS+=-I$(OPENCV_PREFIX)/include
CILKFLAGS+=-D_GLIBCXX_USE_CXX11_ABI=1
CILKFLAGS+=-L$(OPENCV_PREFIX)/lib
#
# Library locations
#
LDFLAGS=
LDFLAGS+=-L$(CILKPLUS_PREFIX)/lib64
LDFLAGS+=-L$(OPENCV_PREFIX)/lib
#LDFLAGS+=`pkg-config --libs hdf5`
LDFLAGS+=-L$(HDF5_PREFIX)/lib
#LDFLAGS+=-L/usr/lib/gcc/x86_64-linux-gnu/4.9
#
# Libraries
#
LDFLAGS+= -lopencv_highgui -lopencv_imgproc -lopencv_core
LDFLAGS+=-lcilkrts
LDFLAGS+=-lstdc++ -gcc_s
LDFLAGS+=-lhdf5_cpp -lhdf5
#AR=ar

DEPS= cilk_tools/engine.h Graph.h cilk_tools/scheduler.h
OBJ= cilk_tools/engine.o Graph.o cilk_tools/scheduler.o


DEPS= cilk_tools/engine.h Graph.h cilk_tools/scheduler.h
OBJ= main.o

all: main LUT/LUT.txt

%.o: %.cpp $(DEPS)
	$(CILK) $(CILKFLAGS) -c -o $@ $<

main : $(OBJ) graph_extraction_init.h output_functions.h loadData.h create_layout.h vector3D.h thinning.h graph_properties.h Graph.h Graph.cpp cilk_tools/engine.cpp cilk_tools/engine.h cilk_tools/scheduler.cpp cilk_tools/scheduler.h graph_extraction_config.h Makefile
	LD_LIBRARY_PATH=$(CILKPLUS_PREFIX)/lib64 \
	$(CILK) $(LDFLAGS) $(OBJ) -o $@

LUT/LUT.txt:
	cd LUT &&\
	make lutgen &&\
	./lutgen

clean :
	rm -f main *~ *.o

lint :
	python ../cpplint.py main.cpp graph_properties.h

test :
	./run_test.sh
