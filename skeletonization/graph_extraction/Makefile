CC=/home/armafire/tools/cilkplus-install/bin/gcc
CFLAGS= -O3
CILK=/home/armafire/tools/cilkplus-install/bin/g++
CILKFLAGS= -w -g -O3 -gdwarf-2 -fcilkplus -lhdf5_cpp -lhdf5  -lcilkrts -L/home/armafire/tools/cilkplus-install/lib64
LDFLAGS= -lopencv_highgui -lopencv_imgproc -lopencv_core -L/home/armafire/tools/opencv-install/lib
#AR=ar

DEPS= cilk_tools/engine.h Graph.h cilk_tools/scheduler.h
OBJ= cilk_tools/engine.o Graph.o cilk_tools/scheduler.o

OPENCV=/home/armafire/tools/opencv-install/include/opencv
all: main

%.o: %.cpp $(DEPS)
	$(CILK) $(CILKFLAGS) -c -o $@ $<

main : $(OBJ) main.cpp graph_extraction_init.h output_functions.h loadData.h create_layout.h vector3D.h thinning.h graph_properties.h Graph.h Graph.cpp cilk_tools/engine.cpp cilk_tools/engine.h cilk_tools/scheduler.cpp cilk_tools/scheduler.h graph_extraction_config.h Makefile
	export LD_LIBRARY_PATH=/home/armafire/tools/cilkplus-install/lib64
	$(CILK) $(CILKFLAGS) `pkg-config --cflags opencv` $@.cpp `pkg-config --libs opencv` -o $@ $< 

clean :
	rm -f main *~ *.o

lint :
	python ../cpplint.py main.cpp graph_properties.h

test :
	./run_test.sh
