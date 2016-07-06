#

#OPENCV_DIR=/home/victorj/libraries/opencv_install
OPENCV_DIR=/home/leek/tools/opencv-2.4/opencv-install

#VIGRA_DIR=/home/victorj/libraries/vigra_install
VIGRA_DIR=/home/leek/tools/vigra/vigra-install

SPARSEHASH_DIR=/home/leek/tools/sparsehash/sparsehash-install

JSONCPP_DIR=/home/leek/tools/jsoncpp/jsoncpp-install

CILK_DIR=/home/leek/tools/cilk/cilkplus-install

BOOST_DIR=/home/leek/tools/boost/boost-install
CXX=$(CILK_DIR)/bin/g++

OBJ_DIR=./build/
INC=-I$(OPENCV_DIR)/include/opencv2 -I$(OPENCV_DIR)/include
INC+=-I$(BOOST_DIR)/include
INC+=-I$(VIGRA_DIR)/include 
INC+=-I/usr/include/vtk-5.8 -I/usr/include/hdf5/serial -ISemiSupervised/ 
INC+=-I$(SPARSEHASH_DIR)/include -I$(JSONCPP_DIR)/include

PREPROC_FLAGS=
CXXFLAGS=-std=c++11
CXXFLAGS+=-fcilkplus -m64 -march=native -ffast-math -mfma -funroll-loops -flto -O3
CXXFLAGS+=$(INC)
CXXFLAGS+=$(PREPROC_FLAGS)

LDFLAGS=-L$(OPENCV_DIR)/lib -L$(VIGRA_DIR)/lib -L/usr/local/lib
LDFLAGS+=-L$(BOOST_DIR)/lib
LDFLAGS+=-L/usr/lib/x86_64-linux-gnu
LDFLAGS+=-L$(SPARSEHASH_DIR)/lib
LDFLAGS+=-L$(JSONCPP_DIR)/lib/x86_64-linux-gnu
LDFLAGS+=-L/usr/lib/x86_64-linux-gnu/hdf5/serial -L$(CILK_DIR)/lib64
LDFLAGS+=-lcilkrts -ljsoncpp -lhdf5 -lhdf5_hl -lvigraimpex -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lboost_program_options -lboost_filesystem -lboost_thread -lboost_system -lstdc++ -lgcc_s #-lprofiler

SOURCES_NP_LEARN=$(wildcard src/neuroproof_graph_learn.cpp Stack/*.cpp Classifier/*.cpp FeatureManager/*.cpp BioPriors/*.cpp Algorithms/*.cpp EdgeEditor/*.cpp Rag/*.cpp SemiSupervised/*.cpp)
SOURCES_NP_PREDICT=$(wildcard src/neuroproof_graph_predict.cpp Stack/*.cpp Classifier/*.cpp FeatureManager/*.cpp BioPriors/*.cpp Algorithms/*.cpp EdgeEditor/*.cpp Rag/*.cpp SemiSupervised/*.cpp)

OBJECTS_NP_LEARN=$(SOURCES_NP_LEARN:.cpp=.o)

OBJECTS_NP_PREDICT=$(SOURCES_NP_PREDICT:.cpp=.o)

EXECUTABLE_NP_PREDICT=neuroproof_graph_predict
EXECUTABLE_NP_LEARN=neuroproof_graph_learn

debug: PREPROC_FLAGS:=-D DEBUG_PRINT -ggdb
debug: all

all: $(SOURCES_NP_PREDICT) $(EXECUTABLE_NP_PREDICT)

learn: $(SOURCES_NP_LEARN) $(EXECUTABLE_NP_LEARN)

$(EXECUTABLE_NP_PREDICT): $(OBJECTS_NP_PREDICT)
	mkdir -p ./build
	$(CXX) $(LDFLAGS) $(OBJECTS_NP_PREDICT) -o ./build/$@

$(EXECUTABLE_NP_LEARN): $(OBJECTS_NP_LEARN) 
	mkdir -p ./build
	$(CXX) $(LDFLAGS) $(OBJECTS_NP_LEARN) -o ./build/$@

.cpp.o:
	$(CXX) $(CXXFLAGS) ${PREPROC_FLAGS} -c $< -o $@

clean:
	find . -type f -name '*.o' -exec rm {} +
	rm -f build/*
