#
# To be run as part of the root directory Makefile: "make neuroproof"
#
# Defines coming into here should be defined by the parent makefile
#
# OPENCV_DIR Location of OpenCV-2.4 install
#
# VIGRA_DIR Location of Vigra install
#
# SPARSEHASH_DIR Location of sparsehash install
#
# JSONCPP_DIR=tools/jsoncpp/jsoncpp-install
#
# CILK_DIR=tools/cilk/cilkplus-install
#
# BOOST_DIR=tools/boost/boost-install
CXX=$(CILK_DIR)/bin/g++

OBJ_DIR=./build/
INC=-I$(OPENCV_DIR)/include/opencv2 -I$(OPENCV_DIR)/include
INC+=-I$(BOOST_DIR)/include
INC+=-I$(VIGRA_DIR)/include
#INC+=-I$(CILK_DIR)/include/c++/4.9.0
INC+=-I/usr/include/vtk-5.8 -I/usr/include/hdf5/serial -ISemiSupervised/ 
INC+=-I$(SPARSEHASH_DIR)/include -I$(JSONCPP_DIR)/include

PREPROC_FLAGS=
CXXFLAGS=-std=c++11
CXXFLAGS+=-D_GLIBCXX_USE_CXX11_ABI=1
CXXFLAGS+=-fcilkplus -m64 -march=native -ffast-math -mfma -funroll-loops -flto -O3 -g
CXXFLAGS+=$(INC)
CXXFLAGS+=$(PREPROC_FLAGS)

LDFLAGS=-L$(OPENCV_DIR)/lib -L$(VIGRA_DIR)/lib
LDFLAGS+=-L$(BOOST_DIR)/lib
LDFLAGS+=-L$(SPARSEHASH_DIR)/lib
LDFLAGS+=-L$(JSONCPP_DIR)/lib
LDFLAGS+=-L/usr/lib/x86_64-linux-gnu/hdf5/serial -L$(CILK_DIR)/lib64
LDFLAGS+=-L/usr/lib/x86_64-linux-gnu -L/usr/local/lib
LDFLAGS+=-std=c++11
LDFLAGS+=-lcilkrts -ljsoncpp -lhdf5 -lhdf5_hl -lvigraimpex -lopencv_core -ltiff
LDFLAGS+=-lopencv_imgproc -lopencv_highgui -lopencv_ml -lboost_program_options
LDFLAGS+=-ltiff
LDFLAGS+=-lboost_filesystem -lboost_thread -lboost_system -lstdc++ -lgcc_s #-lprofiler
#LDFLAGS+=-lmcheck

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
