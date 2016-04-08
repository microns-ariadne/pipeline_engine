CXX:=g++
CXXFLAGS:=-std=c++11 -O2
LFLAGS:=-L/home/armafire/tools/opencv-install/lib -lopencv_core -lopencv_imgcodecs -lopencv_ml -lboost_program_options -lboost_filesystem -lboost_system -lopencv_highgui -lopencv_imgproc 
#-lopencv_contrib

# not sure why jsoncpp didn't find the absolute path
# we hopefully don't need this: -I/usr/include/python2.7/
# on haswell: 
#	pkg-config --cflags opencv
# 		-I/usr/local/include/opencv -I/usr/local/include 
INC:=-I/home/armafire/tools/opencv-install/include/ -I/home/armafire/tools/opencv-install/include/opencv2
# PREPROC_FLAGS:=-D DEBUG_PRINT
PREPROC_FLAGS:=
SOURCES:=$(wildcard watershed.cpp ws_alg.cpp ws_queue.cpp)

all: main 

debug: PREPROC_FLAGS:=-D DEBUG_PRINT
debug: main

main: ${SOURCES}
	${CXX} ${CXXFLAGS} ${PREPROC_FLAGS} ${INC} -o build/watershed.x ${SOURCES} ${LFLAGS}

clean:
	rm -f build/*
