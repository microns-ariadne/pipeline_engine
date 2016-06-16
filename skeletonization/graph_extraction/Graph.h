// Copyright 2014

#include <stdio.h>
#include <cilk/cilk.h>
#include <cilk/reducer_list.h>
#include <cilk/reducer_min.h>
#include <cilk/reducer_max.h>
#include <cilk/holder.h>

#include <vector>
#include <cmath>
#include <list>
#include <map>
#include <string>
#include <set>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

#ifndef GRAPH_H_
#define GRAPH_H_

template<typename VertexType, typename EdgeType>
class Graph {
 private:
    VertexType* vertexData;
 public:
    Graph();
    int* vertexColors;
    int vertexCount;
    int num_vertices();
    int compute_trivial_coloring();
    int compute_27coloring(int rows, int cols, int height);
    int compute_8coloring(int rows, int cols, int height);
    void resize(int size);
    VertexType* getVertexData(int vid);
    VertexType* getGridVertexData(int x, int y, int z);
    void setGridDimensions(int rows, int cols, int height);
    void getManGridEdges(int vid, std::vector<int>* b);
    void c_getCube(int vid, int radius,  int label, std::vector<int>* b);
    int _height;
    int _rows;
    int _cols;
};

#endif  // GRAPH_H_
