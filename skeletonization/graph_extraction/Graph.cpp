#include "Graph.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>

//Non-deleted bject points in a cube
template<typename VertexType, typename EdgeType>
Graph<VertexType, EdgeType>::Graph() {
  vertexData = NULL;
}

///// Getters
template<typename VertexType, typename EdgeType>
VertexType* Graph< VertexType,  EdgeType>::getVertexData(int vid){
  return &vertexData[vid];
}

template<typename VertexType, typename EdgeType>
VertexType* Graph< VertexType,  EdgeType>::getGridVertexData(int x, int y, int z){
  int vid = z*_rows*_cols + y*_cols + x;
  return &vertexData[vid];
}

template<typename VertexType, typename EdgeType>
int Graph< VertexType,  EdgeType>::num_vertices(){
  return vertexCount;
}

//Non-deleted bject points in a cube
template <typename VertexType, typename EdgeType>
void Graph<VertexType, EdgeType>::c_getCube(int vid, int radius,  int label, std::vector<int>* b) {
  VertexType* vd = this->getVertexData(vid);
  int x = vd->x;
  int y = vd->y;
  int z = vd->z;
  int rows = _rows;
  int cols = _cols;
  int height = _height;
  for (int dx = -radius; dx <= radius; dx++) {
    for (int dy = -radius; dy <= radius; dy++) {
      for (int dz = -radius; dz <= radius; dz++) {
        if (y+dy < 0 || y+dy >= rows) continue;
        if (x+dx < 0 || x+dx >= cols) continue;
        if (z+dz < 0 || z+dz >= height) continue;
		int pixel = (z+dz)*_rows*_cols + (y+dy)*_cols + x+dx;
        if (vid == pixel) continue;
        VertexType* nvd = this->getVertexData(pixel);

        if (label_set_find(nvd->object_label_set, label)) {
	  b->push_back(pixel);
        }
      }
    }
  }
}

//Manhattan distance
template <typename VertexType, typename EdgeType>
void Graph<VertexType, EdgeType>::getManGridEdges(int vid, std::vector<int>* b) {
  VertexType* vd = this->getVertexData(vid);
  int x = vd->x;
  int y = vd->y;
  int z = vd->z;
  int rows = _rows;
  int cols = _cols;
  int height = _height;
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dz = -1; dz <= 1; dz++) {
        if (y+dy < 0 || y+dy >= rows) continue;
        if (x+dx < 0 || x+dx >= cols) continue;
        if (z+dz < 0 || z+dz >= _height) continue;
	if (abs(dx)+abs(dy)+abs(dz)>1) continue;
        int pixel = (z+dz)*_rows*_cols + (y+dy)*_cols + x+dx;
        if (vid == pixel) continue;
        b->push_back(pixel);
      }
    }
  }
}

/// Setters
template<typename VertexType, typename EdgeType>
void Graph<VertexType, EdgeType>::setGridDimensions(int rows, int cols, int height) {
  this->_rows = rows;
  this->_cols = cols;
  this->_height = height;
}



/// Modifier
template<typename VertexType, typename EdgeType>
void Graph<VertexType, EdgeType>::resize(int size){
  vertexCount = size;
  uint64_t new_size = size * sizeof(VertexType);
  //vertexData = (VertexType*) paged_malloc(new_size);
  vertexData = (VertexType*) realloc(vertexData, vertexCount * sizeof(VertexType));
}

////////// Colorings

template<typename VertexType, typename EdgeType>
int Graph< VertexType,  EdgeType>::compute_trivial_coloring(){
  vertexColors = (int*)calloc(sizeof(int), vertexCount);
//  memset(vertexColors,0,sizeof(int)*vertexCount);
  return 1;
}

// Compute a distance 2 coloring.
template<typename VertexType, typename EdgeType>
int Graph< VertexType, EdgeType>::compute_27coloring(int rows, int cols, int height) {
  vertexColors = (int*) malloc(sizeof(int) * (rows)*(cols)*(height));
  int max_color = 0;
  cilk_for (int z = 0; z < height; z++) {
    int z_offset = (z%3)*9;/* + 8*(z/2);*/
    //int z_offset = (z%2)*4;
    for (int y = 0; y < rows; y++) {
      int y_offset = (y%3)*3;
      for (int x = 0; x < cols; x++) {
        int x_offset = x%3;
        int color = x_offset + y_offset + z_offset;
        vertexColors[z*rows*cols+y*cols+x] = color;
        if (color > max_color) max_color = color;
      }
    }
  }
  return max_color+1;
}


// Compute a distance 2 coloring.
template<typename VertexType, typename EdgeType>
int Graph< VertexType, EdgeType>::compute_8coloring(int rows, int cols, int height) {
  vertexColors = (int*) malloc(sizeof(int) * (rows)*(cols)*(height));
  int max_color = 0;
  cilk_for (int z = 0; z < height; z++) {
    int z_offset = (z%2)*4;/* + 8*(z/2);*/
    //int z_offset = (z%2)*4;
    for (int y = 0; y < rows; y++) {
      int y_offset = (y%2)*2;
      for (int x = 0; x < cols; x++) {
        int x_offset = x%2;
        int color = x_offset + y_offset + z_offset;
        vertexColors[z*rows*cols+y*cols+x] = color;
        if (color > max_color) max_color = color;
      }
    }
  }
  return max_color+1;
}