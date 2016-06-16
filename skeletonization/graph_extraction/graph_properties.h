// Copyright 2014

#ifndef GRAPH_EXTRACTION_GRAPH_PROPERTIES_H_
#define GRAPH_EXTRACTION_GRAPH_PROPERTIES_H_

#include "Graph.h"
#include "cilk_tools/scheduler.h"
#include "cilk_tools/engine.h"
#include "label_set.h"
#include <pthread.h>
#include "./vector3D.h"

// Programmer defined vertex and edge structures.
//   x,y,z required.
struct uncoarsened_vdata{
  explicit uncoarsened_vdata(uint16_t object_label = 0) : object_label(object_label) { }
  uint32_t object_label;
  uint16_t synapse_label;
  uint16_t x;
  uint16_t y;
  uint16_t z;
  uint8_t pixel_value;
};

// Programmer defined vertex and edge structures.
//   x,y,z required.
struct vdata{
  explicit vdata(uint16_t object_label = 0) : object_label(object_label) { }
  int checked;
  std::map<int,int>* roundNum;
  std::map<int,int>* BFSd;
  std::map<int,int>* BFSparent;
  std::map<int,int>* numChildren;
  std::map<int, std::set<int>* >* children;
  std::map<int, std::map<int,int>* >* cycles;
  std::map<int,int>* prunable;
  std::map<int,int>* nearestBranchDist;
  std::map<int, vector3D* >* direction;
  int level;
  uint32_t object_label;
  uint16_t x;
  uint16_t y;
  uint16_t z;
  uint8_t pixel_value;
  bool deleted;
  bool empty;
  label_set scheduled_labels;
  label_set object_label_set;
  pthread_mutex_t lock;
  pthread_mutex_t nlock;

};

// This structure isn't really used since we're
// representing the edges implicitly.
struct edata {
  double weight;
  explicit edata(double weight = 1) : weight(weight) { }
};

// The scheduler and graph objects. Global for
// convenience.
static Scheduler* scheduler;
static engine<uncoarsened_vdata, edata>* e;

static Scheduler* c_scheduler;
static engine<vdata, edata>* c_engine;

static Graph<uncoarsened_vdata, edata>* graph;
static Graph<vdata, edata>* c_graph;
static bool terminated = false;

static int s_height = 100;  // Number of images in the stack.
static int s_rows = 1024;  // pixel rows in image in stack.
static int s_cols = 1024;  // pixel cols in image in stack.

#endif  // GRAPH_EXTRACTION_GRAPH_PROPERTIES_H_
