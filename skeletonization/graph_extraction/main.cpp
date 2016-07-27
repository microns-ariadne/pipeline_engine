// Copyright 2014

#include <assert.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk/reducer_opadd.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdio>

#include <string>
#include <set>
#include <queue>
#include <stack>
#include <map>
#include <utility> //std::pair

#include "graph_properties.h"  // Includes vdata/edata and height/cols/rows.
//#include "imageGraphIO.h"
#include "Graph.h"
#include "cilk_tools/engine.h"
#include "cilk_tools/scheduler.h"
#include "Graph.cpp"
#include "cilk_tools/engine.cpp"
#include "cilk_tools/scheduler.cpp"


//// Defaults
#include "graph_extraction_config.h"
//init
#include "graph_extraction_init.h"
/// functions and procedures
#include "loadData.h"
#include "thinning.h"
#include "create_layout.h"
#include "output_functions.h"
#include <dirent.h>


//Input:  Object-Labeled image stack
// Output: Graph with objects as vertices and interactions as edges.
int main(int argc, char **argv) {
  if (debug) printf("A total of %d arguments provided.\n", argc-1);
  if (argc != 8) {
    syntax:
    printf("./main -{s,d,f} [downsampling scale] [path of h5 files] [output_directory] [width] [height] [depth]\n");
    printf("For arg 1 (-{s,d,f}) specify -s to compute and save skeletons, -d to read multiple files from a directory, -f read only one file\n");
    printf("For arg 2 (downsampling scale) could be 1, 2 or 4\n");
    printf("For arg 3 (labeled_image_directory or file) specify a directory or file containing the h5 file(s) corresponding to the segmentation.\n");
    printf("For arg 4 (output_directory) specify the directory where output files will be written, note that this directory must contain a subdirectory named SWC.\n");
    printf("For arg 5, 6 and 7, specify the width, height and depth of the volume being processed\n");
    exit(-1);
  }

  if (argc > 1) {
     std::string argstring(argv[1]);
     if (argstring[0]=='-') {
       std::string str2 = "s";
       std::size_t found = argstring.find(str2);
       if (found!=std::string::npos) {
	 	 doThinning=true;
		 saveSWC=true;
       }
     
       str2 = "a";
       found = argstring.find(str2);
       if (found!=std::string::npos) {
           INPUT_TYPE="all";
       }
     
	   str2 = "b";
       found = argstring.find(str2);
       if (found!=std::string::npos) {
           INPUT_TYPE="block";
       }
      
       str2 = "c";
       found = argstring.find(str2);
       if (found!=std::string::npos) {
           findCycles=true;
       }
    }

  }
  
  s_rows = atoi(argv[5]);
  s_cols = atoi(argv[6]);
  s_height = atoi(argv[7]);

  //Downsampling
  if (argc > 2) {
    std::string argstring(argv[2]);
    scale = atoi(argstring.c_str());
    mBS = unscaledBS/scale;
    block_x_size=8*scale; block_y_size=8*scale; block_z_size=1*scale;
    c_height= (s_height + block_z_size - 1) / block_z_size; 
    c_cols  = (s_cols + block_x_size - 1) / block_x_size; 
    c_rows  = (s_rows + block_y_size - 1) / block_y_size;
  } 

  // If input directories are specified, then overwrite defaults.
  if (argc>4) {
    if (strcmp(INPUT_TYPE.c_str(),"block")==0) 
    	LABELED_IMAGE_BLOCK = std::string(argv[3]);
    else if (strcmp(INPUT_TYPE.c_str(),"all")==0) 
    	LABELED_IMAGE_ALL = std::string(argv[3]);
    OUTPUT_IMAGE_DIR = std::string(argv[4]);
  }


  /// Random seed
  srand(0);

  std::string LABELED_IMAGE_FILENAME = argv[3];
  std::string LABELED_IMAGE_FILENAME_EXT=argv[3]+LABELED_IMAGE_FILENAME.find_last_of(".");
  //////////////// Figure out dimensions ///////////////////
  if (strcmp(LABELED_IMAGE_FILENAME_EXT.c_str(), ".h5")==0) {
    if (strcmp(INPUT_TYPE.c_str(), "all")==0) loadImages_all_h5_init();
    else if (strcmp(INPUT_TYPE.c_str(), "block")==0) loadImages_block_h5_init();
  }
  else if (strcmp(LABELED_IMAGE_FILENAME_EXT.c_str(), ".png")==0) {
    if (strcmp(INPUT_TYPE.c_str(), "all")==0) loadImages_all_png_init();
    else if (strcmp(INPUT_TYPE.c_str(), "block")==0) loadImages_block_png_init();
  }

  //////////// Initializing the graph
  // Modify the graph_properties.h file local to graph_extraction directory
  //   to add additional fields to vdata structure.
  c_graph = init_coarse_graph( rows, cols, height);

  int c_colorCount = c_graph->compute_8coloring(c_graph->_rows,
//  int c_colorCount = c_graph->compute_27coloring(c_graph->_rows,
      c_graph->_cols, c_graph->_height); 
  if (debug) printf("Using %d colors for datagraph parallelization\n",c_colorCount );

 
  c_scheduler = new Scheduler(c_graph->vertexColors, c_colorCount,
      c_graph->num_vertices());
  c_engine = new engine<vdata, edata>(c_graph, c_scheduler);
 
 
  if (debug) printf("Initializing graph nodes\n");
 // cilk_for (int xyz = 0; xyz < c_graph->num_vertices(); xyz++) {
  //    c_scheduler->add_task(xyz, c_get_object_label_update);
  //}
  for (int xyz = 0; xyz < c_graph->num_vertices(); xyz++) c_get_object_label_update(xyz);
  
  //c_engine->run();
  if (debug) printf("Done initializing\n");

  //////////////// Loading the images ///////////////////
  if (strcmp(LABELED_IMAGE_FILENAME_EXT.c_str(), ".h5")==0) {
    if (strcmp(INPUT_TYPE.c_str(), "all")==0)  loadImages_all_h5();
    else if (strcmp(INPUT_TYPE.c_str(), "block")==0) loadImages_block_h5();
  } 
  else if (strcmp(LABELED_IMAGE_FILENAME_EXT.c_str(), ".png")==0) {
    if (strcmp(INPUT_TYPE.c_str(), "all")==0)  loadImages_all_png();
    else if (strcmp(INPUT_TYPE.c_str(), "block")==0) loadImages_block_png();
  }   
  
    
  if (doThinning) {
    loadLUT();
    
    ///////////////// Thinning ////////////////////////////////
    if (debug) printf("Adding thinning tasks\n");
    cilk_for (int count = 0; count< c_graph->num_vertices();  count++) {
     // int xyz= (int)(count * pow(3,4)) % c_graph->num_vertices(); //"randomizing" order
      c_scheduler->add_task(count, c_peelInit);
    }
    if (debug) printf("Starting thinning\n");
    c_engine->run();
    if (debug) printf("Thinning done!\n");

 
    ///////////// BFS ////////////////////////////////
    if (debug) printf("Starting to create layout graph\n");
    BFSseads=new int[max+1];
    BFScycle=new bool[max+1];
    for (int i=0; i<max+1; i++) {
      BFSseads[i]=-1;
      BFScycle[i]=false;
    }
 
    // Keep this serial for now to get deterministic roots.
    for (int xyz = 0; xyz < c_graph->num_vertices(); xyz++) {
      BFSInit(xyz);
    }
    
    cilk_for (int i=1; i<max+1; i++) {
      if (BFSseads[i]!=-1) {
        c_scheduler->add_task(BFSseads[i], BFS);
      }
    }
    c_engine->run(); 
 
    //////////// Creating Layout graph ////////////////////
    // NOTE(TFK): Serial for now.
    for (int label=1; label<max+1; label++) {
      if (BFSseads[label]!=-1) {
         createLayout(label,mBS, removeBranches, removeDeg2);
      }
    }
    if (debug) printf("Layout graph done\n");

  } //end of doThinning

  //////////////// OUTPUT ////////////////
  if (saveSWC) save_swc();
  
  //////////  Coloring  /////////////////////////

  if (saveTIF) save_tif();

  
  if (debug) printf("DONE!\n");


  return 0;
}
