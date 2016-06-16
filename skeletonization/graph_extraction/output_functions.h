void save_swc() {
    if (debug) printf("Saving extracted data to SWC file\n");
    if (debug) printf("Max label is %d, of these %d correspond to objects\n", max+1, distinct_labels);
    
    int count_labels=0;
    
    for (int label=1; label <max+1; label++) {
      if (BFSseads[label]!=-1) {
        count_labels++;
        
        char skeleton_filename[10000];
        sprintf(skeleton_filename,"%s/SWC/%s_%d.swc", OUTPUT_IMAGE_DIR.c_str(), OUTPUT_IMAGE_FILENAME_PREFIX.c_str(), label);
        FILE* skeleton_file;
        skeleton_file= fopen(skeleton_filename,"w");
        
    	char cycles_filename[10000];
        sprintf(cycles_filename,"%s/cycles/%s_%d.swc", OUTPUT_IMAGE_DIR.c_str(), OUTPUT_IMAGE_FILENAME_PREFIX.c_str(), label);
        FILE* cycles_file;
        if (findCycles && BFScycle[label]) cycles_file= fopen(cycles_filename,"w");

        std::stack<int> stack;
        stack.push(BFSseads[label]);
	    //double DFS on graph

        while(!stack.empty()) {
          int vid=stack.top();
          stack.pop();
          std::set<int>* childrenPointer = (*(c_graph->getVertexData(vid)->children))[label];
          std::map<int,int>* cyclesPointer = (*(c_graph->getVertexData(vid)->cycles))[label];

          vdata* vd = c_graph->getVertexData(vid);
          
	      int block_x_id = vd->x * block_x_size;
          int block_y_id = vd->y * block_y_size;                                                                                                                  
          int block_z_id = vd->z * block_z_size * 6; // Accounting for resolution distorsion
 
     
          int type=2; //normal point
          int numChildren=0;
          int parent = (*(vd->BFSparent))[label];
          float size = (float)(*(vd->roundNum))[label]; 
     
          numChildren=(*(c_graph->getVertexData(vid)->children))[label]->size();   
          if (numChildren>1) type=5; //branch point
          if (numChildren==0 || (parent==-1 && numChildren==1)) type=6; //end point
     
          fprintf(skeleton_file,"%d %d %f %f %f %f %d\n", vid , type, (float)block_x_id, (float)block_y_id, (float)block_z_id, size, parent);
          
          //Outputting cycles
          if (findCycles && BFScycle[label]) {
            for (std::map<int,int>::iterator i = cyclesPointer->begin(); i != cyclesPointer->end(); i ++) {               
              vdata* nvd = c_graph->getVertexData(i->first);
              
	          int n_block_x_id = nvd->x * block_x_size;
              int n_block_y_id = nvd->y * block_y_size;                                                                                                                  
              int n_block_z_id = nvd->z * block_z_size * 6; // Accounting for resolution distorsion
              
              fprintf(cycles_file,"//%d\n", i->second);
              fprintf(cycles_file,"%d %d %f %f %f %f %d\n", vid , 4, (float)block_x_id, (float)block_y_id, (float)block_z_id, size, -1);
              fprintf(cycles_file,"%d %d %f %f %f %f %d\n", i->first , 4, (float)n_block_x_id, (float)n_block_y_id, (float)n_block_z_id, size, vid);
            }
          }
          
          // Add the children to the stack to continue DFS
          for (std::set<int>::iterator i = childrenPointer->begin(); i != childrenPointer->end(); i ++) {
            stack.push(*i);
          }
            
        }//while

        fclose(skeleton_file); 
        if (findCycles && BFScycle[label]) fclose(cycles_file);

      }//if
    }//for
    
    if (debug) printf("Output: %d files\n", count_labels);
}

//From Tim's utils
template <typename T>
  std::string NumberToString ( T Number )
  {
     std::ostringstream ss;
     ss << Number;
    return ss.str();
}

void save_tif() {
    cilk_for (int d = 0; d < s_height; ++d) {
        cv::Mat image(s_rows, s_cols, CV_8UC3);
        for (int p = 0; p < s_rows * s_cols; ++p) {
            uint32_t id = data[d * s_rows * s_cols + p];
            if (saveOne==-1 || saveOne==id) {
              // opencv has bgr, we want rgb
              image.data[3 * p + 0] = ((uint8_t)id & 255);            // cv::blue
              image.data[3 * p + 1] = ((uint8_t)(id >> 8) & 255);     // cv::green
              image.data[3 * p + 2] = ((uint8_t)(id >> 16) & 255);    // cv::red
            }
        }

 	      std::string pathtif = OUTPUT_IMAGE_DIR + "/" +
          OUTPUT_IMAGE_FILENAME_PREFIX + "-" +
          NumberToString(d+1) + ".tif";
      imwrite(pathtif.c_str(), image);
      std::cout<<pathtif.c_str() << " written\n";
    }
}



