
void c_get_object_label_update(int vid) {
                    // ,void* scheduler_void) {
  c_graph->getVertexData(vid)->deleted = false;
  c_graph->getVertexData(vid)->checked = 0;
  c_graph->getVertexData(vid)->empty = true;
  pthread_mutex_init(&(c_graph->getVertexData(vid)->lock),NULL);
  pthread_mutex_init(&(c_graph->getVertexData(vid)->nlock),NULL);
  c_graph->getVertexData(vid)->BFSd= new std::map<int,int>();
  c_graph->getVertexData(vid)->BFSparent= new std::map<int,int>();
  c_graph->getVertexData(vid)->prunable = new std::map<int,int>();
  c_graph->getVertexData(vid)->nearestBranchDist= new std::map<int,int>();
  c_graph->getVertexData(vid)->children= new std::map<int, std::set<int>* >();
  c_graph->getVertexData(vid)->cycles= new std::map<int, std::map<int, int>* >();
  c_graph->getVertexData(vid)->roundNum= new std::map<int,int>();
  c_graph->getVertexData(vid)->direction= new std::map<int,vector3D*>();
  
  int x = (vid % cols);
  int y = ((vid-x)/cols % rows);
  int z = ((vid-x)/cols - y)/ rows;
  
  c_graph->getVertexData(vid)->x=x;
  c_graph->getVertexData(vid)->y=y;
  c_graph->getVertexData(vid)->z=z;

  label_set_init(&(c_graph->getVertexData(vid)->object_label_set));
  label_set_init(&(c_graph->getVertexData(vid)->scheduled_labels));

  if (debug) if ((vid % 1000000)==0) printf("%d\n",vid);
  
  
}

void c_peel(int vid, void* scheduler_void);
bool c_simplePoint(int vid, int label);


void c_peelInit(int vid, void* scheduler_void) {
   vdata* vd = c_graph->getVertexData(vid);
   if (!vd->empty) { // We don't care about empty nodes

   Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);
   std::vector<int> neighbors;
   c_graph->getManGridEdges(vid, &neighbors); //Manhattan distance

  int z = vd->z;

  std::vector<int> labels;
  label_set_contents(vd->object_label_set, &labels);

  for (int i = 0; i < labels.size(); i++) {
    int label = labels[i];
    if (label <= 0) continue;
    bool onSurface = false;
    for (int j = 0; j < neighbors.size(); j++) {
      vdata* nvd = c_graph->getVertexData(neighbors[j]);
      //printf("lsf: %d\n",label_set_find(nvd->object_label_set, label));
      if (nvd->z == z && !label_set_find(nvd->object_label_set, label)) { /// Surface condition
        onSurface = true;
      }
    }

    if(vd->y== 0 || vd->y == c_graph->_rows-1)
    	onSurface = true;

    if(vd->x== 0 || vd->x == c_graph->_cols-1)
        onSurface = true;

    if (onSurface) {
      label_set_insert(vd->scheduled_labels, label);
      (*(vd->roundNum))[label]=0;      
      scheduler->add_task(vid, c_peel);
    }
  }
  }
}



void c_peel(int vid, void* scheduler_void) {
//   if (graph->getVertexData(vid)->deleted ) return;
   Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);

   vdata* vd = c_graph->getVertexData(vid);
   std::vector<int> scheduled_labels;
   //printf("Working on %d\n",vid);

   //pthread_mutex_lock(&(vd->lock));
   label_set_contents(vd->scheduled_labels, &scheduled_labels);
   for (int i = 0; i < scheduled_labels.size(); i++) { /// Peel each label separately
     int label = scheduled_labels[i];
     std::vector<int> neighbors;
     c_graph->c_getCube(vid, 1, label, &neighbors);
     
     if (neighbors.size() > 1 && c_simplePoint(vid, label)) { /// Schedule neighbors
       label_set_remove(vd->object_label_set, label);
       for (int j = 0; j < neighbors.size(); j++) {
         vdata* nvd = c_graph->getVertexData(neighbors[j]);
         
         pthread_mutex_lock(&(nvd->nlock));
         label_set_insert(nvd->scheduled_labels, label);  /// The neighbor point should be scheduled with this particular label  
         (*(nvd->roundNum))[label]=scheduler->roundNum;
		 pthread_mutex_unlock(&(nvd->nlock));
		 
         scheduler->add_task(neighbors[j], c_peel);   /// The neighbor point should be scheduled 
       }//for
     }//if
	 //printf("Removing %d %d %d\n",vid,label, (*(vd->roundNum))[label]);
	
     label_set_remove(vd->scheduled_labels, label);   /// The actual deletion part of peeling
   }
   //pthread_mutex_unlock(&(vd->lock));
}

bool c_simplePoint(int vid, int label) {

  //  printf("Simple start %d\n", vid);
    int x = c_graph->getVertexData(vid)->x;
    int y = c_graph->getVertexData(vid)->y;
    int z = c_graph->getVertexData(vid)->z;

   int ex=-1; //exponent starting from 3^3-1
   int val=0; //hash value
   int rows=c_graph->_rows;
   int cols=c_graph->_cols;
   int height= c_graph->_height;

   for(int dx=-1; dx < 2; dx++) {
       for(int dy=-1; dy<2; dy++) {
	   for(int dz=-1; dz<2; dz++) {
	      ex++;
	      if (y+dy < 0 || y+dy >= rows) continue;
	      if (x+dx < 0 || x+dx >= cols) continue;
	      if (z+dz < 0 || z+dz >= height) continue;
	      int pixel = (z+dz)*rows*cols + (y+dy)*c_graph->_cols + x+dx;
              vdata* nvd = c_graph->getVertexData(pixel);
              if (label_set_find(nvd->object_label_set, label)) {
                val += pow(2,ex);
              }
	   }
       }
   }
   //printf("%d %d\n",val,label);
  if(val>pow(2,27)) printf("Simple overflow");
  return simple[val];
}



