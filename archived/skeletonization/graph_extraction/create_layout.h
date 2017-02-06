#include <csignal>

void BFSInit(int vid);
void BFS(int vid, void* scheduler_void);
void createLayout(int label);
void removeShortBranches(int label, int minBranchSize);
void removeDeg2Nodes(int label, bool removeDeg2Nodes);
void markSynapses();
void addSynapses(int label);

void BFSInit(int vid) {
 
  vdata* vd = c_graph->getVertexData(vid);
  std::vector<int> labels;
  label_set_contents(vd->object_label_set, &labels);

  for (int i = 0; i < labels.size(); i++) {
    int label = labels[i];
    if (label <= 0) continue;

    if (BFSseads[label]==-1) {
       BFSseads[label]=vid;
       (*(c_graph->getVertexData(vid)->BFSd))[label]=0;
       (*(c_graph->getVertexData(vid)->BFSparent))[label]=-1;
       label_set_insert(vd->scheduled_labels, label);
      // scheduler->add_task(vid, BFS);
       distinct_labels+=1;
    }
  }
}


void BFS(int vid, void* scheduler_void) {
   Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);

   vdata* vd = c_graph->getVertexData(vid);

   std::vector<int> scheduled_labels;
   label_set_contents(vd->scheduled_labels, &scheduled_labels);

   //printf("BFS enter %d, scheduled size= %d \n",vid, scheduled_labels.size());
   
   for (int i = 0; i < scheduled_labels.size(); i++) { //BFS for each label
     int label = scheduled_labels[i];

     std::vector<int> neighbors;
     c_graph->c_getCube(vid, 1, label, &neighbors);
     std::vector<int> children; 
     std::vector<int> cycles;     
     std::vector<int> cycle_lengths;      
 
     for (int j = 0; j < neighbors.size(); j++) {
       pthread_mutex_lock(&(c_graph->getVertexData(neighbors[j])->lock));
       std::map<int,int>::iterator it = c_graph->getVertexData(neighbors[j])->BFSd->find(label);
       if (it==c_graph->getVertexData(neighbors[j])->BFSd->end()){
		 //These are the children
         (*(c_graph->getVertexData(neighbors[j])->BFSd))[label]=(*(c_graph->getVertexData(vid)->BFSd))[label]+1;
         (*(c_graph->getVertexData(neighbors[j])->BFSparent))[label]=vid;
	     label_set_insert(c_graph->getVertexData(neighbors[j])->scheduled_labels, label);
         scheduler->add_task(neighbors[j] , BFS);
         children.push_back(neighbors[j]);
       }
      else if (findCycles  && (*(c_graph->getVertexData(vid)->BFSparent))[label]!=neighbors[j]) {
         //Do chordless cycle detection
         std::map<int,int> parentSet;
         int curr1 = vid;
         int curr2 = neighbors[j];
         
         // Find least common parent
         int level_up = 0; int curr1m1=-1; int curr2m1=-1; 
         while ((curr1!=curr2) && (parentSet.count(curr1)==0) &&  (parentSet.count(curr2)==0)) {
            if (curr1>=0) {
             	parentSet.insert(std::pair<int,int>(curr1,level_up));
             	curr1 = (*(c_graph->getVertexData(curr1)->BFSparent))[label];
             	if (curr1==-1) curr1m1=curr1;
			}
            if (curr2>=0) {
         		parentSet.insert(std::pair<int,int>(curr2,level_up));
    			curr2 = (*(c_graph->getVertexData(curr2)->BFSparent))[label];
             	if (curr2==-1) curr2m1=curr2;
    		}
    		level_up++;
    	 }
    	 
    	 // Find cycle lenght
    	 int cycle_length = 0;
    	 if (parentSet.count(curr1)>0) cycle_length = level_up + parentSet.at(curr1);
    	 if (parentSet.count(curr2)>0) cycle_length = level_up + parentSet.at(curr2);
    	 if ((curr1==curr2) && (curr1>=0)) cycle_length = 2*level_up;
    	 if ((curr1==curr2) && (curr1==-1)) cycle_length = curr1m1 + curr2m1;
    	 if (cycle_length>minCycleSize){
    	   if (debug) printf("In object %d there is a cycle of length %d\n", label,cycle_length);
    	   
    	   cycle_lengths.push_back(cycle_length);
    	   cycles.push_back(neighbors[j]);
    	   BFScycle[label]=true;
         }
    	 
       }

       
       pthread_mutex_unlock(&(c_graph->getVertexData(neighbors[j])->lock));
     }

     /// Save the children!!
     (*(c_graph->getVertexData(vid)->children))[label]=new std::set<int>;
     for (int i=0; i<children.size(); i++) {
       (*(c_graph->getVertexData(vid)->children))[label]->insert(children[i]);
     }
     
     (*(c_graph->getVertexData(vid)->cycles))[label]=new std::map<int,int>;
     for (int i=0; i<cycles.size(); i++) {
       (*(c_graph->getVertexData(vid)->cycles))[label]->insert(std::pair<int,int>(cycles[i],cycle_lengths[i]));
     }

     label_set_remove(vd->scheduled_labels, label);
  }  
}   


void createLayout(int label, int minBranchSize, bool removeBranch, bool removeDeg2) {
  if (removeBranch) removeShortBranches(label, minBranchSize);
  removeDeg2Nodes(label, removeDeg2);
}



void removeShortBranches(int label, int minBranchSize) {
  //removing short branches
  std::stack<int> stack;
  stack.push(BFSseads[label]);
  //printf("SEED: %d\n",BFSseads[label]);
  //DFS on graph

  
  while(!stack.empty()) {
    int vid=stack.top();
    stack.pop();

    //avoiding deleted nodes
    while (!label_set_find(c_graph->getVertexData(vid)->object_label_set, label)) {
      if (stack.empty()) return;
      vid=stack.top();
      stack.pop();
    }

    std::set<int>* children = (*(c_graph->getVertexData(vid)->children))[label];
    int numChildren = children->size();
    
    for (std::set<int>::iterator i = children->begin(); i != children->end(); i++)
    {
      int child = *i; 
      stack.push(child);
      //calculating branch size
      if (numChildren==1) { 
        (*(c_graph->getVertexData(child)->nearestBranchDist))[label]=(*(c_graph->getVertexData(vid)->nearestBranchDist))[label]+1;
      }
      else {
        (*(c_graph->getVertexData(child)->nearestBranchDist))[label]=1;
      }
    }

    //don't prune things that were once a branchpoint
    if (numChildren>1)  (*(c_graph->getVertexData(vid)->prunable))[label]=0;
    else                (*(c_graph->getVertexData(vid)->prunable))[label]=1;
    

    
    //don't prune cycle endings
    if (findCycles && (*(c_graph->getVertexData(vid)->cycles))[label]->size()>0) {
      //printf("Override! %d\n",label);
      (*(c_graph->getVertexData(vid)->prunable))[label]=0;
    }
    
    //removing short branches
   // minBranchSize = (*(c_graph->getVertexData(vid)->roundNum))[label];
    
    if (numChildren==0 &&  (*(c_graph->getVertexData(vid)->prunable))[label]) {
      if ((*(c_graph->getVertexData(vid)->nearestBranchDist))[label]<minBranchSize) {
        int parent = (*(c_graph->getVertexData(vid)->BFSparent))[label];
		int me = vid;
		if (parent!=-1  ) {
          do {
          //     printf("+1 remove\n");
              label_set_remove(c_graph->getVertexData(me)->object_label_set, label);
              (*(c_graph->getVertexData(parent)->children))[label]->erase(me);
              me=parent;
              parent=(*(c_graph->getVertexData(parent)->BFSparent))[label];
          }  while ((parent!=-1 && (*(c_graph->getVertexData(me)->children))[label]->size()==0) && (*(c_graph->getVertexData(me)->prunable))[label]>0);
        }
      }//in short
    
    }//if endpoint
  }//while stack
}


void removeDeg2Nodes(int label, bool doRemove) {
  //removing short branches
  std::stack<int> stack;
  stack.push(BFSseads[label]);
  //printf("SEED: %d\n",BFSseads[label]);
  //DFS on graph

  int numel=0;
  
  while(!stack.empty()) {
    numel++;
    int vid=stack.top();
    stack.pop();

    std::set<int>* childrenPointer = (*(c_graph->getVertexData(vid)->children))[label];
    int numChildren = childrenPointer->size();
    int* children = new int[numChildren];
    int parent= (*(c_graph->getVertexData(vid)->BFSparent))[label];
    
    //Deepcopy because we are going to delete from the original children set
    int index=0;
    for (std::set<int>::iterator i = childrenPointer->begin(); i != childrenPointer->end(); i++) {
	   children[index]=*i;
	   index++;
    }

    ////////// direction
    float factor = 0.7;
    if (parent==-1) {
      (*(c_graph->getVertexData(vid)->direction))[label]= new vector3D();
    }
    else {
      vdata* v = c_graph->getVertexData(vid);
      vdata* vp = c_graph->getVertexData(parent);

      (*(c_graph->getVertexData(vid)->direction))[label]= new vector3D((float)(v->x - vp->x),(float)(v->y - vp->y),(float) (v->z - vp->z));
      (*(c_graph->getVertexData(vid)->direction))[label]->add((*(c_graph->getVertexData(parent)->direction))[label]->multiply(factor));
      //printf("vector %s\n",(*(c_graph->getVertexData(vid1)->direction))[label1]->toString()); 
    }
    (*(c_graph->getVertexData(vid)->direction))[label]->normalize();
    ////// end direction 

  //  printf("v, %d l %d\n",vid, label);

    if (doRemove && parent!=-1 && numChildren==1) {
       //Delete the point
       label_set_remove(c_graph->getVertexData(vid)->object_label_set, label);
       (*(c_graph->getVertexData(parent)->children))[label]->erase(vid);
    } 
    //Maintain graph structure + adding child to stack
    for (int i=0 ; i<numChildren; i++)
    {
      int child = children[i]; 
      stack.push(child);

     if (doRemove && parent!=-1 && numChildren==1) {
        (*(c_graph->getVertexData(child)->BFSparent))[label]=(*(c_graph->getVertexData(vid)->BFSparent))[label];
        (*(c_graph->getVertexData(parent)->children))[label]->insert(child);
     }
    }
   
        
  }//while stack
}

