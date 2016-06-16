#include "H5Cpp.h"
#include <dirent.h>
#include <string>
#include <sstream>
#include <vector>


#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

typedef uint32_t  label_t;


std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


void create_cgraph_block(Graph<vdata, edata>* c_graph, label_t* data,
    int block_x_begin, int block_x_end, int block_y_begin, int block_y_end,
    int block_z_begin, int block_z_end, int block_x_size, int block_y_size,
    int block_z_size, int shift_z, int shift_x, int shift_y)
{    
  int block_x_id = block_x_begin / block_x_size;
  int block_y_id = block_y_begin / block_y_size;
  int block_z_id = block_z_begin / block_z_size;
  
  vdata* c_vdata = c_graph->getGridVertexData(block_x_id+shift_x, block_y_id+shift_y,
      block_z_id+shift_z);
  
  if ((c_vdata->x!=block_x_id+shift_x) || (c_vdata->y!=block_y_id+shift_y) || (c_vdata->z!=block_z_id+shift_z)) std::cerr << "Indexing error!";

  for (int z = block_z_begin; z < block_z_end; z++) {
    for (int x = block_x_begin; x < block_x_end; x++) {
      for (int y = block_y_begin; y < block_y_end; y++) {
       // uint32_t data_label = data[s_height*(s_rows*x + y)+z];
//       uint32_t data_label = data[s_height*(s_cols*y + x)+z];
       uint32_t data_label = data[z*s_cols*s_rows+s_cols*y + x];
        if (data_label != 0) {
          label_set_insert(c_vdata->object_label_set, data_label);
          c_vdata->empty=false;
              
          if (data_label > max) max = data_label; /////// Find max label
        }
      }
    }
  }
}

////// h5 input functions
void loadImages_all_h5_init() {
  ///////////////// Here we will determine the dimensions of the data based on the block indices
  DIR*    dir;
  dirent* pdir;
  dir = opendir(LABELED_IMAGE_ALL.c_str());
  
  b_height=0;
  b_cols=0;
  b_rows=0;
  
  while (pdir = readdir(dir)) {
   // if (!std::regex_match (pdir->d_name, std::regex("out(.*)\\.h5") )) continue;
    if ((pdir->d_name[0]!='o') || (pdir->d_name[1]!='u') || (pdir->d_name[2]!='t')) continue;
    std::vector<std::string> parts =split(pdir->d_name, '_');
    int b_z=std::atoi(parts.at(2).c_str());
    int b_x=std::atoi(parts.at(3).c_str());
    int b_y=std::atoi(parts.at(4).c_str());
    if (b_z>b_height) b_height= b_z;
    if (b_x>b_cols) b_cols=b_x;
    if (b_y>b_rows) b_rows=b_y;
    if (debug) printf("Block indices %d %d %d\n", b_z, b_x, b_y);
  }
  
  if (debug) printf("Max block indices %d %d %d\n", b_height, b_cols, b_rows);
  
  height = c_height * (1+b_height);
  cols   = c_cols   * (1+b_cols);
  rows   = c_rows   * (1+b_rows);
  if (debug) printf("The dimension will be %d %d %d\n", height, cols, rows);
}

void loadImages_all_h5() {
  ///////////////////////// We will read in the data one by one and coarsen it on the fly
  DIR*    dir;
  dirent* pdir;
  dir = opendir(LABELED_IMAGE_ALL.c_str());
  std::vector<std::string> fnames;
  while (pdir = readdir(dir)) {
    if ((pdir->d_name[0]!='o') || (pdir->d_name[1]!='u') || (pdir->d_name[2]!='t')) continue;
    fnames.push_back(std::string(pdir->d_name));
  }

  int batch_size=36;
  int BLOCK_SIZE= s_rows*s_cols*s_height;
  label_t** data= new label_t*[batch_size];
  for(int i=0; i < batch_size; i++) {
        data[i] = new label_t[BLOCK_SIZE];
  }

  for(int batch_id=0; batch_id<fnames.size(); batch_id+=batch_size){
   int temp_var=fnames.size()-batch_id;
   
   cilk_for(int i=0; i < std::min(batch_size,temp_var); i++) {
    int batch_index=i+batch_id;
    std::string fname = fnames.at(batch_index);
    std::vector<std::string> x =split(fname, '_');
	int b_z=std::atoi(x.at(2).c_str());
    int b_x=std::atoi(x.at(3).c_str());
    int b_y=std::atoi(x.at(4).c_str());
    
    std::string full_path= LABELED_IMAGE_ALL + "/" + fname;
     
    if (debug) printf("Reading H5 file %s\n", full_path.c_str());

    H5File file(full_path, H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet("/stack");
    dataset.read(data[i], H5::PredType::NATIVE_UINT32);
    file.close();


	if (debug) printf("Coarsening block %d %d %d\n", b_z, b_x, b_y);
    ///////////      Coarsening
    for (int z = 0; z < s_height; z += block_z_size) {
      for (int x = 0; x < s_cols; x += block_x_size) {
        for (int y = 0; y < s_rows; y += block_y_size) {
          int block_x_begin = x;
          int block_x_end = x+block_x_size;
          int block_y_begin = y;
          int block_y_end = y+block_y_size;
          int block_z_begin = z;
          int block_z_end = z+block_z_size;
          create_cgraph_block(c_graph, data[i], block_x_begin, block_x_end, block_y_begin, block_y_end, block_z_begin, block_z_end,
                               block_x_size, block_y_size, block_z_size, b_z*c_height, b_x*c_cols, b_y*c_rows);
        }
      }
    }
    
    if (debug) printf("Done with batch\n");
  } // cilk for 
  } // blocks for
  if (debug) printf("Max label is %d\n", max);
  
  /// Freeing up memory
  for(int i=0; i < batch_size; i++) {
    delete[] data[i];
  }
  delete[] data;

}

void loadImages_block_h5_init() {
  height = c_height;
  cols   = c_cols;
  rows   = c_rows;
}

void loadImages_block_h5() {
	std::string full_path= LABELED_IMAGE_BLOCK;
    
    if (debug) printf("Reading H5 file %s\n", full_path.c_str());

    int BLOCK_SIZE= s_rows*s_cols*s_height;
    label_t* data= new label_t[BLOCK_SIZE];
    
    H5File file(full_path, H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet("/stack");
    dataset.read(data, H5::PredType::NATIVE_UINT32);
    file.close();
    
    int b_z=0; int b_x=0; int b_y=0; // Only one block

	if (debug) printf("Coarsening block %d %d %d\n", b_z, b_x, b_y);
    ///////////      Coarsening
    cilk_for (int z = 0; z < s_height; z += block_z_size) {
      for (int x = 0; x < s_cols; x += block_x_size) {
        for (int y = 0; y < s_rows; y += block_y_size) {
          int block_x_begin = x;
          int block_x_end = x+block_x_size;
          int block_y_begin = y;
          int block_y_end = y+block_y_size;
          int block_z_begin = z;
          int block_z_end = z+block_z_size;
          create_cgraph_block(c_graph, data, block_x_begin, block_x_end, block_y_begin, block_y_end, block_z_begin, block_z_end,
                               block_x_size, block_y_size, block_z_size, b_z*c_height, b_x*c_cols, b_y*c_rows);
        }
      }
    }
    delete[] data;

}

////// png input functions
void loadImages_all_png_init() {return;}

void loadImages_all_png() {return;}

void loadImages_block_png_init() {
  height = c_height;
  cols   = c_cols;
  rows   = c_rows;
}

void loadImages_block_png() {
  DIR*    dir;
  dirent* pdir;
  dir = opendir(LABELED_IMAGE_BLOCK.c_str());
  
  b_height=0;
  b_cols=0;
  b_rows=0;
  
  std::string filenames[s_height];
  for (int z = 0; z < s_height; z++) {
    filenames[z].assign("Not initialized");
  }
  
  while (pdir = readdir(dir)) {
    if ((pdir->d_name[0]!='o') || (pdir->d_name[1]!='u') || (pdir->d_name[2]!='t')) continue;
    std::vector<std::string> parts =split(pdir->d_name, '_');
    std::size_t pos = parts.at(parts.size()-1).find("\\."); 
    int i=atoi(parts.at(parts.size()-1).substr(0,pos).c_str());

    std::string fname (pdir->d_name);
    filenames[i]=fname;
  }
       
  // load the image stack
  cilk_for (int z = 0; z < s_height; z++) {  
    if (strcmp("Not initialized", filenames[z].c_str())==0)  {
      for (int p = 0; p < s_rows * s_cols; ++p) data[z * s_rows * s_cols + p] = 0;
    }
    else {
      // Load ith level of label stack.
      std::string labels_path = LABELED_IMAGE_BLOCK + "/" + filenames[z]; 
      cv::Mat im2;
      printf("Labels path is %s\n", labels_path.c_str());
      im2 = cv::imread(labels_path.c_str(), CV_LOAD_IMAGE_UNCHANGED);
    
      //Convert it to labels
	  for (int p = 0; p < s_rows * s_cols; ++p) {
        uint32_t blu = im2.data[3 * p + 0];
        uint32_t grn = im2.data[3 * p + 1];    
        uint32_t red = im2.data[3 * p + 2]; 
    
        // opencv has bgr, we want rgb
        data[z * s_rows * s_cols + p] = ((red << 16) | (grn << 8) | blu); // keep opencv order
      }
    }
  }    
    
  int b_z=0; int b_x=0; int b_y=0; // Only one block

  if (debug) printf("Coarsening block %d %d %d\n", b_z, b_x, b_y);
   ///////////      Coarsening
  cilk_for (int z = 0; z < s_height; z += block_z_size) {
    for (int x = 0; x < s_cols; x += block_x_size) {
      for (int y = 0; y < s_rows; y += block_y_size) {
        int block_x_begin = x;
        int block_x_end = x+block_x_size;
        int block_y_begin = y;
        int block_y_end = y+block_y_size;
        int block_z_begin = z;
        int block_z_end = z+block_z_size;
        create_cgraph_block(c_graph, data, block_x_begin, block_x_end, block_y_begin, block_y_end, block_z_begin, block_z_end,
                               block_x_size, block_y_size, block_z_size, b_z*c_height, b_x*c_cols, b_y*c_rows);
      }
    }
  }
  delete[] data;
}



void loadLUT() {
  //Reading LUT
  int N=pow(2,pow(3,3));
  // Probably no need to use a separate array for this, we can just use what's mmap'd.
  simple = (bool*) malloc (N*sizeof(bool));
  FILE* in = fopen("LUT/LUT.txt","r");
  char* LUT = (char*)mmap(NULL, N*sizeof(char), PROT_READ, MAP_SHARED, fileno(in),0);
  cilk::reducer_opadd<int> sum(0);
  cilk_for(int i=0; i<N; i++) {
	//char temp;
        char temp = LUT[i];
	//fscanf(in,"%c",&temp);
	if (temp=='0')
	   simple[i]=0;
	else if (temp=='1')
	   simple[i]=1;
	else
	   printf("ERROR, %d",temp);
	if (simple[i]) {
         sum+= 1;
          //__sync_fetch_and_add(&sum, 1);
        }
  }
  fclose(in);
  if (debug) printf("LUT file read. %d simple points detected\n",sum.get_value() );

}


