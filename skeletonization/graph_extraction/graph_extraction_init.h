//// Initialize global variables
LABEL_TYPE** labels_array;
LABEL_TYPE** labels2_array;
LABEL_TYPE** synapse_array;

uint8_t** input_image_stack;
uint8_t** input_image_stackc1;
uint8_t** input_image_stackc2;
uint8_t** input_image_stackc3;

int* mapping;
std::set<int>* mapping2;

static int* BFSseads;
volatile bool* BFScycle;
static int* treeSizes;
static int* objectBorders;
int distinct_labels=0;
bool* simple;

uint32_t max = 0;

int rows;
int cols;
int height;

int _height;
int _rows;
int _cols;

int b_height;
int b_rows;
int b_cols;

int BLOCK_SIZE= s_rows*s_cols*s_height;
LABEL_TYPE* data= new LABEL_TYPE[BLOCK_SIZE];


// Coarse graph.
Graph<vdata, edata>* init_coarse_graph(int rows,int cols,int height) {

  Graph<vdata, edata>* c_graph = new Graph<vdata,edata>();
  c_graph->setGridDimensions(rows, cols,
      height);
  c_graph->resize(c_graph->_rows * c_graph->_cols * c_graph->_height);
  
  return c_graph;
}