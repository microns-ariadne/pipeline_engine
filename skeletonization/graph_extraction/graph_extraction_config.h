// Defaults
bool debug = 1; // Std output
bool doThinning=1; // Set this to one to compute skeletons
bool saveSWC=1; // Outputs skeletons to the output folder in the "SWC" directory which has to be created first
int scale=4;    // Downsampling

//Post processing
bool findCycles = 0; // Do cycle detection (need cycles folder)
bool removeBranches=1; // Pruning algorithm after skeletonization
bool removeDeg2=0; //Creates a layout graph from a skeleton (so removes nodes of degree 2)
int unscaledBS = 10;  // Minimum size for pruning
int mBS=unscaledBS/scale; // After scaling
int minCycleSize = 15; //For cycle detection

/// Image output
bool saveTIF=0; // Saves TIF files of the original data (after merging) in the output folder
bool saveCoarsened=0; //Saves the coarsened images in a tif format
int saveOne=530; // Save a coarsened image of only the bject with this label. -1 means all objects


//Coarsening factors. They should help to scale the image to an even resolution.
int block_x_size=8*scale;
int block_y_size=8*scale;
int block_z_size=1*scale;

static std::string LABELED_IMAGE_ALL =
//    "/scratch/segmentation_directory/labels_input1";
//    "testInput/labels_input1";
//      "/home/gergely/data/labels_input2"; 
//      "/afs/csail.mit.edu/u/o/odor/public_html/dummy_input"; 
//    "/mnt/disk3/armafire/datasets/K11_S1/blocks_full/K11_S1_1024x1024x100_np/";
//    "/mnt/disk3/armafire/datasets/K11_S1_debug_full/K11_S1_1024x1024x100_merge";
//      "/mnt/disk4/greg/data/K11_0to4";
//      "/mnt/disk4/greg/data/isbi/";
      "/mnt/disk4/greg/data/K11_all_h5";
static std::string LABELED_IMAGE_BLOCK = 
//     "/mnt/disk4/greg/data/isbi/out_segmentation_0000_0000_0000.h5";
//      "/mnt/disk4/greg/data/K11_0to4/out_segmentation_0004_0003_0004.h5";
      "/mnt/disk4/greg/data/K11_all_png/out_segmentation_0004_0006_0006";      
      
static std::string LABELED_IMAGE_FILENAME_EXT = ".png";
static std::string INPUT_TYPE = "block";

typedef uint32_t LABEL_TYPE;

// Path format for output file. Don't forget to make an SWC folder for the skeleton input
//static std::string OUTPUT_IMAGE_DIR = "/home/gergely/data/graph_extraction_output";
static std::string OUTPUT_IMAGE_DIR =  "/mnt/disk4/greg/data/graph_extraction_output";
static std::string OUTPUT_IMAGE_FILENAME_PREFIX = "output-stack-label";
