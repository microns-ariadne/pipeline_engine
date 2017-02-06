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

int c_rows=0;
int c_cols=0;
int c_height=0;
int s_rows=0;
int s_cols=0;
int s_height=0;

static std::string INPUT_TYPE = "block";
static std::string OUTPUT_IMAGE_DIR;
static std::string LABELED_IMAGE_BLOCK;
static std::string LABELED_IMAGE_ALL;
static std::string OUTPUT_IMAGE_FILENAME_PREFIX="output-stack-label";
typedef uint32_t LABEL_TYPE;

