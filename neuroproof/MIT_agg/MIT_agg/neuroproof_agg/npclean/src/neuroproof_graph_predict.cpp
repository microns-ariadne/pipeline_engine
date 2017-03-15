#include "../FeatureManager/FeatureMgr.h"
#include "../BioPriors/BioStack.h"

#include "../Utilities/ScopeTime.h"
#include "../Utilities/OptionParser.h"
#include "../Rag/RagIO.h"

#include "../BioPriors/StackAgglomAlgs.h"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <fstream>
#include <json/value.h>
#include <json/reader.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <unistd.h>
#include <stdio.h>


#include <cilk/cilk.h>
//#include <google/profiler.h>

namespace fs = boost::filesystem;

using namespace NeuroProof;

using std::cerr; using std::cout; using std::endl;
using std::string;
using std::vector;
using namespace boost::algorithm;
using std::tr1::unordered_set;

static const char * SEG_DATASET_NAME = "stack";
static const char * PRED_DATASET_NAME = "volume/predictions";

void remove_inclusions(Stack& stack)
{
    cout<<"Inclusion removal ..."; 
    stack.remove_inclusions();
    cout<<"done with "<< stack.get_num_labels()<< " nodes\n";	
}

struct PredictOptions
{
    PredictOptions(int argc, char** argv) : synapse_filename(""), output_filename("segmentation.h5"),
        graph_filename("graph.json"), threshold(0.2), watershed_threshold(0), post_synapse_threshold(0.0),
        merge_mito(false), agglo_type(1), enable_transforms(true), postseg_classifier_filename(""),
        location_prob(true), num_top_edges(1), rand_prior(0)
    {
        OptionParser parser("Program that predicts edge confidence for a graph and merges confident edges");

        // positional arguments
        parser.add_positional(watershed_filename, "watershed-file",
                "gala h5 file with label volume (z,y,x) and body mappings"); 
        parser.add_positional(prediction_filename, "prediction-file",
                "ilastik h5 file (x,y,z,ch) that has pixel predictions"); 
        parser.add_positional(classifier_filename, "classifier-file",
                "opencv or vigra agglomeration classifier"); 

        // optional arguments
        parser.add_option(synapse_filename, "synapse-file",
                "json file that contains synapse annotations that are used as constraints in merging"); 
        parser.add_option(output_filename, "output-file",
                "h5 file that will contain the output segmentation (z,y,x) and body mappings"); 
        parser.add_option(graph_filename, "graph-file",
                "json file that will contain the output graph"); 
        parser.add_option(threshold, "threshold",
                "segmentation threshold"); 
        parser.add_option(watershed_threshold, "watershed-threshold",
                "threshold used for removing small bodies as a post-process step"); 
        parser.add_option(postseg_classifier_filename, "postseg-classifier-file",
                "opencv or vigra agglomeration classifier to be used after agglomeration to assign confidence to the graph edges -- classifier-file used if not specified"); 
        parser.add_option(post_synapse_threshold, "post-synapse-threshold",
                "Merge synapses indepedent of constraints"); 

        // invisible arguments
        parser.add_option(num_top_edges, "num-top-edges",
                "number of top edges to look at from priority queue", true, false, true); 
        parser.add_option(rand_prior, "rand-prior",
                "randomize priority of the k top edges", true, false, true); 
        parser.add_option(merge_mito, "merge-mito",
                "perform separate mitochondrion merge phase", true, false, true); 
        parser.add_option(agglo_type, "agglo-type",
                "merge mode used", true, false, true); 
        parser.add_option(enable_transforms, "transforms",
                "enables using the transforms table when reading the segmentation", true, false, true); 
        parser.add_option(location_prob, "location_prob",
                "enables pixel prediction when choosing optimal edge location", true, false, true); 

        parser.parse_options(argc, argv);
    }

    // manadatory positionals
    string watershed_filename;
    string prediction_filename;
    string classifier_filename;
   
    // optional (with default values) 
    string synapse_filename;
    string output_filename;
    string graph_filename;

    double threshold;
    int watershed_threshold; // might be able to increase default to 500
    string postseg_classifier_filename;
    double post_synapse_threshold;

    // hidden options (with default values)
    bool merge_mito;
    int agglo_type;
    bool enable_transforms;
    bool location_prob;
    int num_top_edges;
    bool rand_prior;
};



VolumeLabelPtr return_volume_label_ptr(PredictOptions& options) {
    VolumeLabelPtr initial_labels = VolumeLabelData::create_volume(
            options.watershed_filename.c_str(), SEG_DATASET_NAME);
    return initial_labels;
}

void get_dir_files(
    string dirpath, 
    std::vector<std::string> &v_files) {
    
    printf("get_dir_files: start\n");
        
    v_files.clear();
    
    int idx = 0;
    for (fs::directory_iterator it(dirpath); it != fs::directory_iterator(); it++, idx++)
    {
        if (!fs::is_regular_file(it->status())) {
            continue;
        }
            
        string path = it->path().string();
        string ext = it->path().extension().string();
        string fileName = it->path().filename().string();
        //transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
         
        if (fileName.length() >= 1 && fileName[0] == '.') {
            continue;
        }
        
        if (ext != ".png" && ext != ".tif" && ext != ".tiff") {
            continue;
        }
        
        //printf("[%d] %s\n", idx, path.c_str());
        
        v_files.push_back(path);
    }
    

    sort(v_files.begin(), v_files.end());
    
    for (int i = 0; i < v_files.size(); i++) {
        printf("[%d] %s\n", i, v_files[i].c_str());
    }
    
    printf("get_dir_files: finish\n");
    
}

/*
 * get_json_files - read in the probability volumes and watershed input files
 *                  from a JSON document
 *
 * path - path to the JSON document
 * prob_list - a vector of probability volumes in the order expected by the
 *             classifier.
 * ws_input_files - the .PNG files containing the planes of labels for the
 *                  watershed.
 *
 * The format of the JSON file:
 * { "probabilities": 
 *      [ [ filenames for channel 0...], 
 *        [ filenames for channel 1...],
 *	 ...
 *	 [ filenames for channel N...]],
 *   "watershed": [ filenames for watershed...],
 *   "output": [ filenames for the planes to be written ]}
 *
 */
void get_json_files(std::string path, 
                    std::vector<VolumeProbPtr> &prob_list,
                    std::vector<std::string> &ws_input_files,
		    std::vector<std::string> &output_files)
{
    Json::Reader reader;
    Json::Value d;
    
    std::cout << "Using configuration, " << path << std::endl;
    ifstream fin(path);
    if (! fin) {
	throw ErrMsg("Error: input file, \"" + path + "\" cannot be opened.");
    }
    if (! reader.parse(fin, d)) {
	throw ErrMsg("Cannot parse \"" + path + "\" as json.");
    }
    fin.close();
    /*
     * Read the probabilities: a list of lists of file names
     */
    Json::Value probabilities = d["probabilities"];
    Json::Value config = d["config"];
    for (int i=0; i < probabilities.size(); i++) {
	Json::Value probability = probabilities[i];
	std::vector<std::string> file_names;
	for (int j=0; j < probability.size(); j++) {
	    file_names.push_back(probability[j].asString());
	}
	std::vector<VolumeProbPtr> tmp = VolumeProb::create_volume_from_images(
	    file_names);
	prob_list.push_back(tmp[0]);
	/*
	 * The membrane probabilities appear as the first and their inverse
         * appears as the second element.
	 */
	if (i == 0) {
	    VolumeProbPtr vInv = VolumeProb::create_volume();
	    vInv->reshape(tmp[0]->shape());
	    for (auto it=vInv->begin(); it != vInv->end(); ++it) {
	      *it = 1;
	    }
	    *vInv -= *tmp[0];
	    prob_list.push_back(vInv);
	}
    }
    /*
     * Invert probability maps other than #0 if configured to do so.
     */
    if (config.isObject()) {
	Json::Value invert = config["invert"];
	if (invert.isArray()) {
	    for (int i=0; (i<invert.size()) && (i < prob_list.size()-1); i++) {
		Json::Value bit=invert[i];
		if (bit.isBool() && bit.asBool()) {
		    std::cout << "Inverting channel # " << i << std::endl;
		    VolumeProbPtr v = prob_list[i+1];
		    VolumeProbPtr vInv = VolumeProb::create_volume();
		    vInv->reshape(v->shape());
		    for (auto it=vInv->begin(); it != vInv->end(); ++it) {
			*it = 1;
		    }
		    *vInv -= *v;
		    prob_list[i+1] = vInv;
		    std::cout << "v[100, 100, 0]=" << (*v)(100, 100, 0) <<
			       ", vInv[100, 100, 0]=" << (*vInv)(100, 100, 0) << std::endl;
		} else {
		    std::cout << "Not inverting channel # " << i << std::endl;
		}
	    }
	} else {
	    std::cout << "No invert array" << std::endl;
	}
    } else {
	std::cout << "No config object." << std::endl;
    }
    /*
     * Capture the filenames for the watershed stack.
     */
    Json::Value watershed = d["watershed"];
    for (int i=0; i < watershed.size(); i++) {
	ws_input_files.push_back(watershed[i].asString());
    }
    /*
     * Capture the output files
     */
     Json::Value output = d["output"];
     for (int i=0; i< output.size(); i++) {
	 output_files.push_back(output[i].asString());
    }
}

/*
 * compress_labels
 *
 * Rebase the labels so that they are numbered consecutively from 1
 */
void compress_labels(VolumeLabelPtr labelvol,
                     std::vector<Label_t> &mapping)
{
    std::vector<bool> has_label;
    has_label.reserve(10000000);
    for (VolumeLabelData::iterator it=labelvol->begin(); 
	 it != labelvol->end(); it++) {
	if (*it >= has_label.size()) {
	    has_label.resize(*it + 1, false);
	}
	has_label[*it] = true;
    }
    mapping.resize(has_label.size(), 0);
    int dest = 1;
    for (int src=1; src < has_label.size(); src++) {
      if (has_label[src] == true) {
	mapping[src] = dest++;
      }
    }
    std::cout << "Found " << dest << " labels" << std::endl;
}

/*
 * write_labels - write a label volume to a series of .png files
 */
void write_labels(VolumeLabelPtr labelvol, 
                  std::vector<std::string> filenames,
		  std::vector<Label_t> mapping)
{
    cilk_for (int i=0; i < labelvol->shape(2); i++) {
        cv::Mat plane(labelvol->shape(1), labelvol->shape(0), CV_8UC3);
	for (int x = 0; x < labelvol->shape(0); x++) {
	    for (int y=0; y < labelvol->shape(1); y++) {
		Label_t val = mapping[(*labelvol)(x, y, i)];
		cv::Vec3b &rgb = plane.at<Vec3b>(y, x);
		rgb[0] = (vigra::UInt8)val;
		rgb[1] = val >> 8;
		rgb[2] = val >> 16;
	    }
	}
	cv::imwrite(filenames[i].c_str(), plane);
    }
}

void run_prediction(PredictOptions& options)
{
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::ptime start_all = boost::posix_time::microsec_clock::local_time();

    std::vector<std::string> prob_input_files;
    std::vector<std::string> ws_input_files;
    std::vector<std::string> output_files;
    
    printf("-- Read dirs\n");
    
    size_t ext_pos = options.prediction_filename.find_last_of('.');
    bool is_hdf5 = false;
    bool is_json = false;
    if (ext_pos != string::npos) {
	std:string ext = options.prediction_filename.substr(ext_pos);
	is_hdf5 = ((ext == ".hdf5") || (ext == ".h5"));
	is_json = (ext == ".json");
    }
    vector<VolumeProbPtr> prob_list;
    if (is_json) {
	get_json_files(options.prediction_filename,
		       prob_list,
		       ws_input_files,
		       output_files);
    } else if (is_hdf5) {
	prob_list = VolumeProb::create_volume_array(
	    options.prediction_filename.c_str(), PRED_DATASET_NAME);
        get_dir_files(options.watershed_filename, ws_input_files);
    } else {
	get_dir_files(options.prediction_filename, prob_input_files);
        get_dir_files(options.watershed_filename, ws_input_files);
	prob_list = VolumeProb::create_volume_from_images(prob_input_files);
    }    
    VolumeLabelPtr initial_labels = cilk_spawn VolumeLabelData::create_volume_from_images_seg(ws_input_files);
    
    
    // create watershed volume from the oversegmentation file.
    EdgeClassifier* eclfr;
    if (ends_with(options.classifier_filename, ".h5")) {
        eclfr = new VigraRFclassifier(options.classifier_filename.c_str());	
    } else if (ends_with(options.classifier_filename, ".xml")) {
        eclfr = new OpencvRFclassifier(options.classifier_filename.c_str());
    }
    EdgeClassifier* back_up = eclfr->clone();
/*VolumeLabelData::create_volume(
            options.watershed_filename.c_str(), SEG_DATASET_NAME);*/
    cilk_sync;

    boost::posix_time::ptime now;
    now = boost::posix_time::microsec_clock::local_time();
    VolumeProbPtr boundary_channel = prob_list[0];
    printf("-- Read probs H5 file - finish\n"); 	
    cout << "Read prediction array" << endl;

    cout << "Read watershed" << endl;

    now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "------------------------ TIME TO LOAD DATA: " << (now - start).total_milliseconds() << " ms\n";
   
    start = boost::posix_time::microsec_clock::local_time();
    now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "---------------------- INIT CLASSIFIER: " << (now - start).total_milliseconds() << " ms\n";

 
    // TODO: move feature handling to stack (load classifier if file provided)
    // create feature manager and load classifier
    FeatureMgrPtr feature_manager(new FeatureMgr(prob_list.size()));
    feature_manager->set_basic_features(); 

    feature_manager->set_classifier(eclfr);   	 

    // create stack to hold segmentation state
    BioStack stack(initial_labels); 
    stack.set_feature_manager(feature_manager);
    stack.set_prob_list(prob_list);

    start = boost::posix_time::microsec_clock::local_time();
    cout<<"Building RAG ..."; 	
    stack.build_rag(false);
    cout<<"done with "<< stack.get_num_labels()<< " nodes\n";
    now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "---------------------- TIME TO BUILD RAG: " << (now - start).total_milliseconds() << " ms\n";

    // stack.print_rag();

    // add synapse constraints (send json to stack function)
    if (options.synapse_filename != "") {   
        stack.set_synapse_exclusions(options.synapse_filename.c_str());
    }
    
    // stack.print_fm();
    
    printf("-- remove_inclusions\n");
    
    remove_inclusions(stack);
    // stack.print_rag();
    // stack.print_fm();
    
    printf("-- AGGLO_TYPE: %d\n", options.agglo_type);
    
    start = boost::posix_time::microsec_clock::local_time();
    switch (options.agglo_type) {
        case 0: 
            cout<<"Agglomerating (flat) upto threshold "<< options.threshold<< " ..."; 
            agglomerate_stack_flat(stack, options.threshold, options.merge_mito);
            break;
        case 1:
            cout<<"Agglomerating (agglo) upto threshold "<< options.threshold<< " ..."; 
            agglomerate_stack(stack, options.threshold, options.merge_mito);
            break;        
        case 2:
            cout<<"Agglomerating (mrf) upto threshold "<< options.threshold<< " ..."; 
            agglomerate_stack_mrf(stack, options.threshold, options.merge_mito);
            break;
        case 3:
            cout<<"Agglomerating (queue) upto threshold "<< options.threshold<< " ..."; 
            agglomerate_stack_queue(stack, options.threshold, options.merge_mito);
            break;
        case 4:
            cout<<"Agglomerating (flat) upto threshold "<< options.threshold<< " ..."; 
            agglomerate_stack_flat(stack, options.threshold, options.merge_mito);
            break;
        case 5:
            cout <<"Agglomerating (parallel) upto threshold "<< options.threshold << " ...";
            agglomerate_stack_parallel(stack, options.num_top_edges, options.rand_prior, options.threshold, options.merge_mito);
            break;
        case 6:
            cout <<"Agglomerating (mrf parallel) upto threshold "<< options.threshold << " ...";
            agglomerate_stack_mrf_parallel(stack, options.num_top_edges, options.rand_prior, options.threshold, options.merge_mito);
            break;

        default: throw ErrMsg("Illegal agglomeration type specified");
    }
    now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "---------------------- TIME TO AGGLOMERATE: " << (now - start).total_milliseconds() << " ms\n";

    cout << "Done with "<< stack.get_num_labels()<< " regions\n";
   
    
    if (options.post_synapse_threshold > 0.00001) {
        cout << "Agglomerating (agglo) ignoring synapse constraints upto threshold "
            << options.post_synapse_threshold << endl;
        string dummy1, dummy2;
        agglomerate_stack(stack, options.post_synapse_threshold,
                    options.merge_mito, false, true);
        cout << "Done with "<< stack.get_num_labels() << " regions\n";
    }
    

    start = boost::posix_time::microsec_clock::local_time();
    remove_inclusions(stack);


    if (options.merge_mito) {
        cout<<"Merge Mitochondria (border-len) ..."; 
        agglomerate_stack_mito(stack);
    	cout<<"done with "<< stack.get_num_labels() << " regions\n";	

        remove_inclusions(stack);        
    } 	

    if (options.watershed_threshold > 0) {
        cout << "Removing small bodies ... ";

        unordered_set<Label_t> synapse_labels;
        stack.load_synapse_labels(synapse_labels);
        int num_removed = stack.absorb_small_regions(boundary_channel,
                        options.watershed_threshold, synapse_labels);
        cout << num_removed << " removed" << endl;	
    }
    now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "---------------------- TIME TO REMOVE INCLUSION AND SMALL BODIES: " << (now - start).total_milliseconds() << " ms\n";


    start = boost::posix_time::microsec_clock::local_time();
    
    if (options.postseg_classifier_filename == "") {
        options.postseg_classifier_filename = options.classifier_filename;
        eclfr = back_up;
    } else {
        delete eclfr;
        if (ends_with(options.postseg_classifier_filename, ".h5"))
            eclfr = new VigraRFclassifier(options.postseg_classifier_filename.c_str()); 
        else if (ends_with(options.postseg_classifier_filename, ".xml"))    
            eclfr = new OpencvRFclassifier(options.postseg_classifier_filename.c_str());            
    }

    now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "---------------------- TIME TO INIT CLASSIFIER (2nd): " << (now - start).total_milliseconds() << " ms\n";


    feature_manager->clear_features();
    feature_manager->set_classifier(eclfr);   	
    start = boost::posix_time::microsec_clock::local_time(); 
    // stack.Stack::build_rag();
    stack.build_rag(false);
    now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "---------------------- TIME TO BUILD RAG (2nd time): " << (now - start).total_milliseconds() << " ms\n";

    // add synapse constraints (send json to stack function)
    if (options.synapse_filename != "") {   
        stack.set_synapse_exclusions(options.synapse_filename.c_str());
    }
        
    start = boost::posix_time::microsec_clock::local_time();
    if (is_json) {
	VolumeLabelPtr pLabels = stack.get_labelvol();
	pLabels->rebase_labels();
	std::vector<Label_t> mapping;
	compress_labels(pLabels, mapping);
	write_labels(pLabels, output_files, mapping);
    } else {
	stack.serialize_stack(options.output_filename.c_str(),
	            options.graph_filename.c_str(), options.location_prob);
    }
    now = boost::posix_time::microsec_clock::local_time();
    cout << endl << "---------------------- TIME TO SERIALIZE: " << (now - start).total_milliseconds() << " ms\n";

    delete eclfr;
    boost::posix_time::ptime end_all = boost::posix_time::microsec_clock::local_time();
    cout << endl << "------------------------ TIME TOTAL: " << (end_all - start_all).total_milliseconds() << " ms\n";
    
    prob_list[1] = NULL;
    //ProfilerStop();
}

void testing_func () {
    cout << "TESTING CILK..." << endl;
}

/*
 * Print the value of VmHWM from /proc/<my-pid>/status
 */
void print_vmhwm() {
    pid_t pid = getpid();
    char filename[100];
    snprintf(filename, 100, "/proc/%d/status", pid);
    std::ifstream status(filename);
    string line;
    while (std::getline(status, line)) {
	size_t loc = line.find("VmHWM:");
	if (loc != std::string::npos) {
	    size_t start = loc + 8;
	    while ((line[start] < '0') || (line[start] > '9')) start++;
	    size_t end=start+1;
	    while ((line[end] >= '0') && (line[end] <= '9')) end++;
	    std::cout << line.substr(loc, end) << std::endl;
	}
    }
}

int main(int argc, char** argv) 
{
    PredictOptions options(argc, argv);
    ScopeTime timer;
    
    printf("-- NP start\n");
    // printf("Hello I added a print statement\n");
    run_prediction(options);
    // run_prediction(options);

    testing_func();
    // cilk_sync;
    
    print_vmhwm();
    printf("-= PROC SUCCESS =-\n");
    
    return 0;
}


