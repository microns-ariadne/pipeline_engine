#include "../FeatureManager/FeatureMgr.h"
#include "BioStack.h"
#include "MitoTypeProperty.h"

#include <json/value.h>
#include <json/reader.h>
#include <vector>

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#include <time.h>
#include <boost/thread/mutex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <pthread.h>
#include <sparsehash/dense_hash_set>
#include <map>

using std::tr1::unordered_set;
using std::tr1::unordered_map;
using std::vector;


#define BLOCK_SIZE_LIMIT 50
namespace NeuroProof {

// boost::mutex global_labels_set_mu;
// boost::mutex mu;
// boost::mutex labelvol_mu;


void move_node_feature (FeatureMgrPtr fm1, FeatureMgrPtr fm2, RagNode_t* node1, RagNode_t* node2);
void move_edge_feature (FeatureMgrPtr fm1, FeatureMgrPtr fm2, RagEdge_t* edge1, RagEdge_t* edge2);
void merge_node_features (FeatureMgrPtr fm1, FeatureMgrPtr fm2, RagNode_t* node1, RagNode_t* node2);
void merge_edge_features (FeatureMgrPtr fm1, FeatureMgrPtr fm2, RagEdge_t* edge1, RagEdge_t* edge2);
void merge_mito_probs(unordered_map<Label_t, MitoTypeProperty> &prob1, unordered_map<Label_t, MitoTypeProperty> &prob2);
void merge_rag_list_recurse (RagPtr rag_list [], FeatureMgrPtr fm_list [], int start, int end);
void merge_mito_prob_list_recurse (unordered_map<Label_t, MitoTypeProperty> prob_list [], int start, int end);

VolumeLabelPtr BioStack::create_syn_label_volume()
{
    if (!labelvol) {
        throw ErrMsg("No label volume defined for stack");
    }

    return create_syn_volume(labelvol);
}

VolumeLabelPtr BioStack::create_syn_gt_label_volume()
{
    if (!gt_labelvol) {
        throw ErrMsg("No GT label volume defined for stack");
    }

    return create_syn_volume(gt_labelvol);
}

VolumeLabelPtr BioStack::create_syn_volume(VolumeLabelPtr labelvol2)
{
    vector<Label_t> labels;

    for (int i = 0; i < synapse_locations.size(); ++i) {
        Label_t label = (*labelvol2)(synapse_locations[i][0],
            synapse_locations[i][1], synapse_locations[i][2]); 
        labels.push_back(label);
    }
    
    VolumeLabelPtr synvol = VolumeLabelData::create_volume();   
    synvol->reshape(VolumeLabelData::difference_type(labels.size(), 1, 1));

    for (int i = 0; i < labels.size(); ++i) {
        synvol->set(i, 0, 0, labels[i]);  
    }
    return synvol;
}

void BioStack::load_saved_synapse_counts(unordered_map<Label_t, int>& synapse_counts)
{
    saved_synapse_counts = synapse_counts;
}

void BioStack::load_synapse_counts(unordered_map<Label_t, int>& synapse_counts)
{
    for (int i = 0; i < synapse_locations.size(); ++i) {
        Label_t body_id = (*labelvol)(synapse_locations[i][0],
                synapse_locations[i][1], synapse_locations[i][2]);

        if (body_id) {
            synapse_counts[body_id]++;
        }
    }
}

void BioStack::load_synapse_labels(unordered_set<Label_t>& synapse_labels)
{
    for (int i = 0; i < synapse_locations.size(); ++i) {
        Label_t body_id = (*labelvol)(synapse_locations[i][0],
                synapse_locations[i][1], synapse_locations[i][2]);
        synapse_labels.insert(body_id);
    }
}

void BioStack::read_prob_list(std::string prob_filename, std::string dataset_name)
{
    prob_list = VolumeProb::create_volume_array(prob_filename.c_str(), dataset_name.c_str());
    cout << "Read prediction array" << endl;  
    
}

bool BioStack::is_mito(Label_t label)
{
    RagNode_t* rag_node = rag->find_rag_node(label);

    MitoTypeProperty mtype;
    try {
        mtype = rag_node->get_property<MitoTypeProperty>("mito-type");
    } catch (ErrMsg& msg) {
    }

    if ((mtype.get_node_type()==2)) {	
        return true;
    }
    return false;
}
void BioStack::set_classifier()
{
    assert(feature_manager);
    EdgeClassifier* eclfr = new OpencvRFclassifier();	
    feature_manager->set_classifier(eclfr);
}

void BioStack::save_classifier(std::string clfr_name)
{
    assert(feature_manager);
    feature_manager->get_classifier()->save_classifier(clfr_name.c_str());
}



inline void insert_rag_edge (RagPtr rag, Label_t label1, Label_t label2, FeatureMgrPtr feature_man, vector<double> &predictions) {
    RagNode_t * node1 = rag->find_rag_node(label1);
    if (!node1) {
        node1 = rag->insert_rag_node(label1);
    }
    
    RagNode_t * node2 = rag->find_rag_node(label2);
    if (!node2) {
        node2 = rag->insert_rag_node(label2);
    }
   
    assert(node1 != node2);

    RagEdge_t* edge = rag->find_rag_edge(node1, node2);
    if (!edge) {
        edge = rag->insert_rag_edge(node1, node2);
    }

    if (feature_man) {
        feature_man->add_val(predictions, edge);
    }

    edge->incr_size();
}



#define TFK_ARRAY_ACCESS(array, stride, x,y,z) array[stride[0]*(x)+stride[1]*(y)+stride[2]*(z)]

void BioStack::build_rag_loop(RagPtr &rag, FeatureMgrPtr &feature_man, std::tr1::unordered_map<Label_t, MitoTypeProperty> &mito_probs, 
                                            int x_start, int x_end, int y_start, int y_end, int z_start, int z_end, bool use_mito_prob)
{
    //unordered_set<Label_t> labels;
    google::dense_hash_set<Label_t> labels;
    labels.set_empty_key(0);
    vector<double> predictions(prob_list.size(), 0.0);
    unsigned int maxx = get_xsize() - 1; 
    unsigned int maxy = get_ysize() - 1; 
    unsigned int maxz = get_zsize() - 1; 


    Label_t* label_vol_array = labelvol->data();
    vigra::Shape3 label_vol_stride = labelvol->stride(); 
   

    std::map<Label_t, RagNode_t* > label_to_map_cache;
    std::map<Label_t, std::vector<double> > label_to_val_map;
    int prob_list_size = prob_list.size();
    for (int z = z_start; z < z_end; z++) {
        for (int y = y_start; y < y_end; y++) {
            for (int x = x_start; x < x_end; x++) {
                Label_t label = TFK_ARRAY_ACCESS(label_vol_array, label_vol_stride, x, y, z);
                if (!label) continue;
                if (label_to_map_cache.find(label) == label_to_map_cache.end()) {
                  RagNode_t * node = rag->find_rag_node(label);
                  if (!node) {
                      node =  rag->insert_rag_node(label); 
                  }
                  label_to_map_cache[label] = node;
                 }

                 //for (unsigned int i = 0; i < prob_list_size; ++i) {
                 label_to_val_map[label].push_back((*(prob_list[0]))(x,y,z)); 
                 //}
            }
        }
    }

   for(std::map<Label_t,std::vector<double> >::iterator iter = label_to_val_map.begin();
       iter != label_to_val_map.end(); ++iter) {
     Label_t label = iter->first;
     if (feature_man) {
       feature_man->add_val_batch(label_to_val_map[label], label_to_map_cache[label], prob_list_size);     }
   }
 

    for (int z = z_start; z < z_end; z++) {
        for (int y = y_start; y < y_end; y++) {
            for (int x = x_start; x < x_end; x++) {
                Label_t label = TFK_ARRAY_ACCESS(label_vol_array, label_vol_stride, x, y, z); 
                
                if (!label) {
                    continue;
                }
                
                Label_t label2 = 0, label3 = 0, label4 = 0, label5 = 0, label6 = 0, label7 = 0;
                if (x > 0) label2 = TFK_ARRAY_ACCESS(label_vol_array, label_vol_stride, x-1,y,z);
                if (x < maxx) label3 = TFK_ARRAY_ACCESS(label_vol_array, label_vol_stride, x+1,y,z);
                if (y > 0) label4 = TFK_ARRAY_ACCESS(label_vol_array, label_vol_stride,x,y-1,z);
                if (y < maxy) label5 = TFK_ARRAY_ACCESS(label_vol_array, label_vol_stride,x,y+1,z);
                if (z > 0) label6 = TFK_ARRAY_ACCESS(label_vol_array, label_vol_stride,x,y,z-1);
                if (z < maxz) label7 = TFK_ARRAY_ACCESS(label_vol_array,label_vol_stride, x,y,z+1);

                for (unsigned int i = 0; i < prob_list.size(); ++i) {
                    predictions[i] = (*(prob_list[i]))(x,y,z);
                }

                 
                //RagNode_t * node = rag->find_rag_node(label);
                RagNode_t * node = label_to_map_cache[label];

                //if (!node) {
                //    node =  rag->insert_rag_node(label); 
                //}

                node->incr_size();
            
                //if (feature_man) {
                //    feature_man->add_val(predictions, node);
                //}

                if (use_mito_prob)
                    mito_probs[label].update(predictions); 


                if (label2 && (label != label2)) {
                    // rag_add_edge(label, label2, predictions);
                    insert_rag_edge(rag, label, label2, feature_man, predictions);
                    labels.insert(label2);
                }
                if (label3 && (label != label3) && (labels.find(label3) == labels.end())) {
                    // rag_add_edge(label, label3, predictions);
                    insert_rag_edge(rag, label, label3, feature_man, predictions);
                    labels.insert(label3);
                }
                if (label4 && (label != label4) && (labels.find(label4) == labels.end())) {
                    // rag_add_edge(label, label4, predictions);
                    insert_rag_edge(rag, label, label4, feature_man, predictions);
                    labels.insert(label4);
                }
                if (label5 && (label != label5) && (labels.find(label5) == labels.end())) {
                    // rag_add_edge(label, label5, predictions);
                    insert_rag_edge(rag, label, label5, feature_man, predictions);
                    labels.insert(label5);
                }
                if (label6 && (label != label6) && (labels.find(label6) == labels.end())) {
                    // rag_add_edge(label, label6, predictions);
                    insert_rag_edge(rag, label, label6, feature_man, predictions);
                    labels.insert(label6);
                }
                if (label7 && (label != label7) && (labels.find(label7) == labels.end())) {
                    // rag_add_edge(label, label7, predictions);
                    insert_rag_edge(rag, label, label7, feature_man, predictions);
                }

                if (!label2 || !label3 || !label4 || !label5 || !label6 || !label7) {
                    node->incr_boundary_size();
                }
                labels.clear();        
            }
        }
    }
}





void BioStack::cilk_build_rag_loop(RagPtr &ret_rag, FeatureMgrPtr &ret_feature_man, std::tr1::unordered_map<Label_t, MitoTypeProperty> &ret_mito_probs, 
                                            int x_start, int x_end, int y_start, int y_end, int z_start, int z_end, bool use_mito_prob)
{
    unsigned int maxx = get_xsize() - 1; 
    unsigned int maxy = get_ysize() - 1; 
    unsigned int maxz = get_zsize() - 1; 
    int nworkers = __cilkrts_get_nworkers(); // CILK_NWORKERS
    

    FeatureMgrPtr  fm_list [nworkers];
    unordered_map<Label_t, MitoTypeProperty> mitop_list [nworkers];
    RagPtr rag_list [nworkers];

    fm_list[0] = ret_feature_man;
    rag_list[0] = ret_rag;
    mitop_list[0] = ret_mito_probs;

    for (int i = 1; i < nworkers; ++i) {
        fm_list[i] = FeatureMgrPtr(new FeatureMgr(prob_list.size()));
        fm_list[i]->set_basic_features();
        rag_list[i] = RagPtr(new Rag_t());
    }

    cilk_for (int z = z_start; z < z_end; z++) {
    // for (int z = z_start; z < z_end; z++) {
        int worker_id = __cilkrts_get_worker_number();
        RagPtr rag = rag_list[worker_id];
        FeatureMgrPtr feature_man = fm_list[worker_id];
        // unordered_map<Label_t, MitoTypeProperty> mito_probs = mitop_list[worker_id];
        unordered_set<Label_t> labels;
        vector<double> predictions(prob_list.size(), 0.0);        

        for (int y = y_start; y < y_end; y++) {
            for (int x = x_start; x < x_end; x++) {
                
                Label_t label = (*labelvol)(x,y,z); 
                
                if (!label) {
                    continue;
                }


                
                Label_t label2 = 0, label3 = 0, label4 = 0, label5 = 0, label6 = 0, label7 = 0;
                if (x > 0) label2 = (*labelvol)(x-1,y,z);
                if (x < maxx) label3 = (*labelvol)(x+1,y,z);
                if (y > 0) label4 = (*labelvol)(x,y-1,z);
                if (y < maxy) label5 = (*labelvol)(x,y+1,z);
                if (z > 0) label6 = (*labelvol)(x,y,z-1);
                if (z < maxz) label7 = (*labelvol)(x,y,z+1);
                


                for (unsigned int i = 0; i < prob_list.size(); ++i) {
                    predictions[i] = (*(prob_list[i]))(x,y,z);
                }

                
                RagNode_t * node = rag->find_rag_node(label);

                if (!node) {
                    node =  rag->insert_rag_node(label); 
                }

                node->incr_size();

            
                if (feature_man) {
                    feature_man->add_val(predictions, node);
                }

                if (use_mito_prob) {
                    // mito_probs[label].update(predictions); 
                    mitop_list[worker_id][label].update(predictions); 
                }


                if (label2 && (label != label2)) {
                    // rag_add_edge(label, label2, predictions);
                    insert_rag_edge(rag, label, label2, feature_man, predictions);
                    labels.insert(label2);
                }
                if (label3 && (label != label3) && (labels.find(label3) == labels.end())) {
                    // rag_add_edge(label, label3, predictions);
                    insert_rag_edge(rag, label, label3, feature_man, predictions);
                    labels.insert(label3);
                }
                if (label4 && (label != label4) && (labels.find(label4) == labels.end())) {
                    // rag_add_edge(label, label4, predictions);
                    insert_rag_edge(rag, label, label4, feature_man, predictions);
                    labels.insert(label4);
                }
                if (label5 && (label != label5) && (labels.find(label5) == labels.end())) {
                    // rag_add_edge(label, label5, predictions);
                    insert_rag_edge(rag, label, label5, feature_man, predictions);
                    labels.insert(label5);
                }
                if (label6 && (label != label6) && (labels.find(label6) == labels.end())) {
                    // rag_add_edge(label, label6, predictions);
                    insert_rag_edge(rag, label, label6, feature_man, predictions);
                    labels.insert(label6);
                }
                if (label7 && (label != label7) && (labels.find(label7) == labels.end())) {
                    // rag_add_edge(label, label7, predictions);
                    insert_rag_edge(rag, label, label7, feature_man, predictions);
                }

                if (!label2 || !label3 || !label4 || !label5 || !label6 || !label7) {
                    node->incr_boundary_size();
                }
                labels.clear();    
            

                // for (set<Label_t>::iterator it = neighbors.begin(); it != neighbors.end(); ++it) {
                //     if ((*it != label) && (labels.find(*it) == labels.end())) {
                //         labels.insert(*it);
                //         // rag_add_edge(label, *it, predictions);
                //         RagNode_t * node1 = rag->find_rag_node(label);
                //         if (!node1) {
                //             node1 = rag->insert_rag_node(label);
                //         }
                        
                //         RagNode_t * node2 = rag->find_rag_node(*it);
                //         if (!node2) {
                //             node2 = rag->insert_rag_node(*it);
                //         }
                       
                //         assert(node1 != node2);

                //         RagEdge_t* edge = rag->find_rag_edge(node1, node2);
                //         if (!edge) {
                //             edge = rag->insert_rag_edge(node1, node2);
                //         }

                //         if (feature_man) {
                //             feature_man->add_val(predictions, edge);
                //         }

                //         edge->incr_size();
                //     }
                // } 
                // labels.clear();          
            }
        }
    }

    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();

    merge_rag_list_recurse(rag_list, fm_list, 0, nworkers-1);

    if (use_mito_prob) {
        merge_mito_prob_list_recurse(mitop_list, 0, nworkers-1);
        ret_mito_probs = mitop_list[0];
    }

    boost::posix_time::ptime end = boost::posix_time::microsec_clock::local_time();

    cout << endl << "------------------------ TIME TO MERGE: " << (end - start).total_milliseconds() << " ms\n";
}







void move_node_feature (FeatureMgrPtr fm1, FeatureMgrPtr fm2, RagNode_t* node1, RagNode_t* node2) {
    // cout << "Move Node" << endl;
    node1->set_size(node2->get_size());
    node1->set_boundary_size(node2->get_boundary_size());
    NodeCaches &nc1 = fm1->get_node_cache();
    NodeCaches &nc2 = fm2->get_node_cache();
    NodeCaches::iterator node_feat2 = nc2.find(node2);
    // assert(node_feat2 != nc2.end());
    if (node_feat2 == nc2.end())
        return;
    nc1[node1] = nc2[node2];
    nc2[node2] = std::vector<void *>();
    // fm1->add_val(0.0, node1);
    // merge_node_features(fm1, fm2, node1, node2);
}









inline void move_edge_feature (FeatureMgrPtr fm1, FeatureMgrPtr fm2, RagEdge_t* edge1, RagEdge_t* edge2) {
    // cout << "Move Edge" << endl;
    edge1->set_size(edge2->get_size()); 
    EdgeCaches &ec1 = fm1->get_edge_cache();
    EdgeCaches &ec2 = fm2->get_edge_cache();
    EdgeCaches::iterator edge_feat2 = ec2.find(edge2);
    assert(edge_feat2 != ec2.end());
    ec1[edge1] = ec2[edge2];
    ec2[edge2] = std::vector<void *>();
    // fm1->add_val(0.0, edge1);
    // merge_edge_features(fm1, fm2, edge1, edge2);
}











void merge_node_features (FeatureMgrPtr fm1, FeatureMgrPtr fm2, RagNode_t* node1, RagNode_t* node2) {
    // cout << "Merge Node" << endl;
    NodeCaches &nc1 = fm1->get_node_cache();
    NodeCaches &nc2 = fm2->get_node_cache();
    if (nc2.find(node2) == nc2.end()) {
        return;
    }
    if (nc1.find(node1) == nc1.end()) {
        move_node_feature(fm1, fm2, node1, node2);
        return;
    }
    node1->incr_size(node2->get_size());
    node1->incr_boundary_size(node2->get_boundary_size());
    unsigned int pos = 0;
    unsigned int num_chan = fm1->get_num_channels();
    // cout << "Node 2: " << node2->get_node_id() << endl;
    for (int i = 0; i < num_chan; ++i) {
        vector<FeatureCompute*>& features = fm1->get_channel_features()[i];
        for (int j = 0; j < features.size(); ++j) {
            if (nc1[node1][pos] && nc2[node2][pos]) {
                features[j]->merge_cache(nc1[node1][pos], nc2[node2][pos], false);
            }
            ++pos;
        }
    }
}







void merge_edge_features (FeatureMgrPtr fm1, FeatureMgrPtr fm2, RagEdge_t* edge1, RagEdge_t* edge2) {
    // cout << "Merge Edge" << endl;
    edge1->incr_size(edge2->get_size());
    EdgeCaches &ec1 = fm1->get_edge_cache();
    EdgeCaches &ec2 = fm2->get_edge_cache();
    unsigned int pos = 0;
    unsigned int num_chan = fm1->get_num_channels();
    for (int i = 0; i < num_chan; ++i) {
        vector<FeatureCompute*>& features = fm1->get_channel_features()[i];
        for (int j = 0; j < features.size(); ++j) {
            if (ec1[edge1][pos] && ec2[edge2][pos]) {
                features[j]->merge_cache(ec1[edge1][pos], ec2[edge2][pos], false);
            }
            ++pos;
        }
    }    
}








void merge_rags_new (RagPtr &rag1, RagPtr &rag2, FeatureMgrPtr fm1, FeatureMgrPtr fm2) {
    EdgeCaches &ec1 = fm1->get_edge_cache();
    EdgeCaches &ec2 = fm2->get_edge_cache();
    NodeCaches &nc1 = fm1->get_node_cache();
    NodeCaches &nc2 = fm2->get_node_cache();

    for (Rag_t::nodes_iterator it1 = rag2->nodes_begin(); it1 != rag2->nodes_end(); ++it1) {
        RagNode_t * node1 = rag1->find_rag_node((*it1)->get_node_id());
        node1->incr_size((*it1)->get_size());
        merge_node_features(fm1, fm2, node1, *it1);
    }

    for (Rag_t::edges_iterator it1 = rag2->edges_begin(); it1 != rag2->edges_end(); ++it1) {
        RagEdge_t * edge1 = rag1->find_rag_edge((*it1)->get_node1(), (*it1)->get_node2());
        edge1->incr_size((*it1)->get_size());
        merge_edge_features(fm1, fm2, edge1, *it1);
    }
}









void merge_rags (RagPtr &rag1, RagPtr &rag2, FeatureMgrPtr fm1, FeatureMgrPtr fm2) {
    set<Label_t> processed;
    for (Rag_t::nodes_iterator it1 = rag2->nodes_begin(); it1 != rag2->nodes_end(); ++it1) {
        assert(processed.find((*it1)->get_node_id()) == processed.end());
        RagNode_t * node1 = rag1->find_rag_node((*it1)->get_node_id());
        if (!node1) {
            // if not, insert the node and its incident edges
            RagNode_t* new_node = rag1->insert_rag_node((*it1)->get_node_id());
            move_node_feature(fm1, fm2, new_node, *it1);
            // assert(fm1->get_node_cache().find(new_node) != fm1->get_node_cache().end());
            for (RagNode_t::edge_iterator it2 = (*it1)->edge_begin(); it2 != (*it1)->edge_end(); ++it2) {
                RagNode_t* terminal_node = (*it2)->get_other_node(*it1);
                node1 = rag1->find_rag_node(terminal_node->get_node_id());
                if (node1) {
                    // add edge and update node
                    RagEdge_t* new_edge = rag1->insert_rag_edge(node1, new_node);
                    move_edge_feature(fm1, fm2, new_edge, *it2);
                    // assert(fm1->get_edge_cache().find(new_edge) != fm1->get_edge_cache().end());
                }
            }
        } else {
            // merge size. go thru neighbors. if not processed and in rag1, update edge.
            assert(processed.find(node1->get_node_id()) == processed.end());
            merge_node_features(fm1, fm2, node1, *it1);
            for (RagNode_t::edge_iterator it2 = (*it1)->edge_begin(); it2 != (*it1)->edge_end(); ++it2) {
                RagNode_t* terminal_node = (*it2)->get_other_node(*it1);
                RagNode_t* node2 = rag1->find_rag_node(terminal_node->get_node_id()); 
                if (node2) {
                    if (processed.find(node2->get_node_id()) == processed.end()) {
                        RagEdge_t* new_edge = rag1->find_rag_edge(node1->get_node_id(), node2->get_node_id());
                        if (new_edge) {
                            // if edge is already there
                            merge_edge_features(fm1, fm2, new_edge, *it2);
                        } else {
                            new_edge = rag1->insert_rag_edge(node1, node2);
                            move_edge_feature(fm1, fm2, new_edge, *it2);
                        }
                    }
                }
            }
        }
        processed.insert((*it1)->get_node_id());
    }
}







void merge_feature_managers (FeatureMgrPtr fm1, FeatureMgrPtr fm2) {
    // absorb fm2 into fm1
    EdgeCaches &ec1 = fm1->get_edge_cache();
    EdgeCaches &ec2 = fm2->get_edge_cache();
    for (EdgeCaches::iterator it = ec2.begin(); it != ec2.end(); ++it) {
        EdgeCaches::iterator found = ec1.find(it->first);
        if (found != ec1.end()) {
            unsigned int pos = 0;
            unsigned int num_chan = fm1->get_num_channels();
            for (int i = 0; i < num_chan; ++i) {
                vector<FeatureCompute*>& features = fm1->get_channel_features()[i];
                for (int j = 0; j < features.size(); ++j) {
                    if ((found->second)[pos] && (it->second)[pos]) {
                        features[j]->merge_cache((found->second)[pos], (it->second)[pos], false);
                    }
                    ++pos;
                }
            }
        } else {
            RagEdge_t* new_edge = RagEdge_t::New(*it->first);
            ec1[new_edge] = it->second;
            it->second = std::vector<void *>();
        }
    }
}









void merge_mito_prob_list_recurse (unordered_map<Label_t, MitoTypeProperty> prob_list [], int start, int end) {
    if (end == start)
        return;
    if (end - start == 1) {
        merge_mito_probs (prob_list[start], prob_list[end]);
        return;
    }

    int mid = (start + end) / 2;
    merge_mito_prob_list_recurse (prob_list, start, mid);
    merge_mito_prob_list_recurse (prob_list, mid + 1, end);
    merge_mito_probs (prob_list[start], prob_list[mid + 1]);
    return;
}






void merge_rag_list_recurse (RagPtr rag_list [], FeatureMgrPtr fm_list [], int start, int end) {
    if (end == start)
        return;
    if (end - start == 1) {
        merge_rags (rag_list[start], rag_list[end], fm_list[start], fm_list[end]);
        return;
    }

    int mid = (start + end) / 2;
    merge_rag_list_recurse (rag_list, fm_list, start, mid);
    merge_rag_list_recurse (rag_list, fm_list, mid + 1, end);
    merge_rags (rag_list[start], rag_list[mid + 1], fm_list[start], fm_list[mid + 1]);
    return;
}








void print_this_fm(FeatureMgrPtr fm) {
    NodeCaches nodes = fm->get_node_cache();
    EdgeCaches edges = fm->get_edge_cache();

    cout << "EDGE SIZE: " << edges.size() << endl;    
    cout << "NODE SIZE: " << nodes.size() << endl;  

    cout << "Node Features: " << endl;
    for (NodeCaches::iterator it = nodes.begin(); it != nodes.end(); ++it) {
        fm->print_cache(it->first);
    }

    cout << "Edge Features: " << endl;
    for (EdgeCaches::iterator it = edges.begin(); it != edges.end(); ++it) {
        fm->print_cache(it->first);
    }
}








void BioStack::print_fm() {
    NodeCaches nodes = feature_manager->get_node_cache();
    EdgeCaches edges = feature_manager->get_edge_cache();

    cout << "EDGE SIZE: " << edges.size() << endl;    
    cout << "NODE SIZE: " << nodes.size() << endl;

    cout << "Node Features: " << endl;
    for (NodeCaches::iterator it = nodes.begin(); it != nodes.end(); ++it) {
        feature_manager->print_cache(it->first);
    }
    cout << "Edge Features: " << endl;
    for (EdgeCaches::iterator it = edges.begin(); it != edges.end(); ++it) {
        feature_manager->print_cache(it->first);
    }
}








void print_this_rag(RagPtr rag) {
    cout << endl;
    cout << "RAG SIZE:        " << rag->get_rag_size() << endl;
    cout << "RAG NUM REGIONS: " << rag->get_num_regions() << endl;
    cout << "RAG NUM EDGES:   " << rag->get_num_edges() << endl;
    map<Label_t, vector<Label_t> > my_map;
    for (Rag_t::nodes_iterator it1 = rag->nodes_begin(); it1 != rag->nodes_end(); ++it1) {
        my_map[(*it1)->get_node_id()] = vector<Label_t>();  
        for (RagNode_t::edge_iterator it2 = (*it1)->edge_begin(); it2 != (*it1)->edge_end(); ++it2) {
            RagNode_t* terminal_node = (*it2)->get_other_node(*it1);
            RagNode_t* node2 = rag->find_rag_node(terminal_node->get_node_id());
            RagEdge_t* new_edge = rag->find_rag_edge((*it1), node2);
            if (new_edge) {
                assert(node2 == terminal_node);
                my_map[(*it1)->get_node_id()].push_back(node2->get_node_id());
            }
            else
                my_map[(*it1)->get_node_id()].push_back(9999);
        }
    }
    for (map<Label_t, vector<Label_t> >::iterator it1 = my_map.begin(); it1 != my_map.end(); ++it1) {
        cout << "NODE " << it1->first << ": " << "Boundary Size: " << rag->find_rag_node(it1->first)->get_boundary_size() << ", ";
        sort (it1->second.begin(), it1->second.end());
        for (vector<Label_t>::iterator it2 = it1->second.begin(); it2 != it1->second.end(); ++it2) 
            cout << *it2 << " ";
        cout << endl;
    }
}







void BioStack::print_rag() {
    cout << endl;
    cout << "RAG SIZE:        " << rag->get_rag_size() << endl;
    cout << "RAG NUM REGIONS: " << rag->get_num_regions() << endl;
    cout << "RAG NUM EDGES:   " << rag->get_num_edges() << endl;
    map<Label_t, vector<Label_t> > my_map;
    for (Rag_t::nodes_iterator it1 = rag->nodes_begin(); it1 != rag->nodes_end(); ++it1) {
        my_map[(*it1)->get_node_id()] = vector<Label_t>();  
        for (RagNode_t::edge_iterator it2 = (*it1)->edge_begin(); it2 != (*it1)->edge_end(); ++it2) {
            RagNode_t* terminal_node = (*it2)->get_other_node(*it1);
            RagNode_t* node2 = rag->find_rag_node(terminal_node->get_node_id());
            RagEdge_t* new_edge = rag->find_rag_edge((*it1), node2);
            assert(new_edge == *it2);
            if (new_edge) {
                assert(node2 == terminal_node);
                my_map[(*it1)->get_node_id()].push_back(node2->get_node_id());
            }
            else
                my_map[(*it1)->get_node_id()].push_back(9999);
        }
    }
    for (map<Label_t, vector<Label_t> >::iterator it1 = my_map.begin(); it1 != my_map.end(); ++it1) {
        cout << "NODE " << it1->first << ": " << "Boundary Size: " << rag->find_rag_node(it1->first)->get_boundary_size() << ", ";
        sort (it1->second.begin(), it1->second.end());
        for (vector<Label_t>::iterator it2 = it1->second.begin(); it2 != it1->second.end(); ++it2) 
            cout << *it2 << " ";
        cout << endl;
    }
}






void merge_mito_probs(unordered_map<Label_t, MitoTypeProperty> &prob1, unordered_map<Label_t, MitoTypeProperty> &prob2) {
    for (unordered_map<Label_t, MitoTypeProperty>::iterator it = prob2.begin(); it != prob2.end(); ++it) {
        prob1[it->first].merge(it->second);
    }
}








void BioStack::build_rag_recurse (RagPtr &rag1, FeatureMgrPtr &fm1, unordered_map<Label_t, MitoTypeProperty> &mito_prob1, 
                                                    int x_start, int x_end, int y_start, int y_end, int z_start, int z_end, bool use_mito_prob) {
    int x_size = x_end - x_start;
    int y_size = y_end - y_start;
    int z_size = z_end - z_start;

    RagPtr rag2 = RagPtr(new Rag_t());
    FeatureMgrPtr fm2 = FeatureMgrPtr(new FeatureMgr(prob_list.size()));
    fm2->set_basic_features(); 
    unordered_map<Label_t, MitoTypeProperty> mito_prob2;

    bool stop_recurse = false;

    if (x_size >= y_size && x_size >= z_size) {
        if (x_size > BLOCK_SIZE_LIMIT) {
            cilk_spawn build_rag_recurse(rag1, fm1, mito_prob1, x_start, x_start + x_size/2, y_start, y_end, z_start, z_end, use_mito_prob);
            build_rag_recurse(rag2, fm2, mito_prob2, x_start + x_size/2, x_end, y_start, y_end, z_start, z_end, use_mito_prob);
            cilk_sync;
        } else {
            stop_recurse = true;
        }
    } else if (y_size >= x_size && y_size >= z_size) {
        if (y_size > BLOCK_SIZE_LIMIT) {
            cilk_spawn build_rag_recurse(rag1, fm1, mito_prob1, x_start, x_end, y_start, y_start + y_size/2, z_start, z_end, use_mito_prob);
            build_rag_recurse(rag2, fm2, mito_prob2, x_start, x_end, y_start + y_size/2, y_end, z_start, z_end, use_mito_prob);
            cilk_sync;
        } else {
            stop_recurse = true;
        }
    } else {
        if (z_size > BLOCK_SIZE_LIMIT) {
            cilk_spawn build_rag_recurse(rag1, fm1, mito_prob1, x_start, x_end, y_start, y_end, z_start, z_start + z_size/2, use_mito_prob);
            build_rag_recurse(rag2, fm2, mito_prob2, x_start, x_end, y_start, y_end, z_start + z_size/2, z_end, use_mito_prob);
            cilk_sync;
        } else {
            stop_recurse = true;
        }
    }

    if (stop_recurse) {
        build_rag_loop(rag1, fm1, mito_prob1, x_start, x_end, y_start, y_end, z_start, z_end, use_mito_prob);
    } else {
        merge_rags(rag1, rag2, fm1, fm2);
        if (use_mito_prob)
            merge_mito_probs (mito_prob1, mito_prob2);
    }
}









void BioStack::build_rag(bool use_mito_prob)
{
    if (use_mito_prob) {
        if (get_prob_list().size()==0){
            Stack::build_rag();
            return;
        }

        if (!feature_manager){
            FeatureMgrPtr feature_manager_(new FeatureMgr(prob_list.size()));
            set_feature_manager(feature_manager_);
            feature_manager->set_basic_features(); 
        }
    }

    
    //printf("Building bioStack rag\n");
    if (!labelvol) {
        throw ErrMsg("No label volume defined for stack");
    }

    rag = RagPtr(new Rag_t);
   
    unsigned int maxx = get_xsize() - 1; 
    unsigned int maxy = get_ysize() - 1; 
    unsigned int maxz = get_zsize() - 1; 
    unordered_map<Label_t, MitoTypeProperty> mito_probs;
 
    // // For testing
    // int x_full = (int)(*labelvol).shape(0)/2;
    // int y_full = (int)(*labelvol).shape(1)/4;
    // int z_full = (int)(*labelvol).shape(2)/4;

    int x_full = (int)(*labelvol).shape(0);
    int y_full = (int)(*labelvol).shape(1);
    int z_full = (int)(*labelvol).shape(2);


    int z_half = z_full/2;
    int z_fourth = z_full/4;
    int z_eighth = z_full/8;
    int z_three_fourths = z_half + z_fourth;
    int z_three_eighths = z_fourth + z_eighth;
    int z_five_eighths = z_half + z_eighth;
    int z_seven_eighths = z_three_fourths + z_eighth;

    int y_half = y_full/2;
    int y_fourth = y_full/4;

    int x_half = x_full/2;

    int nworkers = 8;

    FeatureMgrPtr fm_list [nworkers];
    unordered_map<Label_t, MitoTypeProperty> mitop_list [nworkers];
    RagPtr rag_list [nworkers];

    fm_list[0] = feature_manager;
    rag_list[0] = rag;

    for (int i = 1; i < nworkers; i++) {
        fm_list[i] = FeatureMgrPtr(new FeatureMgr(prob_list.size()));    
        fm_list[i]->set_basic_features();
        rag_list[i] = RagPtr(new Rag_t());
    }


    // =================================== HARDCODE ================================================

    // // cilk_spawn build_rag_loop(rag, feature_manager, mito_probs, 0, x_full, 0, y_half, 0, z_fourth);
    // // cilk_spawn build_rag_loop(rag2, feature_manager2, mito_probs2, 0, x_full, y_half, y_full, 0, z_fourth);
    // // cilk_spawn build_rag_loop(rag3, feature_manager3, mito_probs3, 0, x_full, 0, y_half, z_fourth, z_half);
    // // cilk_spawn build_rag_loop(rag4, feature_manager4, mito_probs4, 0, x_full, y_half, y_full, z_fourth, z_half);
    // // cilk_spawn build_rag_loop(rag5, feature_manager5, mito_probs5, 0, x_full, 0, y_half, z_half, z_three_fourths);
    // // cilk_spawn build_rag_loop(rag6, feature_manager6, mito_probs6, 0, x_full, y_half, y_full, z_half, z_three_fourths);
    // // cilk_spawn build_rag_loop(rag7, feature_manager7, mito_probs7, 0, x_full, 0, y_half, z_three_fourths, z_full);
    // // build_rag_loop(rag8, feature_manager8, mito_probs8, 0, x_full, y_half, y_full, z_three_fourths, z_full);

    // // cilk_spawn build_rag_loop(rag, feature_manager, mito_probs, 0, x_full, 0, y_full, 0, z_fourth);
    // // cilk_spawn build_rag_loop(rag2, feature_manager2, mito_probs2, 0, x_full, 0, y_full, z_eighth, z_fourth);
    // // cilk_spawn build_rag_loop(rag3, feature_manager3, mito_probs3, 0, x_full, 0, y_full, z_fourth, z_three_eighths);
    // // cilk_spawn build_rag_loop(rag4, feature_manager4, mito_probs4, 0, x_full, 0, y_full, z_three_eighths, z_half);
    // // cilk_spawn build_rag_loop(rag5, feature_manager5, mito_probs5, 0, x_full, 0, y_full, z_half, z_five_eighths);
    // // cilk_spawn build_rag_loop(rag6, feature_manager6, mito_probs6, 0, x_full, 0, y_full, z_five_eighths, z_three_fourths);
    // // cilk_spawn build_rag_loop(rag7, feature_manager7, mito_probs7, 0, x_full, 0, y_full, z_three_fourths, z_seven_eighths);
    // // build_rag_loop(rag8, feature_manager8, mito_probs8, 0, x_full, 0, y_full, z_seven_eighths, z_full);

    // // cilk_spawn build_rag_loop(rag, feature_manager, mito_probs, 0, x_half, 0, y_half, 0, z_half);
    // // cilk_spawn build_rag_loop(rag2, feature_manager2, mito_probs2, x_half, x_full, 0, y_half, 0, z_half);
    // // cilk_spawn build_rag_loop(rag3, feature_manager3, mito_probs3, 0, x_half, y_half, y_full, 0, z_half);
    // // cilk_spawn build_rag_loop(rag4, feature_manager4, mito_probs4, x_half, x_full, y_half, y_full, 0, z_half);
    // // cilk_spawn build_rag_loop(rag5, feature_manager5, mito_probs5, 0, x_half, 0, y_half, z_half, z_full);
    // // cilk_spawn build_rag_loop(rag6, feature_manager6, mito_probs6, x_half, x_full, 0, y_half, z_half, z_full);
    // // cilk_spawn build_rag_loop(rag7, feature_manager7, mito_probs7, 0, x_half, y_half, y_full, z_half, z_full);
    // // build_rag_loop(rag8, feature_manager8, mito_probs8, x_half, x_full, y_half, y_full, z_half, z_full);


    // cilk_spawn build_rag_loop(rag_list[0], fm_list[0], mitop_list[0], 0, x_half, 0, y_half, 0, z_half);
    // cilk_spawn build_rag_loop(rag_list[1], fm_list[1], mitop_list[1], x_half, x_full, 0, y_half, 0, z_half);
    // cilk_spawn build_rag_loop(rag_list[2], fm_list[2], mitop_list[2], 0, x_half, y_half, y_full, 0, z_half);
    // cilk_spawn build_rag_loop(rag_list[3], fm_list[3], mitop_list[3], x_half, x_full, y_half, y_full, 0, z_half);
    // cilk_spawn build_rag_loop(rag_list[4], fm_list[4], mitop_list[4], 0, x_half, 0, y_half, z_half, z_full);
    // cilk_spawn build_rag_loop(rag_list[5], fm_list[5], mitop_list[5], x_half, x_full, 0, y_half, z_half, z_full);
    // cilk_spawn build_rag_loop(rag_list[6], fm_list[6], mitop_list[6], 0, x_half, y_half, y_full, z_half, z_full);
    // build_rag_loop(rag_list[7], fm_list[7], mitop_list[7], x_half, x_full, y_half, y_full, z_half, z_full);

    // cilk_sync;

    // boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();


    // merge_rag_list_recurse(rag_list, fm_list, 0, nworkers-1);
    // merge_mito_prob_list_recurse(mitop_list, 0, nworkers-1);

    // // merge_rags (rag, rag_list[0], feature_manager, fm_list[0]);

    // mito_probs = mitop_list[0];

    // boost::posix_time::ptime end = boost::posix_time::microsec_clock::local_time();

    // cout << endl << "---------------------- TIME TO MERGE: " << (end - start).total_milliseconds() << " ms\n";


    // =================================== SYSTEMATIC ================================================
    // cilk_build_rag_loop(rag, feature_manager, mito_probs, 0, x_full, 0, y_full, 0, z_full, use_mito_prob);
    build_rag_recurse (rag, feature_manager, mito_probs, 0, x_full, 0, y_full, 0, z_full, use_mito_prob);

    if (use_mito_prob) {
        Label_t largest_id = 0;
        for (Rag_t::nodes_iterator iter = rag->nodes_begin(); iter != rag->nodes_end(); ++iter) {
            Label_t id = (*iter)->get_node_id();
            largest_id = (id>largest_id)? id : largest_id;
    	
            MitoTypeProperty mtype = mito_probs[id];
            mtype.set_type(); 
            (*iter)->set_property("mito-type", mtype);
        }
    }
    //printf("Done Biostack rag, largest: %u\n", largest_id);
}



void BioStack::set_edge_locations()
{
  
    EdgeCount best_edge_z;
    EdgeLoc best_edge_loc;
    determine_edge_locations(best_edge_z, best_edge_loc, false); //optimal_prob_edge_loc);
    
    // set edge properties for export 
    for (Rag_t::edges_iterator iter = rag->edges_begin(); iter != rag->edges_end(); ++iter) {
//         if (!((*iter)->is_false_edge())) {
//             if (feature_manager) {
//                 double val = feature_manager->get_prob((*iter));
//                 (*iter)->set_weight(val);
//             } 
//         }
        Label_t x = 0;
        Label_t y = 0;
        Label_t z = 0;
        
        if (best_edge_loc.find(*iter) != best_edge_loc.end()) {
            Location loc = best_edge_loc[*iter];
            x = boost::get<0>(loc);
            // assume y is bottom of image
            // (technically ignored by raveler so okay)
            y = boost::get<1>(loc); //height - boost::get<1>(loc) - 1;
            z = boost::get<2>(loc);
        }
        
        (*iter)->set_property("location", Location(x,y,z));
    }
  
}

void BioStack::set_synapse_exclusions(vector<vector<unsigned int> >& synapse_locations_) 
{
    synapse_locations = synapse_locations_;
}

void BioStack::set_synapse_exclusions(const char* synapse_json)
{
    unsigned int ysize = labelvol->shape(1);

    if (!rag) {
        throw ErrMsg("No RAG defined for stack");
    }

    synapse_locations.clear();

    Json::Reader json_reader;
    Json::Value json_reader_vals;
    
    ifstream fin(synapse_json);
    if (!fin) {
        throw ErrMsg("Error: input file: " + string(synapse_json) + " cannot be opened");
    }
    if (!json_reader.parse(fin, json_reader_vals)) {
        throw ErrMsg("Error: Json incorrectly formatted");
    }
    fin.close();
 
    Json::Value synapses = json_reader_vals["data"];

    for (int i = 0; i < synapses.size(); ++i) {
        vector<vector<unsigned int> > locations;
        Json::Value location = synapses[i]["T-bar"]["location"];
        if (!location.empty()) {
            vector<unsigned int> loc;
            loc.push_back(location[(unsigned int)(0)].asUInt());
            loc.push_back(ysize - location[(unsigned int)(1)].asUInt() - 1);
            loc.push_back(location[(unsigned int)(2)].asUInt());
            synapse_locations.push_back(loc);
            locations.push_back(loc);
        }
        Json::Value psds = synapses[i]["partners"];
        for (int i = 0; i < psds.size(); ++i) {
            Json::Value location = psds[i]["location"];
            if (!location.empty()) {
                vector<unsigned int> loc;
                loc.push_back(location[(unsigned int)(0)].asUInt());
                loc.push_back(ysize - location[(unsigned int)(1)].asUInt() - 1);
                loc.push_back(location[(unsigned int)(2)].asUInt());
                synapse_locations.push_back(loc);
                locations.push_back(loc);
            }
        }

        for (int iter1 = 0; iter1 < locations.size(); ++iter1) {
            for (int iter2 = (iter1 + 1); iter2 < locations.size(); ++iter2) {
                add_edge_constraint(rag, labelvol, locations[iter1][0], locations[iter1][1],
                    locations[iter1][2], locations[iter2][0], locations[iter2][1], locations[iter2][2]);           
            }
        }
    }

}
    
void BioStack::serialize_graph_info(Json::Value& json_writer)
{
    unordered_map<Label_t, int> synapse_counts;
    if (saved_synapse_counts.size() > 0) {
        synapse_counts = saved_synapse_counts;
    } else { 
        load_synapse_counts(synapse_counts);
    }

    int id = 0;
    for (unordered_map<Label_t, int>::iterator iter = synapse_counts.begin();
            iter != synapse_counts.end(); ++iter, ++id) {
        Json::Value synapse_pair;
        synapse_pair[(unsigned int)(0)] = iter->first;
        synapse_pair[(unsigned int)(1)] = iter->second;
        json_writer["synapse_bodies"][id] =  synapse_pair;
    }
}

void BioStack::add_edge_constraint(RagPtr rag, VolumeLabelPtr labelvol2, unsigned int x1,
        unsigned int y1, unsigned int z1, unsigned int x2, unsigned int y2, unsigned int z2)
{
    Label_t label1 = (*labelvol2)(x1,y1,z1);
    Label_t label2 = (*labelvol2)(x2,y2,z2);

    if (label1 && label2 && (label1 != label2)) {
        RagEdge_t* edge = rag->find_rag_edge(label1, label2);
        if (!edge) {
            RagNode_t* node1 = rag->find_rag_node(label1);
            RagNode_t* node2 = rag->find_rag_node(label2);
            edge = rag->insert_rag_edge(node1, node2);
            edge->set_weight(1.0);
            edge->set_false_edge(true);
        }
        edge->set_preserve(true);
    }
}

}
