#include "MergePriorityFunction.h"
#include "../BioPriors/MitoTypeProperty.h"

#include <boost/thread/mutex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#include <pthread.h>

#include <algorithm>

#include <cstdio>

#define MAX_NODE_LABEL 100000

using namespace NeuroProof;


inline unsigned long int pairing_func(unsigned long int num1, unsigned long int num2) {
    if (num1 < num2)
        return (num1 << 32) + num2;
    return (num2 << 32) + num1;
}


inline void pairing_func_inverse_num1(unsigned long int pairing, unsigned int* num1, unsigned int* num2) {
    *num1 = (unsigned int)(pairing >> 32);
    *num2 = (unsigned int)(pairing & 0x00000000ffffffff);
}



double mito_boundary_ratio(RagEdge_t* edge)
{
    RagNode_t* node1 = edge->get_node1();
    RagNode_t* node2 = edge->get_node2();
    double ratio = 0.0;

    try {
        MitoTypeProperty& type1_mito = node1->get_property<MitoTypeProperty>("mito-type");
        MitoTypeProperty& type2_mito = node2->get_property<MitoTypeProperty>("mito-type");
        int type1 = type1_mito.get_node_type(); 
        int type2 = type2_mito.get_node_type(); 

        RagNode_t* mito_node = 0;		
        RagNode_t* other_node = 0;		

        if ((type1 == 2) && (type2 == 1) ){
            mito_node = node1;
            other_node = node2;
        } else if((type2 == 2) && (type1 == 1) ){
            mito_node = node2;
            other_node = node1;
        } else { 
            return 0.0; 	
        }

        if (mito_node->get_size() > other_node->get_size()) {
            return 0.0;
        }

        unsigned long long mito_node_border_len = mito_node->compute_border_length();		

        ratio = (edge->get_size())*1.0/mito_node_border_len; 

        if (ratio > 1.0){
            printf("ratio > 1 for %d %d\n", mito_node->get_node_id(), other_node->get_node_id());
            return 0.0;
        }

    } catch (ErrMsg& err) {
        // # just return 
    } 

    return ratio;
}




void ProbPriority::initialize_priority(double threshold_, bool use_edge_weight)
{
    printf("ProbPriority::initialize_priority: start 1\n");
    printf("ProbPriority::initialize_priority: start 2\n");
    threshold = threshold_;
    int num_edges = (int)rag->get_num_edges();
    printf("ProbPriority::initialize_priority: num_edges = %d\n", num_edges);
    vector<Rag_t::edges_iterator> iter_vec;
    pair<double, std::pair<Node_t, Node_t> > *p_tmp_array = new pair<double, std::pair<Node_t, Node_t> >[num_edges];

    printf("ProbPriority::initialize_priority: 0\n");
    for (Rag_t::edges_iterator iter = rag->edges_begin(); iter != rag->edges_end(); ++iter) {
    	iter_vec.push_back(iter);
    }
    printf("ProbPriority::initialize_priority: 1\n");
    // cilk::reducer< cilk::op_list_append<char> > letters_reducer;
    int nworkers = __cilkrts_get_nworkers(); // CILK_NWORKERS
    initialize_dirty_edges_storage(nworkers);
    vector<int> indices_lists [nworkers];
   
    printf("ProbPriority::initialize_priority: 2\n");
        
    cilk_for (int i = 0; i < num_edges; i++) {
    	Rag_t::edges_iterator iter = iter_vec[i];
    	
    // for (Rag_t::edges_iterator iter = rag->edges_begin(); iter != rag->edges_end(); ++iter) {
		if (valid_edge(*iter)) {
		    double val;
		    if (use_edge_weight)
				val = (*iter)->get_weight();
		    else
				val = feature_mgr->get_prob(*iter);
		    
		    (*iter)->set_weight(val);

		    if (val <= threshold) {
				// ranking.insert(std::make_pair(val, std::make_pair((*iter)->get_node1()->get_node_id(), (*iter)->get_node2()->get_node_id())));
				int worker_id = __cilkrts_get_worker_number();
		    	indices_lists[worker_id].push_back(i);
		    	p_tmp_array[i] = std::make_pair(val, std::make_pair((*iter)->get_node1()->get_node_id(), (*iter)->get_node2()->get_node_id()));
		    }
		}
    }
    
    printf("ProbPriority::initialize_priority: 3\n");
        
    /// merge and sort the indices
    vector<int> indices_to_insert;
    for (int i = 0; i < nworkers; i++) {
    	indices_to_insert.insert(indices_to_insert.end(), indices_lists[i].begin(), indices_lists[i].end());
    }
    
    printf("ProbPriority::initialize_priority: 4\n");
        
    std::sort(indices_to_insert.begin(), indices_to_insert.end());
    
    printf("ProbPriority::initialize_priority: 5\n");
    
    for (vector<int>::iterator it = indices_to_insert.begin(); it != indices_to_insert.end(); ++it) {
    	ranking.insert(p_tmp_array[*it]);
    }
    
    printf("ProbPriority::initialize_priority: 6\n");
    
    delete p_tmp_array;
    
}



void ProbPriority::initialize_random(double pthreshold){

    threshold = pthreshold;
    for (Rag_t::edges_iterator iter = rag->edges_begin(); iter != rag->edges_end(); ++iter) {
    	if (valid_edge(*iter)) {

    	    double val1 = feature_mgr->get_prob(*iter);
    	    (*iter)->set_weight(val1);

    	    if (val1 <= threshold){ 
    		srand ( time(NULL) );
    		double val= rand()*(threshold/ RAND_MAX);

    		(*iter)->set_weight(val);
    		ranking.insert(std::make_pair(val, std::make_pair((*iter)->get_node1()->get_node_id(), (*iter)->get_node2()->get_node_id())));
    	    }
    	}
    }
}
   
void ProbPriority::clear_dirty()
{
    boost::mutex ranking_lock;

    // Dirty_t delete_list; // use to check if an edge is new
    // vector<OrderedPair> delete_vec;
    // for (std::vector<Dirty_t>::iterator it1 = dirty_edges_storage.begin(); it1 != dirty_edges_storage.end(); ++it1) {
    //     for (Dirty_t::iterator it2 = it1->begin(); it2 != it1->end(); ++it2) {
    //         size_t size_before = delete_list.size();
    //         delete_list.insert(*it2);
    //         if (size_before != delete_list.size())
    //             delete_vec.push_back(*it2);
    //     }
    //     it1->clear();
    // }

    // cilk_for (vector<OrderedPair>::iterator it = delete_vec.begin(); it != delete_vec.end(); ++it) {
    //     Node_t node1 = it->region1;
    //     Node_t node2 = it->region2;
    //     RagNode_t* rag_node1 = rag->find_rag_node_no_probe(node1); 
    //     RagNode_t* rag_node2 = rag->find_rag_node_no_probe(node2); 

    //     if (!(rag_node1 && rag_node2)) {
    //         continue;
    //     }
    //     RagEdge_t* rag_edge = rag->find_rag_edge_no_probe(rag_node1, rag_node2);

    //     if (!rag_edge) {
    //         continue;
    //     }

    //     // assert(rag_edge->is_dirty());

    //     if (!rag_edge->is_dirty()) { 
    //         continue;
    //     }

    //     rag_edge->set_dirty(false);

    //     if (valid_edge(rag_edge)) {
    //         double val = feature_mgr->get_prob(rag_edge);
    //         rag_edge->set_weight(val);

    //         if (val <= threshold) {
    //             ranking_lock.lock();
    //             ranking.insert(std::make_pair(val, std::make_pair(node1, node2)));
    //             ranking_lock.unlock();
    //         }
    //         else{ 
    //         kicked_out++;   
    //         if (kicked_fid)
    //           fprintf(kicked_fid, "0 %f %u %u %lu %lu\n", val,
    //             node1, node2, rag_node1->get_size(), rag_node2->get_size());
    //         }
    //     }
    // }


    std::tr1::unordered_set<unsigned long int> reduced_set;
    std::vector<unsigned long int> delete_vec;
    for (std::vector<std::tr1::unordered_set<unsigned long int> >::iterator it1 = dirty_edges_storage.begin(); it1 != dirty_edges_storage.end(); ++it1) {
        for (std::tr1::unordered_set<unsigned long int>::iterator it2 = it1->begin(); it2 != it1->end(); ++it2) {
            int size_before = reduced_set.size(); 
            reduced_set.insert(*it2);
            if (reduced_set.size() != size_before)
                delete_vec.push_back(*it2);
        }
        it1->clear();
    }

    cilk_for (vector<unsigned long int>::iterator it = delete_vec.begin(); it != delete_vec.end(); ++it) {
        Node_t node1, node2;
        pairing_func_inverse_num1(*it, &node1, &node2);
        RagNode_t* rag_node1 = rag->find_rag_node_no_probe(node1); 
        RagNode_t* rag_node2 = rag->find_rag_node_no_probe(node2); 

        if (!(rag_node1 && rag_node2)) {
            continue;
        }
        RagEdge_t* rag_edge = rag->find_rag_edge_no_probe(rag_node1, rag_node2);

        if (!rag_edge) {
            continue;
        }

        // assert(rag_edge->is_dirty());

        if (!rag_edge->is_dirty()) { 
            continue;
        }

        rag_edge->set_dirty(false);

        if (valid_edge(rag_edge)) {
            double val = feature_mgr->get_prob(rag_edge);
            rag_edge->set_weight(val);

            if (val <= threshold) {
                ranking_lock.lock();
                ranking.insert(std::make_pair(val, std::make_pair(node1, node2)));
                ranking_lock.unlock();
            }
            else{ 
            kicked_out++;   
            if (kicked_fid)
              fprintf(kicked_fid, "0 %f %u %u %lu %lu\n", val,
                node1, node2, rag_node1->get_size(), rag_node2->get_size());
            }
        }
    }
}


bool ProbPriority::empty()
{
    if (ranking.empty()) {
        clear_dirty();
    }
    return ranking.empty();
}

boost::mutex mu;
boost::mutex mu1; // for ranking
boost::mutex mu2; // for dirty storage
RagEdge_t* ProbPriority::get_edge_with_iter(EdgeRank_t::iterator iter)
{
    if (ranking.empty())
        return 0;

    mu1.lock();
    if (ranking.empty()) {
        mu1.unlock();
        return 0;
    }

    double curr_threshold = (*iter).first;
    Node_t node1 = (*iter).second.first;
    Node_t node2 = (*iter).second.second;

    // cout << curr_threshold << " " << node1 << " " << node2 << std::endl;

    if (iter == ranking.begin() && curr_threshold > threshold) {
        ranking.clear();
        mu1.unlock();
        return 0;
    }

    ranking.erase(iter);
    mu1.unlock();

    RagNode_t* rag_node1 = rag->find_rag_node_no_probe(node1); 
    RagNode_t* rag_node2 = rag->find_rag_node_no_probe(node2); 

    if (!(rag_node1 && rag_node2)) {
        return 0;
    }
    RagEdge_t* rag_edge = rag->find_rag_edge_no_probe(rag_node1, rag_node2);

    if (!rag_edge) {
        return 0;
    }

    if (!valid_edge(rag_edge)) {
        return 0;
    }

    double val = rag_edge->get_weight();
    
    bool dirty = false;
    if (rag_edge->is_dirty()) {
        dirty = true;
        val = feature_mgr->get_prob(rag_edge);
        rag_edge->set_weight(val);
        rag_edge->set_dirty(false);
        mu2.lock();
        erase_from_dirty_edges_storage(node1, node2);
        mu2.unlock();
    }
    
    if (val > (curr_threshold + Epsilon)) {
        mu1.lock();
        if (dirty && (val <= threshold)) {
            ranking.insert(std::make_pair(val, std::make_pair(node1, node2)));
        }
        else{ 
            //printf("edge prob changed from %.4f to %.4f\n",curr_threshold, val);
            kicked_out++;   
            if (kicked_fid)
              fprintf(kicked_fid, "0 %f %u %u %lu %lu\n", val,
            node1, node2, rag_node1->get_size(), rag_node2->get_size());
            //fprintf(kicked_fid, "0 %f %u %u %lu %lu\n", rag_edge->get_weight(),node1, node2, rag_node1->get_size(), rag_node2->get_size());
            
            //add_dirty_edge(rag_edge);
        }
        mu1.unlock();
        return 0;
    }
    return rag_edge; 
}



inline unsigned int prior_hash(unsigned int a) {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}
 

 
inline void prior_update_max(unsigned int* array, int loc, unsigned int prior) {
    unsigned int old_val;
    do {
        old_val = array[loc];
        if (old_val > prior)
            return;
    } while (!__sync_bool_compare_and_swap(&(array[loc]), old_val, prior));
}


inline void prior_update(unsigned int* array, int loc, unsigned int prior) {
    unsigned int old_val;
    do {
        old_val = array[loc];
        if (old_val != 0 && old_val < prior) 
            return;
    } while (!__sync_bool_compare_and_swap(&(array[loc]), old_val, prior));
}


// irrelevant
inline void prior_update_dist_2(int array [], int loc, int prior) {
    int old_val;
    do {
        old_val = array[loc];
        if (old_val < prior) 
            return;
    } while (!__sync_bool_compare_and_swap(&(array[loc]), old_val, prior));
}


void ProbPriority::get_edges_parallel (vector<EdgeRank_t::iterator> &edges_to_remove_from_queue, vector<RagEdge_t*> &edges_to_process) {
    // cilk_for (int i = 0; i < edges_to_remove_from_queue.size(); i++) {
    //     // mu.lock();
    //     RagEdge_t* rag_edge = get_edge_with_iter(edges_to_remove_from_queue[i]);
    //     // maybe move this to earlier?
    //     if (rag_edge) {
    //         mu.lock();
    //         edges_to_process.push_back(rag_edge);
    //         mu.unlock();
    //     }
    //     // mu.unlock();
    // }

    // loop to remove edges 
    vector<pair<Node_t, Node_t> > node_pairs_vector;
    vector<double> curr_threshold_vector;

    for (vector<EdgeRank_t::iterator>::iterator it = edges_to_remove_from_queue.begin(); it != edges_to_remove_from_queue.end(); ++it) {
        EdgeRank_t::iterator iter = *it;

        if (ranking.empty())
            break;

        double curr_threshold = (*iter).first;
        Node_t node1 = (*iter).second.first;
        Node_t node2 = (*iter).second.second;

        if (iter == ranking.begin() && curr_threshold > threshold) {
            ranking.clear();
            break;
        }

        ranking.erase(iter);
        node_pairs_vector.push_back(make_pair(node1, node2));
        curr_threshold_vector.push_back(curr_threshold);
    }

    // parallel loop to process these node pairs
    int nworkers = __cilkrts_get_nworkers(); // CILK_NWORKERS
    vector<RagEdge_t*> edges_to_process_list [nworkers];
    EdgeRank_t ranking_list [nworkers];
    vector<pair<Node_t, Node_t> > dirty_list [nworkers];

    // cilk
    cilk_for (int i = 0; i < node_pairs_vector.size(); ++i) {
        int worker_id = __cilkrts_get_worker_number();
        double curr_threshold = curr_threshold_vector[i];
        Node_t node1 = node_pairs_vector[i].first;
        Node_t node2 = node_pairs_vector[i].second;
        RagNode_t* rag_node1 = rag->find_rag_node(node1); 
        RagNode_t* rag_node2 = rag->find_rag_node(node2); 
    
        if (!(rag_node1 && rag_node2)) {
            continue;
        }

        mu.lock();
        RagEdge_t* rag_edge = rag->find_rag_edge(rag_node1, rag_node2);
        mu.unlock();

        if (!rag_edge) {
            continue;
        }

        if (!valid_edge(rag_edge)) {
            continue;
        }

        double val = rag_edge->get_weight();
    
        bool dirty = false;
        if (rag_edge->is_dirty()) {
            dirty = true;
            val = feature_mgr->get_prob(rag_edge);
            rag_edge->set_weight(val);
            rag_edge->set_dirty(false);
            dirty_list[worker_id].push_back(make_pair(node1, node2));
        }

        if (val > (curr_threshold + Epsilon)) {
            if (dirty && (val <= threshold)) {
                // mu1.lock();
                // ranking.insert(std::make_pair(val, std::make_pair(node1, node2)));
                // mu1.unlock();
                ranking_list[worker_id].insert(std::make_pair(val, std::make_pair(node1, node2)));
            }
            else{ 
                //printf("edge prob changed from %.4f to %.4f\n",curr_threshold, val);
                // kicked_out++;   
                __sync_fetch_and_add( &kicked_out, 1);
                if (kicked_fid)
                  fprintf(kicked_fid, "0 %f %u %u %lu %lu\n", val,
                node1, node2, rag_node1->get_size(), rag_node2->get_size());
                //fprintf(kicked_fid, "0 %f %u %u %lu %lu\n", rag_edge->get_weight(),node1, node2, rag_node1->get_size(), rag_node2->get_size());
                
                //add_dirty_edge(rag_edge);
            }
            continue;
        }

        edges_to_process_list[worker_id].push_back(rag_edge);
    }
    
    for (int i = 0; i < nworkers; ++i) {
        for (EdgeRank_t::iterator it = ranking_list[i].begin(); it != ranking_list[i].end(); ++it) {
            ranking.insert(*it);
        }
        for (vector<pair<Node_t, Node_t> >::iterator it = dirty_list[i].begin(); it != dirty_list[i].end(); ++it) {
            // dirty_edges.erase(OrderedPair(it->first, it->second));
            erase_from_dirty_edges_storage(it->first, it->second);
        }
        for (vector<RagEdge_t*>::iterator it = edges_to_process_list[i].begin(); it != edges_to_process_list[i].end(); ++it) {
            edges_to_process.push_back(*it);
        }
        edges_to_process_list[i].clear();
    }
}


map<Node_t, vector<int> > neighbors_cache;


void ProbPriority::get_top_independent_edges (int nbd_size, vector<RagEdge_t*> &edges_to_process, bool rand_prior) {
    vector<EdgeRank_t::iterator> top_edges;
    int num_edges_looked_at = 0;
    for (EdgeRank_t::iterator it = ranking.begin(); it != ranking.end() && num_edges_looked_at < nbd_size; ++it, ++num_edges_looked_at) {
        double curr_threshold = (*it).first;
        if (curr_threshold > threshold)
            break;
        top_edges.push_back(it);
    }

    // do priority updates to see which edges can be processed

    // int node_priority_update_array_1 [MAX_NODE_LABEL];
    // int node_priority_update_array_2 [MAX_NODE_LABEL];


    unsigned int* node_priority_update_array_1 = (unsigned int*)calloc(MAX_NODE_LABEL, sizeof(unsigned int));
    unsigned int* node_priority_update_array_2 = (unsigned int*)calloc(MAX_NODE_LABEL, sizeof(unsigned int));

    // if (!rand_prior) {
    //     for (int i = 0; i < MAX_NODE_LABEL; ++i) {
    //         node_priority_update_array_1[i] = nbd_size + 1;
    //         node_priority_update_array_2[i] = nbd_size + 1;
    //     }
    // } else {
    //     for (int i = 0; i < MAX_NODE_LABEL; ++i) {
    //         node_priority_update_array_1[i] = 0;
    //         node_priority_update_array_2[i] = 0; 
    //     }
    // }


    for (unsigned int i = 0; i < top_edges.size(); ++i) {
        Node_t node1 = (*top_edges[i]).second.first;
        Node_t node2 = (*top_edges[i]).second.second;

        if (rand_prior) {
            prior_update_max(node_priority_update_array_1, (int)node1, prior_hash(i));
            prior_update_max(node_priority_update_array_1, (int)node2, prior_hash(i));
        } else {
            prior_update(node_priority_update_array_1, (int)node1, i);
            prior_update(node_priority_update_array_1, (int)node2, i);
        }
    }

    // check which edge can be processed

    vector<EdgeRank_t::iterator> edges_to_remove_from_queue;

    for (int i = 0; i < top_edges.size(); ++i) {
        Node_t node1 = (*top_edges[i]).second.first;
        Node_t node2 = (*top_edges[i]).second.second;
        if (node_priority_update_array_1[(int)node1] == node_priority_update_array_1[(int)node2]) {
            edges_to_remove_from_queue.push_back(top_edges[i]);
        }
    }


    /*
    // =============================== for distance 2 ==============================

    vector<vector<int> > node_ids_vec;
    vector<EdgeRank_t::iterator> edges_to_remove_from_queue;
    vector<EdgeRank_t::iterator> edges_for_round_2;

    for (int i = 0; i < top_edges.size(); ++i) {
        Node_t node1 = (*top_edges[i]).second.first;
        Node_t node2 = (*top_edges[i]).second.second;
        // TODO: else do we reset the priority of the first array at those locations?
        if (node_priority_update_array_1[(int)node1] == node_priority_update_array_1[(int)node2]) {
            
            // ================ do distance 2 for both nodes
            // vector<int> node_ids;
            // bool valid_1 = true;
            // if (neighbors_cache.find(node1) == neighbors_cache.end()) {
            //     valid_1 = rag->get_neighboring_labels(node1, node_ids);
            //     neighbors_cache[node2] = node_ids;
            // } else {
            //     for (vector<int>::iterator it = neighbors_cache[node1].begin(); it != neighbors_cache[node1].end(); ++it)
            //         node_ids.push_back(*it);
            // }

            // bool valid_2 = true;
            // vector<int> node_ids_2;
            // if (neighbors_cache.find(node2) == neighbors_cache.end()) {
            //     valid_2 = rag->get_neighboring_labels(node2, node_ids_2);
            //     neighbors_cache[node2] = node_ids_2;
            // } else {
            //     for (vector<int>::iterator it = neighbors_cache[node2].begin(); it != neighbors_cache[node2].end(); ++it)
            //         node_ids_2.push_back(*it);
            // }
            // node_ids.insert(node_ids.end(), node_ids_2.begin(), node_ids_2.end());
            // if (valid_1 && valid_2) {
            //     edges_for_round_2.push_back(top_edges[i]);
            //     node_ids_vec.push_back(node_ids);
            // }

            // ========= do distance 2 for node_remove only
            bool valid_2 = true;
            vector<int> node_ids;
            // assert(node2 > node1);
            if (neighbors_cache.find(node2) == neighbors_cache.end()) {
                valid_2 = rag->get_neighboring_labels(node2, node_ids);
                neighbors_cache[node2] = node_ids;
            } else {
                for (vector<int>::iterator it = neighbors_cache[node2].begin(); it != neighbors_cache[node2].end(); ++it)
                    node_ids.push_back(*it);
            }
            // node_ids.push_back(node1);

            edges_for_round_2.push_back(top_edges[i]);
            node_ids_vec.push_back(node_ids);
        }
    }


    // priority update round 2
    cilk_for (int i = 0; i < edges_for_round_2.size(); i++) {
        Node_t node1 = (*edges_for_round_2[i]).second.first;
        Node_t node2 = (*edges_for_round_2[i]).second.second;
        int priority_tag = node_priority_update_array_1[(int)node1];
        // do priority update with distance 2

        for (vector<int>::iterator it = node_ids_vec[i].begin(); it != node_ids_vec[i].end(); ++it) {
        // cilk_for (int j = 0; j < node_ids_vec[i].size(); ++j) {
            // cout << node_ids_vec[i].size() << endl;
            // prior_update(node_priority_update_array_2, node_ids_vec[i][j], priority_tag);
            prior_update_max(node_priority_update_array_2, *it, priority_tag);
        }
    }

    // check edges that can be processed in parallel
    for (int i = 0; i < edges_for_round_2.size(); ++i) {
        Node_t node1 = (*edges_for_round_2[i]).second.first;
        int priority_val = node_priority_update_array_1[(int)node1];
        bool pass = true; 
        for (int j = 0; j < node_ids_vec[i].size(); ++j) {
            if (priority_val != node_priority_update_array_2[node_ids_vec[i][j]]) {
                pass = false;
                break;
            }
        }
        if (pass) {
            edges_to_remove_from_queue.push_back(edges_for_round_2[i]);
        }
    }
    */

    if (edges_to_remove_from_queue.size() == 0) {
        ranking.clear();
    } else {
        get_edges_parallel(edges_to_remove_from_queue, edges_to_process);
    }

    free(node_priority_update_array_1);
    free(node_priority_update_array_2);
}





RagEdge_t* ProbPriority::get_top_edge()
{
    EdgeRank_t::iterator first_entry = ranking.begin();
    double curr_threshold = (*first_entry).first;
    Node_t node1 = (*first_entry).second.first;
    Node_t node2 = (*first_entry).second.second;
    ranking.erase(first_entry);

    // cout << curr_threshold << " " << node1 << " " << node2 << std::endl;

    if (curr_threshold > threshold) {
		ranking.clear();
		return 0;
    }


    RagNode_t* rag_node1 = rag->find_rag_node(node1); 
    RagNode_t* rag_node2 = rag->find_rag_node(node2); 

    if (!(rag_node1 && rag_node2)) {
		return 0;
    }
    RagEdge_t* rag_edge = rag->find_rag_edge(rag_node1, rag_node2);

    if (!rag_edge) {
		return 0;
    }

    if (!valid_edge(rag_edge)) {
		return 0;
    }

    double val = rag_edge->get_weight();
    
    bool dirty = false;
    if (rag_edge->is_dirty()) {
		dirty = true;
		val = feature_mgr->get_prob(rag_edge);
		rag_edge->set_weight(val);
		rag_edge->set_dirty(false);
		dirty_edges.erase(OrderedPair(node1, node2));
    }
    
    if (val > (curr_threshold + Epsilon)) {
		if (dirty && (val <= threshold)) {
		    ranking.insert(std::make_pair(val, std::make_pair(node1, node2)));
		}
		else{ 
		    //printf("edge prob changed from %.4f to %.4f\n",curr_threshold, val);
		    kicked_out++;	
		    if (kicked_fid)
		      fprintf(kicked_fid, "0 %f %u %u %lu %lu\n", val,
			node1, node2, rag_node1->get_size(), rag_node2->get_size());
		    //fprintf(kicked_fid, "0 %f %u %u %lu %lu\n", rag_edge->get_weight(),node1, node2, rag_node1->get_size(), rag_node2->get_size());
		    
		    //add_dirty_edge(rag_edge);
		}
		return 0;
    }
    return rag_edge; 
}


// this ig probably not used anymore
void ProbPriority::add_dirty_edge(RagEdge_t* edge)
{
    if (valid_edge(edge)) {
    	edge->set_dirty(true);
    	dirty_edges.insert(OrderedPair(edge->get_node1()->get_node_id(), edge->get_node2()->get_node_id()));
    }  
}




void ProbPriority::add_dirty_edge_parallel(RagEdge_t* edge, int worker_id)
{
    // if (valid_edge(edge)) {
    //     edge->set_dirty(true);
    //     int node1 = edge->get_node1()->get_node_id();
    //     int node2 = edge->get_node2()->get_node_id();
    //     dirty_edges_storage[worker_id].insert(OrderedPair(node1, node2));
    // }
    
    if (valid_edge(edge)) {
        edge->set_dirty(true);
        int node1 = edge->get_node1()->get_node_id();
        int node2 = edge->get_node2()->get_node_id();
        dirty_edges_storage[worker_id].insert(pairing_func(node1, node2));
    }
}



void ProbPriority::initialize_dirty_edges_storage(int num_workers) {
    set_nworkers(num_workers);
    for (int i = 0; i < num_workers; ++i) {
        // Dirty_t map_item;
        std::tr1::unordered_set<unsigned long int> map_item;
        dirty_edges_storage.push_back(map_item);
    }
}


void ProbPriority::erase_from_dirty_edges_storage(Node_t node1, Node_t node2) {
    // for (std::vector<Dirty_t>::iterator it = dirty_edges_storage.begin(); 
    //                                         it != dirty_edges_storage.end(); ++it) {
    //     it->erase(OrderedPair(node1, node2));
    // }

    unsigned long int pairing = pairing_func(node1, node2);
    for (std::vector<std::tr1::unordered_set<unsigned long int> >::iterator it = dirty_edges_storage.begin(); it != dirty_edges_storage.end(); ++it)
        it->erase(pairing);
}







//*******************************************************************************************************************



void MitoPriority::initialize_priority(double threshold_, bool use_edge_weight)
{
    threshold = threshold_;
    for (Rag_t::edges_iterator iter = rag->edges_begin(); iter != rag->edges_end(); ++iter) {
        if (valid_edge(*iter)) {
	    double val;
	    val = 1 - mito_boundary_ratio((*iter));

	    if (val < threshold) {
	        ranking.insert(std::make_pair(val, std::make_pair((*iter)->get_node1()->get_node_id(), (*iter)->get_node2()->get_node_id())));
	    }
        }
    }
}




void MitoPriority::clear_dirty()
{
    for (Dirty_t::iterator iter = dirty_edges.begin(); iter != dirty_edges.end(); ++iter) {
    	Node_t node1 = (*iter).region1;
    	Node_t node2 = (*iter).region2;
    	RagNode_t* rag_node1 = rag->find_rag_node(node1); 
    	RagNode_t* rag_node2 = rag->find_rag_node(node2); 

    	if (!(rag_node1 && rag_node2)) {
    	    continue;
    	}
    	RagEdge_t* rag_edge = rag->find_rag_edge(rag_node1, rag_node2);

    	if (!rag_edge) {
    	    continue;
    	}

    	assert(rag_edge->is_dirty());
    	rag_edge->set_dirty(false);

    	if (valid_edge(rag_edge)) {
    	    double val = 1 - mito_boundary_ratio(rag_edge);

    	    if (val < threshold) {
    		  ranking.insert(std::make_pair(val, std::make_pair(node1, node2)));
    	    }
    	}
    }
    dirty_edges.clear();
}

bool MitoPriority::empty()
{
    if (ranking.empty()) {
        clear_dirty();
    }
    return ranking.empty();
}


RagEdge_t* MitoPriority::get_top_edge()
{
    EdgeRank_t::iterator first_entry = ranking.begin();
    double curr_threshold = (*first_entry).first;
    Node_t node1 = (*first_entry).second.first;
    Node_t node2 = (*first_entry).second.second;
    ranking.erase(first_entry);

    //cout << curr_threshold << " " << node1 << " " << node2 << std::endl;

    if (curr_threshold >= threshold) {
	ranking.clear();
	return 0;
    }

    RagNode_t* rag_node1 = rag->find_rag_node(node1); 
    RagNode_t* rag_node2 = rag->find_rag_node(node2); 

    if (!(rag_node1 && rag_node2)) {
	return 0;
    }
    RagEdge_t* rag_edge = rag->find_rag_edge(rag_node1, rag_node2);

    if (!rag_edge) {
	return 0;
    }

    if (!valid_edge(rag_edge)) {
	return 0;
    }

    double val = 1 - mito_boundary_ratio(rag_edge);

    bool dirty = false;
    if (rag_edge->is_dirty()) {
	dirty = true;
	val = 1 - mito_boundary_ratio(rag_edge);
	rag_edge->set_dirty(false);
	dirty_edges.erase(OrderedPair(node1, node2));
    }

    if (val > (curr_threshold + Epsilon)) {
	if (dirty && (val < threshold)) {
	    ranking.insert(std::make_pair(val, std::make_pair(node1, node2)));
	}
	else{ 
	    //printf("edge prob changed from %.4f to %.4f\n",curr_threshold, val);
	    kicked_out++;	
	    //add_dirty_edge(rag_edge);
	}
	return 0;
    }
    return rag_edge; 
}

void MitoPriority::add_dirty_edge(RagEdge_t* edge)
{
    if (valid_edge(edge)) {
    	edge->set_dirty(true);
    	dirty_edges.insert(OrderedPair(edge->get_node1()->get_node_id(), edge->get_node2()->get_node_id()));
    }
}





