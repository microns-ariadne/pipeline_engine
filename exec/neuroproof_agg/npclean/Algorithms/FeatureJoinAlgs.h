#ifndef FEATUREJOINALGS_H
#define FEATUREJOINALGS_H

#include "../FeatureManager/FeatureMgr.h"
#include "MergePriorityFunction.h"
#include "MergePriorityQueue.h"
#include "../Rag/RagNodeCombineAlg.h"

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <boost/thread/mutex.hpp>

namespace NeuroProof {

class FeatureCombine : public RagNodeCombineAlg {
  public:
    FeatureCombine(FeatureMgr* feature_mgr_, Rag_t* rag_) :
        feature_mgr(feature_mgr_), rag(rag_) {}
    
    virtual void post_edge_move(RagEdge<unsigned int>* edge_new,
            RagEdge<unsigned int>* edge_remove)
    {
        if (feature_mgr) {
            feature_mgr->mv_features(edge_remove, edge_new);
        } 
    }

    virtual void post_edge_join(RagEdge<unsigned int>* edge_keep,
            RagEdge<unsigned int>* edge_remove)
    {
        if (feature_mgr) {
            if (edge_keep->is_false_edge()) {
                feature_mgr->mv_features(edge_remove, edge_keep); 
            } else if (!(edge_remove->is_false_edge())) {
                feature_mgr->merge_features(edge_keep, edge_remove);
            } else {
                feature_mgr->remove_edge(edge_remove);
            }
        }
    }


    
    virtual void post_edge_join_parallel(std::map<RagEdge<unsigned int>*, std::set<RagEdge<unsigned int>*> > &edge_pairs) {
        // std::cout << "EDGES: " << edge_pairs.size() << std::endl;
        if (!feature_mgr)
            return;

        if (edge_pairs.size() > 5) {
            int num_workers = __cilkrts_get_nworkers();
            std::vector<RagEdge<unsigned int>*> delete_vec [num_workers];
            std::vector<std::map<RagEdge_t*, std::set<RagEdge_t*> >::iterator> iter_vec;
            for (std::map<RagEdge_t*, std::set<RagEdge_t*> >::iterator it = edge_pairs.begin(); it != edge_pairs.end(); ++it) 
                iter_vec.push_back(it);

            cilk_for (std::vector<std::map<RagEdge_t*, std::set<RagEdge_t*> >::iterator>::iterator iter = iter_vec.begin(); iter != iter_vec.end(); ++iter) {
                int worker_id = __cilkrts_get_worker_number();
                std::map<RagEdge_t*, std::set<RagEdge_t*> >::iterator it = *iter;
            // for (std::set<std::pair<RagEdge_t*, RagEdge_t*> >::iterator it = edge_pairs.begin(); it != edge_pairs.end(); ++it) {
                RagEdge<unsigned int>* edge_keep = it->first;
                for (std::set<RagEdge_t*>::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
                    RagEdge<unsigned int>* edge_remove = *it2;
                    if (edge_keep->is_false_edge()) {
                        // feature_mgr->mv_features(edge_remove, edge_keep);
                        bool del = feature_mgr->mv_features_no_delete(edge_remove, edge_keep); 
                        if (del) {
                            delete_vec[worker_id].push_back(edge_remove);
                        }

                    } else if (!(edge_remove->is_false_edge())) {
                        // feature_mgr->merge_features(edge_keep, edge_remove);
                        bool del = feature_mgr->merge_features_no_delete(edge_keep, edge_remove);
                        if (del) {
                            delete_vec[worker_id].push_back(edge_remove);
                        }
                    }
                }
            }

            for (int i = 0; i < num_workers; ++i) {
                feature_mgr->delete_edges(delete_vec[i]);
                delete_vec[i].clear();
            }

        } else {
            for (std::map<RagEdge_t*, std::set<RagEdge_t*> >::iterator it = edge_pairs.begin(); it != edge_pairs.end(); ++it) {
                RagEdge<unsigned int>* edge_keep = it->first;
                for (std::set<RagEdge_t*>::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
                    RagEdge<unsigned int>* edge_remove = *it2;
                    if (edge_keep->is_false_edge()) {
                        feature_mgr->mv_features(edge_remove, edge_keep);

                    } else if (!(edge_remove->is_false_edge())) {
                        feature_mgr->merge_features(edge_keep, edge_remove);
                    }
                }
            }
        }
    }



    virtual void post_node_join(RagNode<unsigned int>* node_keep,
            RagNode<unsigned int>* node_remove)
    {
        if (feature_mgr) {
            RagEdge_t* edge = rag->find_rag_edge_no_probe(node_keep, node_remove);
            assert(edge);
            feature_mgr->merge_features(node_keep, node_remove);
            feature_mgr->remove_edge(edge);
        }
    }



    virtual void post_node_join_parallel(std::vector<std::pair<RagNode<unsigned int>*, RagNode<unsigned int>*> > &node_pairs) {
        // std::cout << "NODES: " << node_pairs.size() << std::endl;
        if (!feature_mgr)
            return;

        if (node_pairs.size() > 10) {
            int num_workers = __cilkrts_get_nworkers();
            std::vector<RagNode<unsigned int>*> node_delete_vec [num_workers];
            std::vector<RagEdge<unsigned int>*> edge_delete_vec [num_workers];
            cilk_for (std::vector<std::pair<RagNode<unsigned int>*, RagNode<unsigned int>*> >::iterator it = node_pairs.begin(); it != node_pairs.end(); ++it) {
                int worker_id = __cilkrts_get_worker_number();
                RagNode_t* node_keep = it->first;
                RagNode_t* node_remove = it->second;
                RagEdge_t* edge = rag->find_rag_edge_no_probe(node_keep, node_remove);
                assert(edge);
                // feature_mgr->merge_features(node_keep, node_remove);
                bool del = feature_mgr->merge_features_no_delete(node_keep, node_remove);
                // feature_mgr->remove_edge(edge);
                if (del) {
                    node_delete_vec[worker_id].push_back(node_remove);
                }
                edge_delete_vec[worker_id].push_back(edge);
            }

            for (int i = 0; i < num_workers; ++i) {
                for (std::vector<RagEdge<unsigned int>*>::iterator it = edge_delete_vec[i].begin(); it != edge_delete_vec[i].end(); ++it) {
                    feature_mgr->remove_edge(*it);
                }

                feature_mgr->delete_nodes(node_delete_vec[i]);
                edge_delete_vec[i].clear();
                node_delete_vec[i].clear();
            }

        } else {
            for (std::vector<std::pair<RagNode<unsigned int>*, RagNode<unsigned int>*> >::iterator it = node_pairs.begin(); it != node_pairs.end(); ++it) {
                RagNode_t* node_keep = it->first;
                RagNode_t* node_remove = it->second;
                RagEdge_t* edge = rag->find_rag_edge(node_keep, node_remove);
                assert(edge);
                feature_mgr->merge_features(node_keep, node_remove);
                feature_mgr->remove_edge(edge);
            }
        }
    }



  protected:
    FeatureMgr* feature_mgr;
    Rag_t* rag;
};



class DelayedPriorityCombine : public FeatureCombine {
  public:
    DelayedPriorityCombine(FeatureMgr* feature_mgr_, Rag_t* rag_, MergePriority* priority_) :
        FeatureCombine(feature_mgr_, rag_), priority(priority_) {}

    void post_node_join(RagNode<unsigned int>* node_keep,
            RagNode<unsigned int>* node_remove)
    {
        FeatureCombine::post_node_join(node_keep, node_remove);
        
        int num_workers = priority->get_nworkers();
        for(RagNode_t::edge_iterator iter = node_keep->edge_begin();
                iter != node_keep->edge_end(); ++iter) {
            priority->add_dirty_edge_parallel(*iter, rand() % num_workers);

            RagNode_t* node = (*iter)->get_other_node(node_keep);
            for(RagNode_t::edge_iterator iter2 = node->edge_begin();
                    iter2 != node->edge_end(); ++iter2) {
                priority->add_dirty_edge_parallel(*iter2, rand() % num_workers);
            }
        }
    }

    void post_node_join_parallel(std::vector<std::pair<RagNode<unsigned int>*, RagNode<unsigned int>*> > &node_pairs) {

        cilk_for (std::vector<std::pair<RagNode<unsigned int>*, RagNode<unsigned int>*> >::iterator it = node_pairs.begin(); it != node_pairs.end(); ++it) {
            int worker_id = __cilkrts_get_worker_number();
            RagNode<unsigned int>* node_keep = it->first;
            for(RagNode_t::edge_iterator iter = node_keep->edge_begin();
                    iter != node_keep->edge_end(); ++iter) {
                priority->add_dirty_edge_parallel(*iter, worker_id);

                RagNode_t* node = (*iter)->get_other_node(node_keep);
                for(RagNode_t::edge_iterator iter2 = node->edge_begin();
                        iter2 != node->edge_end(); ++iter2) {
                    priority->add_dirty_edge_parallel(*iter, worker_id);
                }
            }
        }
        
        FeatureCombine::post_node_join_parallel(node_pairs);
    }

  private:
    MergePriority* priority;

};

class PriorityQCombine : public FeatureCombine {
  public:
    PriorityQCombine(FeatureMgr* feature_mgr_, Rag_t* rag_,
            MergePriorityQueue<QE>* priority_) :
        FeatureCombine(feature_mgr_, rag_), priority(priority_) {}


    virtual void post_edge_move(RagEdge<unsigned int>* edge_new,
            RagEdge<unsigned int>* edge_remove)
    {
        FeatureCombine::post_edge_move(edge_new, edge_remove); 
  
        int qloc = -1;
        try {
            qloc = edge_remove->get_property<int>("qloc");
        } catch (ErrMsg& msg) {
        }

        if (qloc>=0) {
            priority->invalidate(qloc+1);
        }
    }

    virtual void post_edge_join(RagEdge<unsigned int>* edge_keep,
            RagEdge<unsigned int>* edge_remove)
    {
        FeatureCombine::post_edge_join(edge_keep, edge_remove); 
        
        int qloc = -1;
        try {
            qloc = edge_remove->get_property<int>("qloc");
        } catch (ErrMsg& msg) {
        }
        if (qloc>=0) {
            priority->invalidate(qloc+1);
        }
    }

    void post_node_join(RagNode<unsigned int>* node_keep,
            RagNode<unsigned int>* node_remove)
    {
        FeatureCombine::post_node_join(node_keep, node_remove);

        for(RagNode_t::edge_iterator iter = node_keep->edge_begin();
                iter != node_keep->edge_end(); ++iter) {
            double val = feature_mgr->get_prob(*iter);
            double prev_val = (*iter)->get_weight(); 
            (*iter)->set_weight(val);
            Node_t node1 = (*iter)->get_node1()->get_node_id();
            Node_t node2 = (*iter)->get_node2()->get_node_id();

            QE tmpelem(val, std::make_pair(node1,node2));	

            int qloc = -1;
            try {
                qloc = (*iter)->get_property<int>("qloc");
            } catch (ErrMsg& msg) {
            }

            if (qloc>=0){
                if (val<prev_val) {
                    priority->heap_decrease_key(qloc+1, tmpelem);
                } else if (val>prev_val) {	
                    priority->heap_increase_key(qloc+1, tmpelem);
                }
            } else {
                priority->heap_insert(tmpelem);
            }        

        }    
    }
  private:
    MergePriorityQueue<QE>* priority;

};

class FlatCombine : public FeatureCombine {
  public:
    FlatCombine(FeatureMgr* feature_mgr_, Rag_t* rag_,
            std::vector<QE>* priority_) :
        FeatureCombine(feature_mgr_, rag_), priority(priority_) {}


    virtual void post_edge_move(RagEdge<unsigned int>* edge_new,
            RagEdge<unsigned int>* edge_remove)
    {
        FeatureCombine::post_edge_move(edge_new, edge_remove); 
   
        edge_new->set_weight(edge_remove->get_weight());	
        
        int qloc = -1;
        try {
            qloc = edge_remove->get_property<int>("qloc");
        } catch (ErrMsg& msg) {
        }

        if (qloc>=0){
            QE tmpelem(edge_new->get_weight(),
                    std::make_pair(edge_new->get_node1()->get_node_id(), 
                    edge_new->get_node2()->get_node_id()));	
            tmpelem.reassign_qloc(qloc, FeatureCombine::rag); 	
            (*priority).at(qloc) = tmpelem; 
        }
    }

    virtual void post_edge_join(RagEdge<unsigned int>* edge_keep,
            RagEdge<unsigned int>* edge_remove)
    {
        FeatureCombine::post_edge_join(edge_keep, edge_remove); 

        // FIX?: probability calculation should be deferred to end    
        double prob = feature_mgr->get_prob(edge_keep);
        edge_keep->set_weight(prob);	

        int qloc = -1;
        try {
            qloc = edge_remove->get_property<int>("qloc");
        } catch (ErrMsg& msg) {
        }
        
        if (qloc>=0) {
            (*priority).at(qloc).invalidate();
        }
    }

  private:
    std::vector<QE>* priority;
};

}
#endif
