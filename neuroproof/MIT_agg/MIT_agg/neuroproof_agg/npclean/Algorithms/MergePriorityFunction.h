#ifndef MERGEPRIORITYFUNCTION_H
#define MERGEPRIORITYFUNCTION_H

#include "../FeatureManager/FeatureMgr.h"
#include "../Rag/Rag.h"
#include <tr1/unordered_set>
#include "../Utilities/AffinityPair.h"

namespace NeuroProof {

typedef std::multimap<double, std::pair<unsigned int, unsigned int> > EdgeRank_t; 

class MergePriority {
  public:
    MergePriority(FeatureMgr* feature_mgr_, Rag_t* rag_) : 
                        feature_mgr(feature_mgr_), rag(rag_), synapse_mode(false),
                        kicked_out(0) {} 

    MergePriority(FeatureMgr* feature_mgr_, Rag_t* rag_, bool synapse_mode_) : 
                        feature_mgr(feature_mgr_), rag(rag_), synapse_mode(synapse_mode_),
                        kicked_out(0) {} 
    
    virtual ~MergePriority() {}

    virtual void initialize_priority(double threshold, bool use_edge_weight=false) = 0;

    virtual void initialize_random(double pthreshold) {}
    virtual void get_top_independent_edges (int nbd_size, vector<RagEdge_t*> &vec, bool and_prior=true) {}
    virtual void get_edges_parallel (vector<EdgeRank_t::iterator> &edges_to_remove_from_queue, vector<RagEdge_t*> &edges_to_process) {}
    virtual RagEdge_t* get_top_edge() = 0;

    virtual void add_dirty_edge(RagEdge_t* edge) = 0;
    virtual void add_dirty_edge_parallel(RagEdge_t* edge, int worker_id) {}

    bool valid_edge(RagEdge_t* edge)
    {
        if (!synapse_mode) {
            if (edge->is_preserve() || edge->is_false_edge()) {
                return false;
            }
            return true;
        } else {
            if (edge->is_false_edge()) {
                return false;
            } else if ((edge->get_size() < 625) && edge->is_preserve()) {
                return false;
            }
            return true;
        }
    }

    virtual bool empty() = 0;
    virtual int qlen(){ return 0;};	
    virtual int get_kout() {return 0;}; 	
    virtual void set_fileid(FILE* pid) {};

    // for parallel storage structure
    int get_nworkers(){return nworkers;};
    void set_nworkers(int num_w){nworkers = num_w;};

  protected:
    Rag_t* rag;
    FeatureMgr* feature_mgr;
    int kicked_out;

  private:
    bool synapse_mode;    
    int nworkers;

};

class ProbPriority : public MergePriority {
  public:
    ProbPriority(FeatureMgr* feature_mgr_, Rag_t* rag_) :
                    MergePriority(feature_mgr_, rag_), Epsilon(0.00001), kicked_fid(NULL) {}

    ProbPriority(FeatureMgr* feature_mgr_, Rag_t* rag_, bool synapse_mode_) :
                    MergePriority(feature_mgr_, rag_, synapse_mode_),
                    Epsilon(0.00001), kicked_fid(NULL) {}
    void initialize_priority(double threshold_, bool use_edge_weight=false);
    void initialize_random(double pthreshold);
    void clear_dirty();
    bool empty();
    RagEdge_t* get_edge_with_iter(EdgeRank_t::iterator iter);
    void get_top_independent_edges (int nbd_size, vector<RagEdge_t*> &vec, bool rand_prior = true);
    void get_edges_parallel (vector<EdgeRank_t::iterator> &edges_to_remove_from_queue, vector<RagEdge_t*> &edges_to_process);
    RagEdge_t* get_top_edge();
    void add_dirty_edge(RagEdge_t* edge);
   
    int qlen(){ return ranking.size();}	

    int get_kout(){return kicked_out;};	
    
    void set_fileid(FILE* pid){kicked_fid = pid;};

    void add_dirty_edge_parallel(RagEdge_t* edge, int worker_id);
    void initialize_dirty_edges_storage(int num_workers);
    void erase_from_dirty_edges_storage(Node_t node1, Node_t node2);

    // inline void add_dirty_edge_parallel(RagEdge_t* edge, int worker_id)
    // {
    //     // if (valid_edge(edge)) {
    //     //     edge->set_dirty(true);
    //     //     int node1 = edge->get_node1()->get_node_id();
    //     //     int node2 = edge->get_node2()->get_node_id();
    //     //     dirty_edges_storage[worker_id][node1][node2] = true;
    //     // }

    //     if (valid_edge(edge)) {
    //         edge->set_dirty(true);
    //         int node1 = edge->get_node1()->get_node_id();
    //         int node2 = edge->get_node2()->get_node_id();
    //         dirty_edges_storage[worker_id].insert(OrderedPair(node1, node2));
    //     }
    // }

  private:

    double threshold;
    const double Epsilon;
    typedef std::multimap<double, std::pair<Node_t, Node_t> > EdgeRank_t; 
    typedef std::tr1::unordered_set<OrderedPair, OrderedPair> Dirty_t; 
    EdgeRank_t ranking;
    Dirty_t dirty_edges;
    // std::vector<Dirty_t> dirty_edges_storage;
    std::vector<std::tr1::unordered_set<unsigned long int> > dirty_edges_storage;

    FILE* kicked_fid;

};

class MitoPriority : public MergePriority {
  public:
    MitoPriority(FeatureMgr* feature_mgr_, Rag_t* rag_) :
                    MergePriority(feature_mgr_, rag_), Epsilon(0.00001) {}

    void initialize_priority(double threshold_, bool use_edge_weight=false);
    
    //void initialize_random(double pthreshold);
    void clear_dirty();
    bool empty();
    RagEdge_t* get_top_edge();
    void add_dirty_edge(RagEdge_t* edge);

    int qlen(){ return ranking.size();}	

    int get_kout(){return kicked_out;};	
    

  private:

    double threshold;
    const double Epsilon;
    typedef std::multimap<double, std::pair<Node_t, Node_t> > EdgeRank_t; 
    typedef std::tr1::unordered_set<OrderedPair, OrderedPair> Dirty_t; 
    EdgeRank_t ranking;
    Dirty_t dirty_edges;

};





}

#endif
