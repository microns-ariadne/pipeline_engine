/*!
 * \file
 * Abstract interface for creating actions that should occur when
 * two nodes are merged together
*/

#ifndef RAGNODECOMBINEALG_H
#define RAGNODECOMBINEALG_H

#include "../Utilities/Glb.h"
#include <set>
#include <map>
#include <vector>

namespace NeuroProof {

template <typename Region>
class RagNode;

template <typename Region>
class RagEdge;

/*!
 * This abstract class defines actions that happen when merging
 * two nodes together.  The node to be removed will have all of its
 * edges removed in the process.  Currently, any behavior regarding
 * how two properties should be merged together should be defined
 * here.  Currently, the removed edge and node's properties are
 * discarded except for the boundary-size property maintained by the
 * rag node class.
*/
class RagNodeCombineAlg {
  public:
    /*!
     * Actions that should be done after an edge connected to an old node
     * is moved to another another node
     * \param edge_new pointer to rag edge that is newly created
     * \param edge_remove pointer to rag edge that will be removed
    */
    virtual void post_edge_move(RagEdge<Index_t>* edge_new,
            RagEdge<Index_t>* edge_remove) = 0;

    /*!
     * Actions that should be done after an edge connected to a node being
     * removed is merged with another edge from the other node
     * \param edge_keep pointer to rag edge that will be kept
     * \param edge_remove pointer to rag edge that will be removed
    */
    virtual void post_edge_join(RagEdge<Index_t>* edge_keep,
            RagEdge<Index_t>* edge_remove) = 0;

    // same as above but update many pairs in parallel
    // virtual void post_edge_join_parallel(std::set<std::pair<RagEdge<Index_t>*, RagEdge<Index_t>*> > &edge_pairs) {}
    virtual void post_edge_join_parallel(std::map<RagEdge<Index_t>*, std::set<RagEdge<Index_t>*> > &edge_pairs) {}
    virtual void post_node_join_parallel(std::vector<std::pair<RagNode<Index_t>*, RagNode<Index_t>*> > &node_pairs) {}

    /*!
     * Actions that should be done after internal node values are
     * merged between node_remove and node keep
     * \param node_keep pointer to rag node that will be kept
     * \param node_remove pointer to rag node that will be removed 
    */
    virtual void post_node_join(RagNode<Index_t>* node_keep,
            RagNode<Index_t>* node_remove) = 0;

    /*!
     * Virtual destructor to be reimplemented by derived classes
    */
    virtual ~RagNodeCombineAlg() {}
};

}

#endif


