#ifndef DJ_SET
#define DJ_SET

#include <vector>
#include <algorithm>
#include <map>

#include "util.h"

class DisjointSet {

public:
    std::map< node_t, node_t > parent;
    std::map< node_t, int > rank;

    std::map< node_t, node_t >::iterator find(node_t node) {
        if (parent.find(node) == parent.end()) {
            parent[node] = node;
            rank[node] = 1;
        }
        auto nodeIter = parent.find(node);
        while (nodeIter->first != nodeIter->second) {
            nodeIter = parent.find(nodeIter->second);
        }
        node_t root = nodeIter->first;

        nodeIter = parent.find(node);
        while (nodeIter->first != nodeIter->second) {
            auto currIter = nodeIter;
            nodeIter = parent.find(nodeIter->second);
            currIter->second = root;
        }

        return nodeIter;
    }

    bool join(block_t block1, label_t l1,
              block_t block2, label_t l2) {
        auto p1 = find(std::make_pair(block1, l1));
        auto p2 = find(std::make_pair(block2, l2));

        // std::cout << "Joining " << block1 << ' ' << l1 << ' ' << block2 << ' ' << l2 << '\n';

        if (p1 == p2) {
            return false;
        }
    
        auto r1 = rank.find(p1->first);
        auto r2 = rank.find(p2->first);

        if (r1->second < r2->second) {
            r2->second += r1->second;
            p1->second = p2->first;
        } else {
            r1->second += r2->second;
            p2->second = p1->first;
        }

        return true;
    }

    std::map< node_t, node_t > getMapping() {
        for (const auto& kv : parent) {
            find(kv.first);
        }
        return parent;
    }
};

#endif

