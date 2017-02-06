#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include <time.h>
#include <chrono>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "H5Cpp.h"
#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

#include "util.h"
#include "DisjointSet.h"

namespace fs = boost::filesystem;
namespace po = boost::program_options;

int main(int argc, char ** argv) {

    std::string metaDir;
    float viRatio;

    po::options_description desc("Combining all merge_Z_Y_X_DIR.txt files in meta directory to get label mapping");
    desc.add_options()
        ("help", "Produce help message")
        ("meta-dir", po::value< std::string >(&metaDir), "Meta directory")
        ("vi-ratio", po::value< float >(&viRatio)->default_value(VI_RATIO), "delta_MERGE/delta_SPLIT threshold for merging")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") > 0) {
        std::cout << desc << '\n';
        return 0;
    }

    if (vm.count("meta-dir") == 0) {
        std::cout << "Bad options\n";
        std::cout << desc << '\n';
        return 1;
    }

    DisjointSet ds;

    auto start = std::chrono::system_clock::now();

    std::map< std::vector<int> , std::string > mergeFiles = findMatchingFiles(metaDir, MERGE_FILE_RGX);

    std::cout << "Number of merge files found: " << mergeFiles.size() << '\n';

    for (auto& it : mergeFiles) {
        block_t ind(it.first.begin(), it.first.begin() + 3);
        std::string file = it.second;

        int dir = it.first[3];
        
        block_t otherInd = ind;
        otherInd[dir] ++;

        // Reading the merge file
        std::cout << "Merging: " << file << '\n';

        FILE* merge_file = fopen(file.c_str(), "rb");

        int m1, m2;
        fscanf(merge_file, "%d", &m1);

        for (int i = 0; i < m1; ++i) {
            label_t p, q;
            fscanf(merge_file, "%u%u", &p, &q);
            ds.join(ind, p, otherInd, q);
        }

        fscanf(merge_file, "%d", &m2);
        for (int i = 0; i < m2; ++i) {
            label_t p, q;
            int commonSize;
            double merge_voi, split_voi;
            fscanf(merge_file, "%u%u%d%lf%lf", &p, &q, &commonSize, &merge_voi, &split_voi);
            double all_voi = merge_voi + split_voi;
            if ( (all_voi < 0) && 
                 (split_voi < 0) &&
                 (merge_voi < 0 || (merge_voi > 0 && (-merge_voi / split_voi < viRatio)))) {
                ds.join(ind, p, otherInd, q);
            }
        }
        fclose(merge_file);
    }

    auto end = std::chrono::system_clock::now();
    std::cout << "Reading and merging - done" << '\n';
    std::cout << "   Time: " << (end - start).count() / 1000000.0 << '\n';

    std::cout << "\nPrinting mapping\n";
    start = std::chrono::system_clock::now();

    std::set<node_t> initialNodes;
    std::map<node_t, label_t> newLabels;

    label_t newLabelCounter = 1;

    auto mapping = ds.getMapping();

    for (auto kv : mapping) { // iterates in sorted order, hence in order of blocks
        initialNodes.insert(kv.first);

        auto newLabelIt = newLabels.find(kv.second);
        
        if (newLabelIt == newLabels.end()) {
            newLabels[kv.second] = newLabelCounter;
            newLabelCounter++;
        }
    }

    FILE* all_merges_f = fopen((boost::filesystem::path(metaDir) / boost::filesystem::path(LABELS_MAP_F)).string().c_str(), "wb");
    fprintf(all_merges_f, "%lu %d\n", mapping.size(), newLabelCounter - 1);

    for (auto kv : mapping) {
        block_t ind = kv.first.first;
        fprintf(all_merges_f, "%d %d %d %u %u\n", ind[0], ind[1], ind[2], kv.first.second, newLabels[kv.second]);
    }

    fclose(all_merges_f);

    end = std::chrono::system_clock::now();
    std::cout << "Number of nodes merged: " << initialNodes.size() << '\n';
    std::cout << "Resulting number of nodes: " << newLabelCounter - 1 << '\n';
    std::cout << "Time: " << (end - start).count() / 1000000.0 << '\n';
    std::cout << "-= PROC SUCCESS =-" << std::endl;
    return 0;
}

