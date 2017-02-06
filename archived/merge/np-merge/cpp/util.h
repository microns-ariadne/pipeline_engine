/*
 * This file contains parameters, includes, etc.
 */
#ifndef UTIL_H
#define UTIL_H

///////////////////////// includes ////////////////////////

#include <stdio.h>
#include <string>
#include <time.h>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

namespace fs = boost::filesystem;

///////////////////////// typedefs ////////////////////////

typedef uint32_t label_t;
typedef std::vector<int> block_t;
typedef std::pair< block_t, label_t> node_t;

///////////////////////// Constants ///////////////////////

const float VI_RATIO = 0.1;

const std::string LABELS_MAP_F ="labels_map.txt";

const boost::regex MERGE_FILE_RGX("merge_([0-9]+)_([0-9]+)_([0-9]+)_dir_([0-2]).txt");
const boost::regex SEGM_RGX("segmentation_([0-9]+)_([0-9]+)_([0-9]+)");

const std::string OUT = "out_";

///////////////////////// Methods ///////////////////////

std::map< std::vector< int > , std::string > findMatchingFiles(std::string workDir, boost::regex regex) {
    std::map< std::vector< int >, std::string> matches;

    fs::directory_iterator begin(workDir), end;
    std::vector<fs::directory_entry> v(begin, end);
    for (auto& f: v) {
        std::string fname = f.path().filename().string();
        boost::smatch match;

        if (boost::regex_match(fname, match, regex)) {
            std::vector<int> key;
            for (unsigned int i = 1; i < match.size(); ++i) {
                key.push_back(std::stoi(match[i]));
            }
            matches[key] = f.path().string();
        }
    }

    return matches;
}

#endif

