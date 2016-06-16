#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "util.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char ** argv) {
    std::string metaDir;
    std::string blockDir;
    block_t block;

    po::options_description desc("Relabel given segmentation_Z_Y_X.h5 files using " + LABELS_MAP_F);
    desc.add_options()
        ("help", "Produce help message")
        ("meta-dir", po::value< std::string >(&metaDir), "Meta directory")
        ("block-dir", po::value< std::string >(&blockDir), "Block directory")
        ("block", po::value< block_t >()->multitoken(), "Coordinates of the block to be relabelled")
        ("force", "Rewrite?");

    po::variables_map vm; 
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") > 0) {
        std::cout << desc << '\n';
        exit(0);
    }   

    if (vm.count("meta-dir") == 0 || 
        vm.count("block-dir") == 0 || 
        vm.count("block") == 0 ||
        (block = vm["block"].as< std::vector<int> >()).size() != 3) {
        std::cout << "Bad options\n";
        std::cout << desc << '\n';
        exit(0);
    }

    bool isForce = false;
    if (vm.count("force") > 0) {
        isForce = true;
    }

    std::cout << "Relabelling block: " << block[0] << ' ' << block[1] << ' ' << block[2] << '\n';
    auto start = std::chrono::system_clock::now();

    FILE* map_file = fopen((fs::path(metaDir) / fs::path(LABELS_MAP_F)).string().c_str(), "rb");
    int nMerges;
    label_t maxMergeLabel;
    fscanf(map_file, "%d%u", &nMerges, &maxMergeLabel);

    std::map< std::vector<int> , std::string > segmFiles = findMatchingFiles(blockDir, SEGM_RGX);
    if (segmFiles.count(block) == 0) {
        std::cout << "Block does not exist: " << block[0] << ' ' << block[1] << ' ' << block[2] << '\n';
        return 0;
    }

    std::string blockPath = segmFiles[block];
    std::string outBlockPath = (fs::path(blockDir) / fs::path(OUT + fs::path(blockPath).filename().string())).string(); // Prepend OUT to file name

    if (fs::exists(outBlockPath)) {
        if (isForce) {
            if (!fs::remove_all(outBlockPath)) {
                std::cerr << "Force mode is ON but can't remove " << outBlockPath << '\n';
                exit(1);
            }
        } else {
            std::cout << "SKIPPING relabel " << outBlockPath << '\n';
            exit(0);
        }
    }

    if (!fs::create_directory(outBlockPath)) {
        std::cerr << "Cannot create directory " << outBlockPath << '\n';
        exit(1);
    }

    std::cout << "Number of segmentation files found: " << segmFiles.size() << '\n';

    label_t maxID;
    label_t t_maxID;
    label_t blockStartLabel = maxMergeLabel + 1;

    for (auto &it : segmFiles) {

        if (it.first == block) {
            
            FILE* maxIdFile = fopen((it.second + "_maxID.txt").c_str(), "rb");
            fscanf(maxIdFile, "%d", &maxID);
            fclose(maxIdFile);
            
            maxIdFile = fopen((it.second + "_maxID_fixed.txt").c_str(), "rb");
            fscanf(maxIdFile, "%d", &t_maxID);
            fclose(maxIdFile);
            
            blockStartLabel += t_maxID;
            
            break;
        }
    }

    std::cout << "Block index for block " << 
        block[0] << ' ' << block[1] << ' ' << block[2] << 
        " starts from: " << blockStartLabel << '\n';
    
    std::cout << "MaxID for block " << 
        block[0] << ' ' << block[1] << ' ' << block[2] << 
        " is: " << maxID << '\n';
    
    std::cout << "T_MaxID for block " << 
        block[0] << ' ' << block[1] << ' ' << block[2] << 
        " is: " << t_maxID << '\n';
    
    std::cout << "BlockStartLabel for block " << 
        block[0] << ' ' << block[1] << ' ' << block[2] << 
        " is: " << blockStartLabel << '\n';
    
    std::vector< label_t > newLabels(maxID + 1);
    
    for (size_t i = 0; i < newLabels.size(); i++) {
        newLabels[i] = 0;
    }
    
    for (int i = 0; i < nMerges; ++i) {
        std::vector< int > ind(3);
        label_t label, newLabel;
        fscanf(map_file, "%d %d %d %u %u", &ind[0], &ind[1], &ind[2], &label, &newLabel);

        if (ind == block) {
            newLabels[label] = newLabel;
        }
    }
    fclose(map_file);

    std::cout << "Changing lables on block: " << blockPath << '\n';

    std::vector< std::string > images;
    for (fs::directory_iterator it(blockPath); it != fs::directory_iterator(); it++) {
        images.push_back(it->path().string());
    }   
    std::sort(images.begin(), images.end());
    
    for (size_t d = 0; d < images.size(); ++d) {
        cv::Mat image = cv::imread(images[d].c_str(), CV_LOAD_IMAGE_UNCHANGED);

        for (int p = 0; p < image.rows * image.cols; ++p) {
            uint32_t blu = image.data[3 * p + 0];
            uint32_t grn = image.data[3 * p + 1];
            uint32_t red = image.data[3 * p + 2];

            label_t id = (red << 16) | (grn << 8) | blu;

            if (newLabels[id] > 0) {
                id = newLabels[id];
            } else if (id > 0) {
                id += blockStartLabel;
            }

            image.data[3 * p + 0] = (uint8_t)(id & 255);
            image.data[3 * p + 1] = (uint8_t)((id >> 8) & 255);
            image.data[3 * p + 2] = (uint8_t)((id >> 16) & 255);
        }

        cv::imwrite((fs::path(outBlockPath) / fs::path(OUT + fs::path(images[d]).filename().string())).string(), image);
    }

    auto end = std::chrono::system_clock::now();
    std::cout << "DONE: Relabeling block " << blockPath << '\n';
    std::cout << "    Time: " << (end - start).count() / 1000000.0 << '\n';
    std::cout << "-= PROC SUCCESS =-" << std::endl;    
    return 0;
}

