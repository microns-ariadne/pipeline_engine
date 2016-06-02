#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <ctime>
#include "ws_alg.h"
#include "labeler.h"
#include "connectivity.h"

namespace fs = boost::filesystem;
namespace po = boost::program_options;
using namespace std;

// void savetotxt(int D, int N, int M, uint32_t *data)
// {
//   FILE *f = fopen("cpp_ws.txt", "w");
//   for (int d = 0; d < D; d++)
//     {
//       fprintf(f, "d = %d\n", d);
//       for (int i =0; i < N; i++)
//  {
//    for (int j = 0; j < M; j++)
//      fprintf(f, "%d ", data[d*N*M + i*M + j]);
//    fprintf(f, "\n");
//  }
//     }
//   fclose(f);
// }

void writeImage(
    string filePath, 
    string fileName, 
    int N, 
    int M, 
    uint32_t *data)
{
    vector<int> compressionParams;
    compressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compressionParams.push_back(0);
    
    cv::Mat out_mat(N, M, CV_8UC3);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            uint32_t datum = data[i*M + j];
            
            cv::Vec3b color;
        	color.val[0] = (uint8_t)(datum & 0xFF);
        	color.val[1] = (uint8_t)((datum >> 8) & 0xFF);
        	color.val[2] = (uint8_t)((datum >> 16) & 0xFF);
        	
        	out_mat.at<cv::Vec3b>(i,j) = color;
        }
    }
    
    string out_path = filePath + "/" + fileName;
    cv::imwrite(
        out_path, 
        out_mat, 
        compressionParams);
    
}

void parseOptions(
    int argc, 
    char **argv, 
    string *inputPath, 
	string *outputPath, 
	bool *is_labels_2D,
	bool *is_ws_2D, 
	bool *is_useBG)
{
    po::options_description desc("Watershed options");
    desc.add_options()
        ("help", "displays available options")
        ("labels-2D", "use 2D connectivity for labeling seeds")
        ("ws-2D", "use 2D connectivity for the BFS of watershed")
        ("useBG", "use 255 as background value")
        ("input-path", po::value< vector<string> >(), "input directory path")
        ("output-path", po::value< vector<string> >(), "output directory path");

    po::positional_options_description p;
    p.add("input-path", 1);
    p.add("output-path", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << endl;
        exit(0);
    }

    if (!vm.count("input-path") || !vm.count("output-path")) {
        cout << "Please specify the input and output directories" << endl;
        cout << desc << endl;
        exit(1);
    }

    if (vm.count("labels-2D")) {
        *is_labels_2D = true;
    }
    
    if (vm.count("ws-2D")) {
        *is_ws_2D = true;
    }
    
    if (vm.count("useBG")) {
        *is_useBG = true;
    }
    
    // warning: no boundary check here
    *inputPath = vm["input-path"].as< vector<string> >()[0];
    *outputPath = vm["output-path"].as< vector<string> >()[0];
}

int main(int argc, char** argv)
{
    bool is_labels_2D = false;
    bool is_ws_2D = false;
    bool is_useBG = false;
    string inputPath, outputPath;
    
    parseOptions(
        argc, 
        argv, 
        &inputPath,
        &outputPath, 
        &is_labels_2D,
        &is_ws_2D,
        &is_useBG);
    
    printf("===================================================================\n");
    printf("RUN PARAMS:\n");
    printf(" -- inputPath    : %s\n", inputPath.c_str());
    printf(" -- outputPath   : %s\n", outputPath.c_str());
    printf(" -- is_labels_2D : %d\n", (int)is_labels_2D);
    printf(" -- is_ws_2D     : %d\n", (int)is_ws_2D);
    printf(" -- is_useBG     : %d\n", (int)is_useBG);
    printf("===================================================================\n");
    
    clock_t readStart, markersStart, wsStart, writeStart, everythingDone;

    readStart = clock();
    // full path, file name
    vector<pair<string,string> > files;
    if (!fs::exists(inputPath) || !fs::is_directory(inputPath))
    {
        cout << inputPath << " is not a directory" << endl;
        return 1;
    }
    
    if (!fs::exists(outputPath) || !fs::is_directory(outputPath))
    {
        cout << outputPath << " is not a directory" << endl;
        return 1;
    }
    
    for (fs::directory_iterator it(inputPath); it != fs::directory_iterator(); it++)
    {
        if (!fs::is_regular_file(it->status())) {
            continue;
        }
        
        string path = it->path().string();
        string ext = it->path().extension().string();
        string fileName = it->path().filename().string();
        
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        if (fileName.length() >= 1 && fileName[0] == '.') {
            continue;
        }
        
        if (ext != ".png" && ext != ".tif" && ext != ".tiff") {
            continue;
        }
        
        files.push_back(pair<string, string>(path, fileName));
    }
    
    sort(files.begin(), files.end());
    
    cout << files.size() << " files in the input stack" << endl;
    
    uint8_t *img = NULL;
    uint64_t N, M, D;
    
    D = files.size();
    
    for (uint64_t f = 0; f < D; f++)
    {
        string filePath = files[f].first;
        string fileName = files[f].second;
        
        cout << "processing[" << f << "]: " << filePath << endl;
        
        cv::Mat cvimg = cv::imread((char*)filePath.c_str(), CV_LOAD_IMAGE_UNCHANGED);
        if (cvimg.empty())
        {
            cout << " -- Failed to load image " << filePath << endl;
            return 1;
        }
        
        if (cvimg.channels() != 1)
        {
            cout << " -- Too many channels (" << cvimg.channels() << ") in image " << filePath << endl;
            return 1;
        }
        
        if (f == 0) {
            N = cvimg.rows;
            M = cvimg.cols;
        } else {
            if ((cvimg.rows != N) || (cvimg.cols != M)) {
                printf("Image size [%d,%d] is unexpected (must be: [%lu,%lu])\n", 
                    cvimg.rows,
                    cvimg.cols,
                    N,
                    M);
                return 1;
            }
        }
        
        if (img == NULL) {
            img = new uint8_t[D*N*M];
        }
        
        for (uint64_t i = 0; i < N; i++) {
            for (uint64_t j = 0; j < M; j++) {
                img[f*N*M + i*M + j] = (uint8_t)cvimg.at<uint8_t>(i,j);
            }
        }    
    }
    
    markersStart = clock();
    uint32_t *markers = new uint32_t[D*N*M];
    uint64_t *indexBuffer = new uint64_t[D*N*M];
    
    for (uint64_t i = 0; i < D*N*M; i++) {
	    markers[i] = 0;
    }
    
    uint64_t nonzerocount = 0;
    for (uint64_t d = 0; d < D; d++) {
        for (uint64_t i = 0; i < N; i++) {
            for (uint64_t j = 0; j < M; j++) {
                if (is_useBG) {
                    if (img[d*N*M + i*M + j] == WS_BG_VAL) {
                        markers[d*N*M + i*M + j] = WS_BG_MARKER;
                        continue;
                    }
                }

                if (img[d*N*M + i*M + j] <= WS_MIN_SEED_VAL) {
                    markers[d*N*M + i*M + j] = 1;
                    nonzerocount++;
                }
            }
        }
    }
    
    cout << "non-zero markers counts: " << nonzerocount << endl;
    
    uint32_t currentLabel = 2;
    
    cout << "starting labeling" << endl;
    
    WatershedConnectivity labels_conn(is_labels_2D == true ? 4 : 6);
    WatershedConnectivity ws_conn(is_ws_2D == true ? 4 : 6);
    
    ComponentLabeler labeler(&labels_conn, D*N*M, indexBuffer);
    
    for (int d = 0; d < D; d++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
    			if (markers[d*N*M + i*M + j] == 1) {
                    markers[d*N*M + i*M + j] = currentLabel;
                    int currentLabelSize = labeler.labelComponent(
                        markers, 
                        D, N, M, 
                        d, i, j, 
                        currentLabel);
                    
                    if (currentLabelSize < WS_MIN_SEED_SIZE) {
                        // too small, zero-out everything that has just been marked
                        labeler.unlabelComponent(markers);
                        continue;
                    }
                    
                    //cout << "at " << i << " " << j << " component size = " << currentLabelSize << endl;
                    currentLabel++;
                    
                }
            }
        }    
    }
    
    wsStart = clock();
    do_watershed(
        D, N, M, 
        img, markers, indexBuffer, 
        &ws_conn);
    
    if (is_useBG) {
        for (uint64_t d = 0; d < D; d++) {
            for (uint64_t i = 0; i < N; i++) {
                for (uint64_t j = 0; j < M; j++) {
                    if (markers[d*N*M + i*M + j] == WS_BG_MARKER) {
                        markers[d*N*M + i*M + j] = 0;
                    }
                }
            }  
        }  
    }
     
    //savetotxt(D, N, M, markers);
    writeStart = clock();
    for (uint64_t d = 0; d < D; d++)
	{
	  cout << "writing to: " << (outputPath + "/" + files[d].second) << endl;
	  writeImage(outputPath, files[d].second, N, M, &markers[d*N*M]);
	}
	
    everythingDone = clock();

    cout << "********* TIME ***********" << endl;
    cout << "\t read: " << (markersStart - readStart)/(double)CLOCKS_PER_SEC << endl;
    cout << "\t markers: " << (wsStart - markersStart)/(double)CLOCKS_PER_SEC << endl;
    cout << "\t watershed: " << (writeStart - wsStart)/(double)CLOCKS_PER_SEC << endl;
    cout << "\t write: " << (everythingDone - writeStart)/(double)CLOCKS_PER_SEC << endl;
    cout << "\t total: " << (everythingDone - readStart)/(double)CLOCKS_PER_SEC << endl;
    
    cout << PROC_SUCCESS_STR << endl;
    
    return 0;
}
