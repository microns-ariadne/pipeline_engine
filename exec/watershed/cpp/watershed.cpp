#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <ctime>
#include "ws_alg.h"
#include "labeler.h"

namespace fs = boost::filesystem;
using namespace std;

uint64_t MIN_SEED_VAL = 0;
uint64_t MIN_SEED_SIZE = 5;

void transformColors(cv::Mat &markers, cv::Mat &outputMarkers)
{
  // assumes the input markers are of 32-bit
  // the output is 8-bit colored markers
  cv::Mat colorMarkers(markers.size(), CV_8U);
  colorMarkers = cv::Scalar::all(0);
  map<uint32_t, int> cmap;
  int ncolors = 1;
  for (int i = 0; i < markers.rows; i++)
    for (int j = 0; j < markers.cols; j++)
      {
	uint32_t markerid = markers.at<uint32_t>(i,j);
	if (markerid > ((uint32_t)-100))
	  {
	    uint8_t &target = colorMarkers.at<uint8_t>(i,j);
	    target = 0;
	  }
	if (markerid > 0) {
	  if (cmap.find(markerid) == cmap.end())
	    {
	      cmap[markerid] = ncolors++;
	      ncolors %= 256;
	      if (ncolors == 0)
		ncolors = 1;	      
	    }
	  uint8_t &target = colorMarkers.at<uint8_t>(i,j);
	  target = cmap[markerid];
	}
      }
  outputMarkers = colorMarkers;
}

void savetotxt(int D, int N, int M, uint32_t *data)
{
  FILE *f = fopen("cpp_ws.txt", "w");
  for (int d = 0; d < D; d++)
    {
      fprintf(f, "d = %d\n", d);
      for (int i =0; i < N; i++)
	{
	  for (int j = 0; j < M; j++)
	    fprintf(f, "%d ", data[d*N*M + i*M + j]);
	  fprintf(f, "\n");
	}
    }
  fclose(f);
}

void writeImage(string filePath, string fileName, int N, int M, uint32_t *data)
{
  vector<int> compressionParams;
  compressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(0);

  // cv::Mat tmpmat2 = cv::Mat(N, M, CV_16U, (void *)data);
  // cv::Mat tmpmat2 = cv::Mat(N, M, CV_8SC4, (void *)data);
  cv::Mat tmpmat2(N, M, CV_8UC4);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < M; j++)
      {
	// this conversion is so stupid, that i can't even...
	uint32_t datum = data[i*M + j];
	cv::Vec4b color;
	color.val[0] = (uint8_t)(datum & 0xFF);
	color.val[1] = (uint8_t)((datum >> 8) & 0xFF);
	color.val[2] = (uint8_t)((datum >> 16) & 0xFF);
	// we're flipping the ALPHA channel to make it visible
	// in normal people's viewers
	color.val[3] = (uint8_t)(255 - ((datum >> 24) & 0xFF));
	//color.val[3] = 255;
	/*if (i < 3 && j < 3)
	  cout << datum << " -> color vals: " << (uint32_t)color.val[0] << ", " <<
	    (uint32_t) color.val[1] << ", " << 
	    (uint32_t)color.val[2] << ", " << 
	    (uint32_t)color.val[3] << endl;*/
	tmpmat2.at<cv::Vec4b>(i,j) = color;
	//tmpmat2.at<uint32_t>(i,j) = data[i*M + j];
      }
  cv::imwrite(filePath + "/" + fileName, tmpmat2, compressionParams);
  cout << "done writing" << endl;  
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    cout << "Usage " << argv[0] << " inputStackDir outputWatershedDir" << endl;
    return 1;
  }
  string stackPath = argv[1];
  string outputPath = argv[2];
  cout << "Computing watershed on stack: " << stackPath << endl;
  cout << "Oversegmentation written to: " << outputPath << endl;

  clock_t readStart, markersStart, wsStart, writeStart, everythingDone;

  readStart = clock();
  // full path, file name
  vector<pair<string,string> > files;
  if (!fs::exists(stackPath) || !fs::is_directory(stackPath))
    {
      cout << stackPath << " is not a directory" << endl;
      return 1;
    }
  if (!fs::exists(outputPath) || !fs::is_directory(outputPath))
    {
      cout << outputPath << " is not a directory" << endl;
      return 1;
    }

  for (fs::directory_iterator it(stackPath); it != fs::directory_iterator(); it++)
    {
      if (!fs::is_regular_file(it->status()))
		continue;
      string path = it->path().string();
      string ext = it->path().extension().string();
      string fileName = it->path().filename().string();
      transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (fileName.length() >= 1 && fileName[0] == '.')
		continue;
      if (ext != ".png" && ext != ".tif" && ext != ".tiff")
		continue;
      files.push_back(pair<string, string>(path, fileName));
    }
  sort(files.begin(), files.end());  
  cout << files.size() << " files in the input stack" << endl;

  uint8_t *img = NULL;
  uint64_t N, M, D = files.size();
  //D = min((int)files.size(), 2);

  int rank[256];
  for (uint64_t f = 0; f < D; f++)
    {
      string filePath = files[f].first;
      string fileName = files[f].second;
      cout << "processing " << filePath << endl;
      cv::Mat cvimg = cv::imread((char*)filePath.c_str(), CV_LOAD_IMAGE_UNCHANGED);
      if (cvimg.empty())
	{
	  cout << "Failed to load image " << filePath << endl;
	  return 1;
	}

      if (cvimg.channels() != 1)
	{
	  cout << "Too many channels (" << cvimg.channels() << ") in image " << filePath << endl;
	  return 1;
	}

      N = cvimg.rows;
      M = cvimg.cols;
      if (img == NULL)
	img = new uint8_t[D*N*M];
      for (uint64_t i = 0; i < N; i++)
	for (uint64_t j = 0; j < M; j++) {
	  img[f*N*M + i*M + j] = (uint8_t)cvimg.at<uint8_t>(i,j);
	  rank[img[f*N*M + i*M + j]] = 1;
	}
    }
  // rank the image
  rank[0] = 0;
  for (int i = 0; i < 256; i++)
    if (rank[i] == 0)
      rank[i] = rank[i-1];
    else
      rank[i] = rank[i-1]+1;
  for (uint64_t d = 0; d < D; d++)
    for (uint64_t i = 0; i < N; i++)
      for (uint64_t j = 0; j < M; j++)
	img[d*N*M + i*M + j] = rank[img[d*N*M + i*M + j]];

  // print this em for debug
  /*cout << "**** EM: *****" << endl;
  uint8_t maxv = 0;
  for (int d = 0; d < D; d++)
    {
      cout << "d = " << d << endl;
      for (int i = 0; i < N; i++)
	{
	  for (int j = 0; j < M; j++) {
	    cout << (uint32_t)img[d*N*M + i*M + j] << " ";
	    maxv = max(maxv, img[d*N*M + i*M + j]);
	  }
	  cout << endl;
	}
    }
	cout << "maxv = " << (uint32_t)maxv << endl;*/

  markersStart = clock();
  uint32_t *markers = new uint32_t[D*N*M];
  uint64_t *indexBuffer = new uint64_t[D*N*M];
  for (uint64_t i = 0; i < D*N*M; i++) {
	markers[i] = 0;
  }
  uint64_t nonzerocount = 0;
  for (uint64_t d = 0; d < D; d++)
	for (uint64_t i = 0; i < N; i++)
	  for (uint64_t j = 0; j < M; j++) {
#ifdef IS_BG	    
	    if (img[d*N*M + i*M + j] == BG_VAL) {
            markers[d*N*M + i*M + j] = BG_MARKER;
        }
#endif
	    if (img[d*N*M + i*M + j] <= MIN_SEED_VAL) {
		  markers[d*N*M + i*M + j] = 1;
		  nonzerocount++;
	    }
  }
  cout << "non-zero markers counts: " << nonzerocount << endl;

  uint32_t currentLabel = 2;
  cout << "starting labeling" << endl;
  //cout << "indexBuffer: " << indexBuffer << endl;
  ComponentLabeler labeler(D*N*M, indexBuffer);
  //cout << "labelere" << endl;
  for (int d = 0; d < D; d++)
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
		  {
			if (markers[d*N*M + i*M + j] == 1) {
			  markers[d*N*M + i*M + j] = currentLabel;
			  uint64_t currentLabelSize = labeler.labelComponent(markers, D, N, M, d, i, j, currentLabel);
			  //cout << "labeled " << currentLabelSize << endl;
			  if (currentLabelSize < MIN_SEED_SIZE)
				{
				  // too small, zero-out everything that has just been marked
				  labeler.unlabelComponent(markers);
				  // cout << "unlabeled" << endl;
				}
			  else
				{
				  //cout << "at " << i << " " << j << " component size = " << currentLabelSize << endl;
				  currentLabel++;
				}
			}
		  }


  /*cout << "**** INITIAL MARKERS: *****" << endl;
  for (int d = 0; d < D; d++)
    {
      cout << "d = " << d << endl;
      for (int i = 0; i < N; i++)
	{
	  for (int j = 0; j < M; j++)
	    cout << (uint32_t)markers[d*N*M + i*M + j] << " ";
	  cout << endl;
	}
    }*/

	  
  cout << "# of non-small components labeled: " << currentLabel-2+1 << endl;
  /*uint64_t nonzerononsmall = 0;
  for (uint64_t i = 0; i < D*N*M; i++)
    if (markers[i] > 0)
      nonzerononsmall++;
	  cout << "nonzerononsmall: " << nonzerononsmall << endl;*/

  wsStart = clock();
  do_watershed(D, N, M, img, markers, indexBuffer);

  // print this watershed for debug
  /*for (int d = 0; d < D; d++)
    {
      cout << "d = " << d << endl;
      for (int i = 0; i < N; i++)
	{
	  for (int j = 0; j < M; j++)
	    cout << markers[d*N*M + i*M + j] << " ";
	  cout << endl;
	}
    }*/

  //savetotxt(D, N, M, markers);
  writeStart = clock();
  // uint16_t *stupidpng = new uint16_t[N*M];
  for (uint64_t d = 0; d < D; d++)
	{
	  cout << "writing to: " << (outputPath + "/" + files[d].second) << endl;
	  // cv::Mat tmpmat2 = cv::Mat(N, M, CV_8SC3, (void *)markers);
	  //for (uint64_t i = 0; i < N*M; i++)
	  //stupidpng[i] = (uint16_t)markers[d*N*M + i];
	  //writeImage(outputPath, files[d].second, N, M, stupidpng);
	  writeImage(outputPath, files[d].second, N, M, &markers[d*N*M]);
	}
  everythingDone = clock();

  cout << "********* TIME ***********" << endl;
  cout << "\t read: " << (markersStart - readStart)/(double)CLOCKS_PER_SEC << endl;
  cout << "\t markers: " << (wsStart - markersStart)/(double)CLOCKS_PER_SEC << endl;
  cout << "\t watershed: " << (writeStart - wsStart)/(double)CLOCKS_PER_SEC << endl;
  cout << "\t write: " << (everythingDone - writeStart)/(double)CLOCKS_PER_SEC << endl;
  cout << "\t total: " << (everythingDone - readStart)/(double)CLOCKS_PER_SEC << endl;
	  
  return 0;
}
