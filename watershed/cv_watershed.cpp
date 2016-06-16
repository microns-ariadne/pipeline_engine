#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
using namespace std;

int MIN_SEED_SIZE = 5;

// issue: this might blow up when called on huge components
// should be pretty fast on small components
int labelComponent(cv::Mat &markers, int startY, int startX, int currentLabel, int previousLabel=1)
{
  int currentLabelCount = 1;
  vector<std::pair<int,int> > queue;
  queue.push_back(std::pair<int,int>(startY, startX));
  while (!queue.empty())
    {
      std::pair<int,int> current = queue.back();
      queue.pop_back();
      uint32_t &px = markers.at<uint32_t>(current.first, current.second);
      // 8-neighborhood
      for (int dy = -1; dy <= 1; dy++)
	for (int dx = -1; dx <= 1; dx++)
	  {
	    int nexty = current.first + dy;
	    int nextx = current.second + dx;
	    if (nexty < 0 || nextx < 0 || nexty >= markers.rows || nextx >= markers.cols)
	      continue;
	    uint32_t &next = markers.at<uint32_t>(nexty, nextx);
	    if (next != previousLabel)
	      continue;
	    next = currentLabel;
	    queue.push_back(pair<int,int>(nexty,nextx));
	    currentLabelCount++;
	  }
    }
  return currentLabelCount;
}

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

  vector<int> compressionParams;
  compressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(0);

  for (pair<string, string> entry : files)
    {
      string filePath = entry.first;
      string fileName = entry.second;
      cout << "processing " << filePath << endl;
      //cv::Mat img = cv::imread((char*)files[0].c_str(), CV_LOAD_IMAGE_UNCHANGED);     
      cout << "cv lod ... " << CV_LOAD_IMAGE_COLOR << endl;
      cv::Mat img = cv::imread((char*)filePath.c_str(), CV_LOAD_IMAGE_COLOR);
      if (img.empty())
	{
	  cout << "Failed to load image " << filePath << endl;
	  return 1;
	}

      /*if (img.channels() != 1)
	{
	cout << "Too many channels (" << img.channels() << ") in image " << files[0] << endl;
	return 1;
	}*/

      // this displays the window
      /*cv::namedWindow("MyWindow", CV_WINDOW_NORMAL);
	cv::imshow("MyWindow", img);
	cv::waitKey(0);
	cv::destroyWindow("MyWindow");*/

      cout << "dims, rows, cols: " << img.dims << " " << img.rows << " " << img.cols << endl;
      cout << "flags: " << img.flags << endl;
      cout << "elem size: " << img.elemSize() << endl;
      cout << "depth: " << img.depth() << endl;
      cout << "channels: " << img.channels() << endl;
      cout << "size: " << img.size() << endl;

      for (int i = 0; i < 10; i++)
	for (int j = 0; j < 10; j++)
	{
	  cv::Vec3b px = img.at<cv::Vec3b>(i,j);
	  cout << px << endl;
	}

      cout << "here" << endl;
      cv::Mat markers(img.size(), CV_32S, 0);
      for (int i = 0; i < img.rows; i++)
	for (int j = 0; j < img.cols; j++)
	  {
	    cv::Vec3b px = img.at<cv::Vec3b>(i,j);
	    if (px.val[0] == 0)
	      {
		uint32_t &target = markers.at<uint32_t>(i,j);
		target = 1;
	      }
	  }
      cout << endl << endl << "markers channels: " << markers.channels() << endl;
      for (int i = 0; i < 10; i++)
	for (int j = 0; j < 10; j++)
	{
	  cv::Vec3b px = markers.at<cv::Vec3b>(i,j);
	  uint32_t px2 = markers.at<uint32_t>(i,j);
	  cout << px << "(" << px2 << ")" << endl;
	}

      cv::imwrite(outputPath + "/initial_markers.png", markers, compressionParams);
      // cout << "markers.channels: " << markers.channels() << endl;

      int currentLabel = 2;
      for (int i = 0; i < markers.rows; i++)
	{
	  for (int j = 0; j < markers.cols; j++)
	    {
	      uint32_t &px = markers.at<uint32_t>(i,j);
	      if (px == 1) {
		px = currentLabel;
		int currentLabelSize = labelComponent(markers, i, j, currentLabel);
		// cout << "clc: " << currentLabelCount << endl;
		if (currentLabelSize <= MIN_SEED_SIZE)
		  {
		    // too small, zero-out
		    px = 0;
		    labelComponent(markers, i, j, 0, currentLabel);
		  }
		else
		  {
		    //cout << "at " << i << " " << j << " component size = " << currentLabelSize << endl;
		    currentLabel++;
		  }
	      }
	    }
	}
      cout << "# of non-small components labeled: " << currentLabel-2 << endl;
      cv::imwrite(outputPath + "/no_small_markers.png", markers, compressionParams);

      /*for (int i = 10; i < 13; i++)
	{
	for (int j = 0; j < markers.cols; j++)
	cout << markers.at<uint32_t>(i,j) << " ";
	cout << endl;
	}*/

      // markers pre-watershed:
      /*cv::Mat colorMarkers;
	transformColors(markers, colorMarkers);
	cv::applyColorMap(colorMarkers, colorMarkers, cv::COLORMAP_JET);
	cv::namedWindow("MyWindow", CV_WINDOW_NORMAL);
	cv::imshow("MyWindow", colorMarkers);
	cv::waitKey(0);
	cv::destroyWindow("MyWindow");*/
  
      // issue: CV's watershed requires img to be of 3-channels!
      // stupid!
      cv::watershed(img, markers);
      cv::imwrite(outputPath + "/after_ws.png", markers, compressionParams);
      // mark all borders as zeros
      for (int i = 0; i < markers.rows; i++)
	{
	for (int j = 0; j < markers.cols; j++)
	  {
	    uint32_t &px = markers.at<uint32_t>(i,j);
	    if (px > ((uint32_t)-100))
	      px = 0;
	    // cout << px << " ";	  
	  }
	//cout << endl;
	}

      /*cout << "****** after watershed: " << endl;
	for (int i = 10; i < 13; i++)
	{
	for (int j = 0; j < markers.cols; j++)
	cout << markers.at<uint32_t>(i,j) << " ";
	cout << endl;
	}*/
  

      /*cv::namedWindow("MyWindow", CV_WINDOW_NORMAL);
	cv::Mat wsmarkers;
	transformColors(markers, wsmarkers);*/
      /*cout << "****** after watershed-colors: " << endl;
	for (int i = 10; i < 13; i++)
	{
	for (int j = 0; j < wsmarkers.cols; j++)
	cout << wsmarkers.at<uint8_t>(i,j) << " ";
	cout << endl;
	}*/
      /*cv::applyColorMap(wsmarkers, wsmarkers, cv::COLORMAP_JET);
	cv::imshow("MyWindow", wsmarkers);
	cv::waitKey(0);
	cv::destroyWindow("MyWindow");*/

      cout << "writing to: " << (outputPath + "/" + fileName) << endl;
      cv::imwrite(outputPath + "/" + fileName, markers, compressionParams);
      cout << "done writing" << endl;
      break;
    }

  return 0;
}
