#include "VolumeLabelData.h"
#include <tr1/unordered_set>
#include <iostream>
#include <fstream>
#include <string>
#include <vigra/tiff.hxx>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace NeuroProof;
using std::vector;
using std::tr1::unordered_set;

VolumeLabelPtr VolumeLabelData::create_volume()
{
    return VolumeLabelPtr(new VolumeLabelData); 
}

VolumeLabelPtr VolumeLabelData::create_volume(int xsize, int ysize, int zsize)
{
    VolumeLabelData* volumedata = new VolumeLabelData;
    vigra::TinyVector<long long unsigned int,3> shape(xsize, ysize, zsize);
    volumedata->reshape(shape);
    
    return VolumeLabelPtr(volumedata); 
}

VolumeLabelPtr VolumeLabelData::create_volume_meta(
        const char * h5_name, const char * dset, bool use_transforms)
{


    std::string line;
    std::ifstream myfile(h5_name);
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t count = 0;
    std::vector<std::string> image_paths; 
    if (myfile.is_open()) {
      while (getline(myfile, line)) {
        std::cout << line << '\n';
        /*uint32_t w,h;

        vigra::TiffImage * tiff = TIFFOpen(line.c_str(), "r");
        TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &h);
        vigra::MultiArrayView<2, vigra::RGBValue<unsigned char>>* img =
            new vigra::MultiArray<2, vigra::RGBValue<unsigned char> >(w,h);
        vigra::importTiffImage(tiff, destImage(*img));
        images.push_back(img);
        TIFFClose(tiff);
        width = w;
        height = h;
        count++;
        std::cout << "did something" << '\n';*/
        image_paths.push_back(line);
        count++;
      } 
    }

    //vigra::HDF5ImportInfo info(h5_name, dset);
    //vigra_precondition(info.numDimensions() == 3, "Dataset must be 3-dimensional.");
    cv::Mat image;
    image = cv::imread((const char*)(image_paths[0]).c_str(), CV_LOAD_IMAGE_UNCHANGED);
    std::cout << "Image dimensions are: " << image.dims << "\n";
    std::cout << "Image rows are: " << image.rows << "\n";
    std::cout << "Image cols are: " << image.cols << "\n";
    std::cout << "Image flags are: " << image.flags << "\n";
    std::cout << "Image datatype is: " << image.type() << "\n";
    std::cout << "Image channel count is: " << image.channels() << "\n";
    if (!image.data) {
      std::cout << "Error image doesn't have data \n";
    }
    width = image.cols;
    height = image.rows;

    VolumeLabelData* volumedata = new VolumeLabelData;
    //vigra::TinyVector<long long unsigned int,3> shape(info.shape().begin());
    vigra::TinyVector<long long unsigned int,3> shape(count, width, height);
    volumedata->reshape(shape);
    vigra::Shape3 data_stride = volumedata->stride();
    uint32_t* volumedata_data = volumedata->data();
    std::cout << "stride is " << data_stride << "\n";
    for (int z = 0; z < count; z++) {
      cv::Mat image_slice = cv::imread((const char*)(image_paths[z]).c_str(), CV_LOAD_IMAGE_UNCHANGED);
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

          cv::Vec3b vec = image_slice.at<cv::Vec3b>(y,x);
          volumedata_data[data_stride[1]*y+data_stride[2]*x + data_stride[0]*z] =
            ((uint32_t)vec[2]) * (1<<16) + ((uint32_t)vec[1])*(1<<8) + ((uint32_t)vec[0]); 
        }
      }
    }
    //vigra::readHDF5(info, *volumedata);

    if (use_transforms) {
        // looks for a dataset called transforms which is a label
        // to label mapping
        try {
            vigra::HDF5ImportInfo info(h5_name, "transforms");
            vigra::TinyVector<long long unsigned int,2> tshape(info.shape().begin()); 
            vigra::MultiArray<2,long long unsigned int> transforms(tshape);
            vigra::readHDF5(info, transforms);

            for (int row = 0; row < transforms.shape(1); ++row) {
                volumedata->label_mapping[transforms(0,row)] = transforms(1,row);
            }
            // rebase all of the labels so the initial label hash is empty
            volumedata->rebase_labels();
        } catch (std::runtime_error& err) {
        }
    }

    return VolumeLabelPtr(volumedata); 
}

VolumeLabelPtr VolumeLabelData::create_volume_from_images_seg(
        std::vector<std::string>& file_names)
{
    assert(!file_names.empty());
    vigra::ImageImportInfo info_init(file_names[0].c_str());
        
    VolumeLabelData *volumedata = new VolumeLabelData;
    volumedata->reshape(
        vigra::MultiArrayShape<3>::type(
            info_init.width(),
            info_init.height(), 
            file_names.size()));

    for (int i = 0; i < file_names.size(); ++i) {
        
        vigra::ImageImportInfo info(file_names[i].c_str());
        vigra::BRGBImage image(info.width(), info.height());
        vigra::importImage(info,destImage(image));
        
        for (int y = 0; y < int(info.height()); ++y) {
            for (int x = 0; x < int(info.width()); ++x) {
                
                //std::cout << "image(x,y): " << image(x,y) << std::endl;
                
                uint32_t val_1 = image(x,y)[0];
                uint32_t val_2 = image(x,y)[1];
                uint32_t val_3 = image(x,y)[2];
                
                // std::cout << "val_1: " << val_1 << std::endl;
                //                 std::cout << "val_2: " << val_2 << std::endl;
                //                 std::cout << "val_3: " << val_3 << std::endl;
                
                val_1 <<= 16;
                val_2 <<= 8;
                
                uint32_t val_res = val_1 | val_2 | val_3;
                
                // std::cout << "val_res: " << val_res << std::endl;
                
                (*(VolumeData<Label_t> *)volumedata)(x,y,i) = val_res; 
                
                // std::cout << "seg: " << (*(VolumeData<Label_t> *)volumedata)(x,y,i) << std::endl;
            }
        }  
    }

    return boost::shared_ptr<VolumeLabelData>(volumedata);
}

VolumeLabelPtr VolumeLabelData::create_volume(
        const char * h5_name, const char * dset, bool use_transforms)
{
    std::string h5_name_string(h5_name);
    int meta = h5_name_string.find(".meta");
    if (meta != std::string::npos) {
      return create_volume_meta(h5_name, dset, use_transforms);
    }
    vigra::HDF5ImportInfo info(h5_name, dset);
    vigra_precondition(info.numDimensions() == 3, "Dataset must be 3-dimensional.");

    VolumeLabelData* volumedata = new VolumeLabelData;
    vigra::TinyVector<long long unsigned int,3> shape(info.shape().begin());
    volumedata->reshape(shape);
    vigra::readHDF5(info, *volumedata);

    if (use_transforms) {
        // looks for a dataset called transforms which is a label
        // to label mapping
        try {
            vigra::HDF5ImportInfo info(h5_name, "transforms");
            vigra::TinyVector<long long unsigned int,2> tshape(info.shape().begin()); 
            vigra::MultiArray<2,long long unsigned int> transforms(tshape);
            vigra::readHDF5(info, transforms);

            for (int row = 0; row < transforms.shape(1); ++row) {
                volumedata->label_mapping[transforms(0,row)] = transforms(1,row);
            }
            // rebase all of the labels so the initial label hash is empty
            volumedata->rebase_labels();
        } catch (std::runtime_error& err) {
        }
    }

    return VolumeLabelPtr(volumedata); 
}

void VolumeLabelData::get_label_history(Label_t label, std::vector<Label_t>& member_labels)
{
    member_labels = label_remapping_history[label];
}

void VolumeLabelData::reassign_label(Label_t old_label, Label_t new_label)
{
    // do not allow label reassignment unless the stack was originally rebased
    // all stacks are read in rebased anyway so this should never execute
    assert(label_mapping.find(old_label) == label_mapping.end());

    label_mapping[old_label] = new_label;

    for (std::vector<Label_t>::iterator iter = label_remapping_history[old_label].begin();
            iter != label_remapping_history[old_label].end(); ++iter) {
        label_mapping[*iter] = new_label;
    }

    // update the mappings of all labels previously mapped to the
    // old label
    label_remapping_history[new_label].push_back(old_label);
    label_remapping_history[new_label].insert(label_remapping_history[new_label].end(),
            label_remapping_history[old_label].begin(), label_remapping_history[old_label].end());
    label_remapping_history.erase(old_label);
} 


void VolumeLabelData::split_labels(Label_t curr_label, vector<Label_t>& split_labels)
{
    vector<Label_t>::iterator split_iter = split_labels.begin();
    ++split_iter;
    label_mapping.erase(split_labels[0]);

    for (; split_iter != split_labels.end(); ++split_iter) {
        label_mapping[*split_iter] = split_labels[0];
        label_remapping_history[(split_labels[0])].push_back(*split_iter);
    }

    unordered_set<Label_t> split_labels_set;
    for (int i = 0; i < split_labels.size(); ++i) {
        split_labels_set.insert(split_labels[i]);
    }
    
    vector<Label_t> base_labels;
    for (std::vector<Label_t>::iterator iter = label_remapping_history[curr_label].begin();
            iter != label_remapping_history[curr_label].end(); ++iter) {
        if (split_labels_set.find(*iter) == split_labels_set.end()) {
            base_labels.push_back(*iter);
        } 
    }
    label_remapping_history[curr_label] = base_labels;
}

void VolumeLabelData::rebase_labels()
{
    if (!label_mapping.empty()) {
        // linear pass throw entire volume if remappings have occured
        for (VolumeLabelData::iterator iter = this->begin(); iter != this->end(); ++iter) {
            if (label_mapping.find(*iter) != label_mapping.end()) {
                *iter = label_mapping[*iter];
            }
        }
    }
    label_remapping_history.clear();
    label_mapping.clear();
}



