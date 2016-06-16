/*!
 * Defines a class that inherits from Vigra multiarray. It assumes
 * that all data volumes will have 3 dimensions.  If a 2D or 1D
 * volume is needed, the volume shape should be X,Y,1 or X,1,1
 * respectively.  Please examine documentation in Vigra 
 * (http://hci.iwr.uni-heidelberg.de/vigra) for more information on
 * how to use multiarray and the algorithms that are available for
 * this data type.
 *
 * \author Stephen Plaza (plaza.stephen@gmail.com)
*/

#ifndef VOLUMEDATA_H
#define VOLUMEDATA_H

/// FIX
#define NP_OUTPUT_FILES
/// FIX

#include <vigra/multi_array.hxx>

// used for importing h5 files
#include <vigra/hdf5impex.hxx>
#include <vigra/impex.hxx>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <string>
#include "../Utilities/ErrMsg.h"
#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <cilk/cilk.h>
namespace NeuroProof {

// forward declaration
template <typename T>
class VolumeData;

// defines some of the common volume types used in NeuroProof
//typedef VolumeData<double> VolumeProb;
// NOTE(TFK): Switching from double to float does not appear to impact
//   predictions at all, and cuts resident set size down by about 1 GB.
typedef VolumeData<float> VolumeProb;

typedef VolumeData<unsigned char> VolumeGray;
typedef boost::shared_ptr<VolumeProb> VolumeProbPtr;
typedef boost::shared_ptr<VolumeGray> VolumeGrayPtr;

/*!
 * This class defines a 3D volume of any type.  In particular,
 * it inherits properties of multiarray and provides functionality
 * for creating new volumes given h5 input.  This interface stipulates
 * that new VolumeData objects get created on the heap and are
 * encapsulated in shared pointers.
*/
template <typename T>
class VolumeData : public vigra::MultiArray<3, T> {
  public:
    /*!
     * Static function to create an empty volume data object.
     * \return shared pointer to volume data
    */
    static boost::shared_ptr<VolumeData<T> > create_volume();

    /*!
     * Static function to create a volume data object from 
     * an h5 file. For now, input h5 files are assumed to be Z x Y x X.
     * TODO: allow user-defined axis specification.
     * \param h5_name name of h5 file
     * \param dset name of dataset
     * \return shared pointer to volume data
    */
    static boost::shared_ptr<VolumeData<T> > create_volume(
            const char * h5_name, const char * dset);
    
    /*!
     * Static function to create an array of volume data from an h5
     * assumed to have format X x Y x Z x num channels.  Each channel,
     * will define a new volume data in the array. TODO: allow user-defined
     * axis specification.
     * \param h5_name name of h5 file
     * \param dset name of dset
     * \return vector of shared pointers to volume data
    */
    static std::vector<boost::shared_ptr<VolumeData<T> > >
        create_volume_array(const char * h5_name, const char * dset);

    static std::vector<boost::shared_ptr<VolumeData<T> > >
        create_volume_array_NEW(const char * h5_name, const char * dset);
    
    /*!
     * Static function to create an array of volume data from an h5
     * assumed to have format X x Y x Z x num channels.  Each channel,
     * will define a new volume data in the array. The dim1size field indicates
     * that a subset of the data will be used.  TODO: allow user-defined
     * axis specification.
     * \param h5_name name of h5 file
     * \param dset name of dset
     * \param dim1size size of the first dimension of a companion volume
     * \return vector of shared pointers to volume data
    */
    static std::vector<boost::shared_ptr<VolumeData<T> > >
        create_volume_array(const char * h5_name, const char * dset, unsigned int dim1size);


    /*!
     * Static function to create a 3D image volume from a list of
     * 2D image files.  While many 2D image formats are supported
     * (see vigra documentation), this function will only work with
     * 8-bit, grayscale values for now.
     * \param file_names array of images in correct order
     * \return shared pointer to volume data
    */
    
    static std::vector<boost::shared_ptr<VolumeData<T> > > create_volume_from_images(
        std::vector<std::string>& file_names);
    

    /*!
     * Write volume data to disk assuming Z x Y x X in h5 output.
     * \param h5_name name of h5 file
     * \param h5_path path to h5 dataset
    */
    void serialize(const char* h5_name, const char * h5_path); 

    // TODO: provide some functionality for enabling visualization

  protected:
    /*!
     * Private definition of constructor to prevent stack allocation.
    */
    VolumeData() : vigra::MultiArray<3,T>() {}
    
    /*!
     * Copy constructor to create VolumeData from a multiarray view.  It
     * just needs to call the multiarray constructor with the view.
     * \param view_ view to a multiarray
    */
    VolumeData(const vigra::MultiArrayView<3, T>& view_) : vigra::MultiArray<3,T>(view_) {}
};


template <typename T>
boost::shared_ptr<VolumeData<T> > VolumeData<T>::create_volume()
{
    return boost::shared_ptr<VolumeData<T> >(new VolumeData<T>); 
}

template <typename T>
boost::shared_ptr<VolumeData<T> > VolumeData<T>::create_volume(
        const char * h5_name, const char * dset)

{
    vigra::HDF5ImportInfo info(h5_name, dset);
    vigra_precondition(info.numDimensions() == 3, "Dataset must be 3-dimensional.");

    VolumeData<T>* volumedata = new VolumeData<T>;
    vigra::TinyVector<long long unsigned int,3> shape(info.shape().begin());
    volumedata->reshape(shape);
    // read h5 file into volumedata with correctly set shape
    vigra::readHDF5(info, *volumedata);

    return boost::shared_ptr<VolumeData<T> >(volumedata); 
}

template <typename T>
std::vector<boost::shared_ptr<VolumeData<T> > > 
    VolumeData<T>::create_volume_array(const char * h5_name, const char * dset)
{
    vigra::HDF5ImportInfo info(h5_name, dset);
    vigra_precondition(info.numDimensions() == 4, "Dataset must be 4-dimensional.");

    vigra::TinyVector<long long unsigned int,4> shape(info.shape().begin());
    vigra::MultiArray<4, T> volumedata_temp(shape);
    vigra::readHDF5(info, volumedata_temp);
    
    // since the X,Y,Z,ch is read in as ch,Z,Y,X transpose
    volumedata_temp = volumedata_temp.transpose();

    std::vector<VolumeProbPtr> vol_array;
    // vigra::TinyVector<long long unsigned int,3> shape2;

    // tranpose the shape dimensions as well
    // shape2[0] = shape[3];
    // shape2[1] = shape[2];
    // shape2[2] = shape[1];

    // for each channel, create volume data and push in array
    for (int i = 0; i < shape[0]; ++i) {
        VolumeData<T>* volumedata = new VolumeData<T>;
        vigra::TinyVector<vigra::MultiArrayIndex, 1> channel(i);
        (*volumedata) = volumedata_temp.bindOuter(channel); 
        
        vol_array.push_back(boost::shared_ptr<VolumeData<T> >(volumedata));
    }

    return vol_array; 
}

// STRIDE_1 stride1 (1, 1024, 1048576)
//#define STRIDE1_i 1
//#define STRIDE1_j 1024
//#define STRIDE1_k 1048576
//
//// STRIDE_2 stride2 (1, 100, 102400)
//#define STRIDE2_i 1
//#define STRIDE2_j 100
//#define STRIDE2_k 102400


#define TFK_TRANSPOSE_CILK_COARSEN 8

template <typename T>
void tfk_transpose_data_helper_serial(T* array1, T* array2, int i_begin, int i_end,
int j_begin, int j_end, int k_begin, int k_end, vigra::Shape3 stride1, vigra::Shape3 stride2) {
    int STRIDE1_i = stride1[0];
    int STRIDE1_j = stride1[1];
    int STRIDE1_k = stride1[2];

    int STRIDE2_i = stride2[0];
    int STRIDE2_j = stride2[1];
    int STRIDE2_k = stride2[2];

    for (int k = k_begin; k < k_end; k++) {
      for (int i = i_begin; i < i_end; i++) {
         for (int j = j_begin; j < j_end; j++) {
            //T tmp = array1[i*stride1[0] + j*stride1[1] + k*stride1[2]];
            //array2[k*stride2[0] + j*stride2[1] + i*stride2[2]] = tmp;
            //array3[k*stride2[0] + j*stride2[1] + i*stride2[2]] = tmp;
            T tmp = array1[i*STRIDE1_i + j*STRIDE1_j + k*STRIDE1_k];
            array2[k*STRIDE2_i + j*STRIDE2_j + i*STRIDE2_k] = tmp;
            //array3[k*STRIDE2_i + j*STRIDE2_j + i*STRIDE2_k] = tmp;
        }
      }
    }
}

template <typename T>
void tfk_transpose_data_helper(T* array1, T* array2, int i_begin, int i_end,
int j_begin, int j_end, int k_begin, int k_end, vigra::Shape3 stride1, vigra::Shape3 stride2) {

  int delta_i = i_end - i_begin;
  int delta_j = j_end - j_begin;
  int delta_k = k_end - k_begin;

  if (delta_i <= 32 && delta_j <= 32 && delta_k <= 32) {
    /*for (int i = i_begin; i < i_end; i++) {
      for (int j = j_begin; j < j_end; j++) {
        for (int k = k_begin; k < k_end; k++) {
            T tmp = array1[i*stride1[0] + j*stride1[1] + k*stride1[2]];
            array2[k*stride2[0] + j*stride2[1] + i*stride2[2]] = tmp;
            array3[k*stride2[0] + j*stride2[1] + i*stride2[2]] = tmp;
        }
      }
    }*/
    tfk_transpose_data_helper_serial(array1,array2, i_begin, i_end,
        j_begin, j_end, k_begin, k_end, stride1, stride2);
    return;
  }

  // otherwise divide and conquer

  if (delta_i >= delta_j && delta_i >= delta_k) {
    // divide on delta_i

    int begin_i_child1 = i_begin;
    int end_i_child1 = i_begin + delta_i/2;

    int begin_i_child2 = end_i_child1;
    int end_i_child2 = i_end;

    if (delta_i >= TFK_TRANSPOSE_CILK_COARSEN) {
      cilk_spawn tfk_transpose_data_helper(array1,array2, begin_i_child1, end_i_child1,
          j_begin, j_end, k_begin, k_end, stride1, stride2);
    } else {
      tfk_transpose_data_helper(array1,array2, begin_i_child1, end_i_child1,
          j_begin, j_end, k_begin, k_end, stride1, stride2);
    }
    tfk_transpose_data_helper(array1,array2, begin_i_child2, end_i_child2,
        j_begin, j_end, k_begin, k_end, stride1, stride2);
    //cilk_sync;
    return;
  }

  if (delta_j >= delta_i && delta_j >= delta_k) {
    // divide on delta_j

    int begin_j_child1 = j_begin;
    int end_j_child1 = j_begin + delta_j/2;

    int begin_j_child2 = end_j_child1;
    int end_j_child2 = j_end;

    if (delta_j >= TFK_TRANSPOSE_CILK_COARSEN) {
    cilk_spawn tfk_transpose_data_helper(array1,array2, i_begin, i_end,
        begin_j_child1, end_j_child1, k_begin, k_end, stride1, stride2);
    } else {
    tfk_transpose_data_helper(array1,array2, i_begin, i_end,
        begin_j_child1, end_j_child1, k_begin, k_end, stride1, stride2);
    }
    tfk_transpose_data_helper(array1,array2, i_begin, i_end,
        begin_j_child2, end_j_child2, k_begin, k_end, stride1, stride2);
    return;
  }

  if (delta_k >= delta_i && delta_k >= delta_j) {
    // divide on delta_k

    int begin_k_child1 = k_begin;
    int end_k_child1 = k_begin + delta_k/2;

    int begin_k_child2 = end_k_child1;
    int end_k_child2 = k_end;

    if (delta_k >= TFK_TRANSPOSE_CILK_COARSEN) {
    cilk_spawn tfk_transpose_data_helper(array1,array2, i_begin, i_end,
        j_begin, j_end, begin_k_child1, end_k_child1, stride1, stride2);
    } else {
    tfk_transpose_data_helper(array1,array2, i_begin, i_end,
        j_begin, j_end, begin_k_child1, end_k_child1, stride1, stride2);
    }
    tfk_transpose_data_helper(array1,array2, i_begin, i_end,
        j_begin, j_end, begin_k_child2, end_k_child2, stride1, stride2);
    return;
  }

}

template <typename T>
std::vector<boost::shared_ptr<VolumeData<T> > > 
    VolumeData<T>::create_volume_array_NEW(const char * h5_name, const char * dset)
{

    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();

    printf("create_volume_array_NEW - 1\n");
    vigra::HDF5ImportInfo info(h5_name, dset);
    vigra_precondition(info.numDimensions() == 3, "Dataset must be 3-dimensional.");

    now = boost::posix_time::microsec_clock::local_time();
    std::cout << std::endl << "baseline 1: " << (now - start).total_milliseconds() << " ms\n";

    printf("create_volume_array_NEW - 2\n");
    vigra::TinyVector<long long unsigned int, 3> shape(info.shape().begin());
    vigra::MultiArray<3, T> volumedata_temp2(shape);
    vigra::readHDF5(info, volumedata_temp2); // time: ~3.6 sec
    
    now = boost::posix_time::microsec_clock::local_time();
    std::cout << std::endl << "baseline 2: " << (now - start).total_milliseconds() << " ms\n";

    printf("create_volume_array_NEW - 3\n");
    // since the X,Y,Z,ch is read in as ch,Z,Y,X transpose

    std::cout << std::endl << "shape " << shape << std::endl;

    vigra::MultiArray<3, T> volumedata_temp(vigra::Shape3(info.shape()[2], info.shape()[1], info.shape()[0]));

    

    VolumeData<T>* volumedata_1 = new VolumeData<T>;
    *volumedata_1 = volumedata_temp;

    T* data_ptr = volumedata_temp2.data();
    T* data_ptr2 = volumedata_1->data();

    vigra::Shape3 data_stride = volumedata_temp2.stride();
    vigra::Shape3 data_stride_new = volumedata_1->stride();
    
    std::cout << std::endl << "stride1 " << data_stride << ", stride2 " << data_stride_new << std::endl;


    tfk_transpose_data_helper(data_ptr, data_ptr2, 0, shape[0], 0, shape[1], 0, shape[2], data_stride, data_stride_new); 
    //volumedata_temp = volumedata_temp.transpose(); // time: ~5.159
/*
      cilk_for (int _j = 0; _j < shape[1]/4; _j++) {
          cilk_for (int _i = 0; _i < shape[0]/4; _i++) {
            cilk_for (int _k = 0; _k < shape[2]/4; _k++) {
            
            int i_end = (_i+1)*4;
            int j_end = (_j+1)*4;
            int k_end = (_k+1)*4;
            if (i_end > shape[0]) i_end = shape[0];
            if (j_end > shape[0]) j_end = shape[1];
            if (k_end > shape[0]) k_end = shape[2];

            for (int i = _i*4; i < i_end; i++) {
              for (int j = _j*4; j < j_end; j++) {
                for (int k = _k*4; k < k_end; k++) {
            T tmp = data_ptr[i*data_stride[0] + j*data_stride[1] + k*data_stride[2]];
            data_ptr2[k*data_stride_new[0] + j*data_stride_new[1] + i*data_stride_new[2]] = tmp;
            data_ptr3[k*data_stride_new[0] + j*data_stride_new[1] + i*data_stride_new[2]] = tmp;
              } } }
        }
      }
    }*/
    //volumedata_temp = volumedata_temp2;
    //volumedata_temp =
    // NOTE(TFK): Try to avoid copying data while transposing. 
    //volumedata_temp.transpose(vigra::Shape3(2,1,0)); // time: ?
    
    now = boost::posix_time::microsec_clock::local_time();
    std::cout << std::endl << "baseline tweek2 3: " << (now - start).total_milliseconds() << " ms\n";

    printf("create_volume_array_NEW - 4\n");
    std::vector<VolumeProbPtr> vol_array;
    // vigra::TinyVector<long long unsigned int,3> shape2;

    // tranpose the shape dimensions as well
    // shape2[0] = shape[3];
    // shape2[1] = shape[2];
    // shape2[2] = shape[1];
    printf("create_volume_array_NEW - 5\n");
    // for each channel, create volume data and push in array
    //for (int i = 0; i < shape[0]; ++i) {
        //VolumeData<T>* volumedata_1 = new VolumeData<T>;
        //VolumeData<T>* volumedata_2 = new VolumeData<T>;
        
        //vigra::TinyVector<vigra::MultiArrayIndex, 1> channel(i);
        //(*volumedata) = volumedata_temp.bindOuter(channel); 

        //(*volumedata_1) = volumedata_temp;
        //(*volumedata_2) = volumedata_temp;

        boost::shared_ptr<VolumeData<T> > volume_shared_ptr = boost::shared_ptr<VolumeData<T> >(volumedata_1);
        vol_array.push_back(volume_shared_ptr);
        vol_array.push_back(volume_shared_ptr);
        
    now = boost::posix_time::microsec_clock::local_time();
    std::cout << std::endl << "baseline 4: " << (now - start).total_milliseconds() << " ms\n";
    //}
    printf("create_volume_array_NEW - 6\n");
    return vol_array; 
}

template <typename T>
std::vector<boost::shared_ptr<VolumeData<T> > > 
    VolumeData<T>::create_volume_array(const char * h5_name, const char * dset, unsigned int dim1size)
{
    vigra::HDF5ImportInfo info(h5_name, dset);
    vigra_precondition(info.numDimensions() == 4, "Dataset must be 4-dimensional.");

    vigra::TinyVector<long long unsigned int,4> shape(info.shape().begin());
    vigra::MultiArray<4, T> volumedata_temp(shape);
    vigra::readHDF5(info, volumedata_temp);
    
    // since the X,Y,Z,ch is read in as ch,Z,Y,X transpose
    volumedata_temp = volumedata_temp.transpose();

    std::vector<VolumeProbPtr> vol_array;
    vigra::TinyVector<long long unsigned int,3> shape2;

    // tranpose the shape dimensions as well
    shape2[0] = shape[3];
    shape2[1] = shape[2];
    shape2[2] = shape[1];

    // prediction must be the same size or larger than the label volume
    if (dim1size > shape2[0]) {
        throw ErrMsg("Label volume has a larger dimension than the prediction volume provided");
    }
    
    // extract border from shape and size of label volume
    unsigned int border = (shape2[0] - dim1size) / 2;

    // if a border needs to be applied the volume should be equal size in all dimensions
    // TODO: specify borders for each dimension
    if (border > 0) {
        if ((shape2[0] != shape2[1]) || (shape2[0] != shape2[2])) {
            throw ErrMsg("Dimensions of prediction should be equal in X, Y, Z");
        }
    }



    // for each channel, create volume data and push in array
    for (int i = 0; i < shape[0]; ++i) {
        VolumeData<T>* volumedata = new VolumeData<T>;
        vigra::TinyVector<vigra::MultiArrayIndex, 1> channel(i);
        (*volumedata) = (volumedata_temp.bindOuter(channel)).subarray(
                vigra::Shape3(border, border, border), vigra::Shape3(shape2[0]-border,
                    shape2[1]-border, shape2[2]-border)); 
        
        vol_array.push_back(boost::shared_ptr<VolumeData<T> >(volumedata));
    }

    return vol_array; 
}

template <typename T>
std::vector<boost::shared_ptr<VolumeData<T> > > 
    VolumeData<T>::create_volume_from_images(
        std::vector<std::string>& file_names)
{
    assert(!file_names.empty());
    vigra::ImageImportInfo info_init(file_names[0].c_str());
    
    if (!info_init.isGrayscale()) {
        throw ErrMsg("Cannot read non-grayscale image stack");
    }

    VolumeData<T>* volumedata = new VolumeData<T>;
    volumedata->reshape(vigra::MultiArrayShape<3>::type(
        info_init.width(),
        info_init.height(), 
        file_names.size()));

    for (int i = 0; i < file_names.size(); ++i) {
        
        vigra::ImageImportInfo info(file_names[i].c_str());
        vigra::BImage image(info.width(), info.height());
        vigra::importImage(info,destImage(image));
        
        for (int y = 0; y < int(info.height()); ++y) {
            for (int x = 0; x < int(info.width()); ++x) {
                (*volumedata)(x,y,i) = (float)image(x,y) / 255.0; 
            }
        }  
    }
    
    std::vector<boost::shared_ptr<VolumeData<T> > > vol_array;
    
    boost::shared_ptr<VolumeData<T> > volume_shared_ptr = boost::shared_ptr<VolumeData<T> >(volumedata);
    
    vol_array.push_back(volume_shared_ptr);
    vol_array.push_back(volume_shared_ptr);
    
    return vol_array;
}

template <typename T>
void VolumeData<T>::serialize(const char* h5_name, const char * h5_path)
{
    // x,y,z data will be written as z,y,x in the h5 file by default
#ifdef NP_OUTPUT_FILES
    std::cout << "Writing files" << std::endl;
    
    // vigra::MultiArray<3, vigra::RGBValue<vigra::UInt8> > rgb_vol((*this).shape());
    
    printf("this.shape[Z/Y/X] = [%d,%d,%d]\n", 
        (int)((*this)).shape(2),
        (int)((*this)).shape(1),
        (int)((*this)).shape(0));
    
    for (int z = 0; z < (int)((*this)).shape(2); ++z) {
        vigra::MultiArray<2, vigra::RGBValue<vigra::UInt8> > out_image(
            (int)((*this).shape(0)), 
            (int)((*this).shape(1)));
        
        for (int y = 0; y < (int)((*this)).shape(1); ++y) {
            for (int x = 0; x < (int)((*this)).shape(0); ++x) {
                uint32_t cur_val = (*this)(x,y,z);
                uint8_t byte_1 = (uint8_t)(cur_val & 0xFF);
                uint8_t byte_2 = (uint8_t)((cur_val >> 8) & 0xFF);
                uint8_t byte_3 = (uint8_t)((cur_val >> 16) & 0xFF);
                
                out_image(x,y)[2] = byte_1;
                out_image(x,y)[1] = byte_2;
                out_image(x,y)[0] = byte_3;
                
                // rgb_vol(x,y,z)[0] = byte_1;
                //                 rgb_vol(x,y,z)[1] = byte_2;
                //                 rgb_vol(x,y,z)[2] = byte_3;
                
            }
        }
        
        char out_path[2000];
        sprintf(out_path, "%s_%.4d.png", h5_name, z);
        vigra::exportImage(out_image, out_path);
    }
    
    // vigra::VolumeExportInfo info(h5_name, ".png");
    //     info.setPixelType("UINT8");
    //     vigra::exportVolume(rgb_vol, info);
    
    //
#else    
    vigra::writeHDF5(h5_name, h5_path, *this);
#endif
}

// convenience macro for iterating a multiarray and derived classes
#define volume_forXYZ(volume,x,y,z) \
    for (int z = 0; z < (int)(volume).shape(2); ++z) \
        for (int y = 0; y < (int)(volume).shape(1); ++y) \
            for (int x = 0; x < (int)(volume).shape(0); ++x) 


}

#endif
