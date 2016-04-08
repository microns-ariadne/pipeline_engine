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

#include <vigra/multi_array.hxx>

// used for importing h5 files
#include <vigra/hdf5impex.hxx>
#include <vigra/impex.hxx>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <string>
#include "../Utilities/ErrMsg.h"

namespace NeuroProof {

// forward declaration
template <typename T>
class VolumeData;

// defines some of the common volume types used in NeuroProof
typedef VolumeData<double> VolumeProb;
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
    static boost::shared_ptr<VolumeData<T> > create_volume_from_images(
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

template <typename T>
std::vector<boost::shared_ptr<VolumeData<T> > > 
    VolumeData<T>::create_volume_array_NEW(const char * h5_name, const char * dset)
{
    printf("create_volume_array_NEW - 1\n");
    vigra::HDF5ImportInfo info(h5_name, dset);
    vigra_precondition(info.numDimensions() == 3, "Dataset must be 3-dimensional.");

    printf("create_volume_array_NEW - 2\n");
    vigra::TinyVector<long long unsigned int, 3> shape(info.shape().begin());
    vigra::MultiArray<3, T> volumedata_temp(shape);
    vigra::readHDF5(info, volumedata_temp);
    
    printf("create_volume_array_NEW - 3\n");
    // since the X,Y,Z,ch is read in as ch,Z,Y,X transpose
    volumedata_temp = volumedata_temp.transpose();
    
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
        VolumeData<T>* volumedata_1 = new VolumeData<T>;
        VolumeData<T>* volumedata_2 = new VolumeData<T>;
        
        //vigra::TinyVector<vigra::MultiArrayIndex, 1> channel(i);
        //(*volumedata) = volumedata_temp.bindOuter(channel); 
        
        (*volumedata_1) = volumedata_temp;
        (*volumedata_2) = volumedata_temp;
                
        vol_array.push_back(boost::shared_ptr<VolumeData<T> >(volumedata_1));
        vol_array.push_back(boost::shared_ptr<VolumeData<T> >(volumedata_2));
        
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
boost::shared_ptr<VolumeData<T> > VolumeData<T>::create_volume_from_images(
        std::vector<std::string>& file_names)
{
    assert(!file_names.empty());
    vigra::ImageImportInfo info_init(file_names[0].c_str());
    
    if (!info_init.isGrayscale()) {
        throw ErrMsg("Cannot read non-grayscale image stack");
    }

    VolumeData<unsigned char>* volumedata = new VolumeData<T>;
    volumedata->reshape(vigra::MultiArrayShape<3>::type(info_init.width(),
                info_init.height(), file_names.size()));

    for (int i = 0; i < file_names.size(); ++i) {
        vigra::ImageImportInfo info(file_names[i].c_str());
        vigra::BImage image(info.width(), info.height());
        vigra::importImage(info,destImage(image));
        for (int y = 0; y < int(info.height()); ++y) {
            for (int x = 0; x < int(info.width()); ++x) {
                (*volumedata)(x,y,i) = image(x,y); 
            }
        }  
    }

    return boost::shared_ptr<VolumeData<T> >(volumedata);
}



template <typename T>
void VolumeData<T>::serialize(const char* h5_name, const char * h5_path)
{
    // x,y,z data will be written as z,y,x in the h5 file by default
    vigra::writeHDF5(h5_name, h5_path, *this);
}

// convenience macro for iterating a multiarray and derived classes
#define volume_forXYZ(volume,x,y,z) \
    for (int z = 0; z < (int)(volume).shape(2); ++z) \
        for (int y = 0; y < (int)(volume).shape(1); ++y) \
            for (int x = 0; x < (int)(volume).shape(0); ++x) 


}

#endif
