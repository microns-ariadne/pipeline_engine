#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdint.h>
#include <algorithm>

#include "Python.h"
#include "numpy/arrayobject.h"

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <string>
#include <vector>

namespace fs = boost::filesystem;

#define TRANSPOSE false

bool read_init(PyObject *args, std::vector< std::string > &images, int &depth, int &rows, int &cols) {
    const char* s = 0;
    if (!PyArg_ParseTuple(args, "s", &s)) {
        return false;
    }

    std::string blockDir(s);

    if (!fs::is_directory(blockDir)) {
        printf("ERROR reading %s - does not exist or not a directory\n", blockDir.c_str());
        return false;
    }

    for (fs::directory_iterator it(blockDir); it != fs::directory_iterator(); it++) {
        images.push_back(it->path().string());
    }

    std::sort(images.begin(), images.end());

    depth = images.size();
    if (depth == 0) {
        printf("ERROR reading %s - nothing to read\n", blockDir.c_str());
        return false;
    }

    cv::Mat image0 = cv::imread(images[0].c_str(), CV_LOAD_IMAGE_UNCHANGED);
    rows = image0.rows;
    cols = image0.cols;

    if (rows == 0 || cols == 0) {
        printf("ERROR reading %s - not an image or empty\n", images[0].c_str());
        return false;
    }

    return true;
}

static PyObject* read_probabilities_int(PyObject *dummy, PyObject *args) {
    std::vector< std::string > images;
    int depth, rows, cols;
    if (!read_init(args, images, depth, rows, cols)) {
        return Py_None;
    }

    uint8_t* data = new uint8_t[depth * rows * cols];
    std::vector< bool > ok(depth, true);
    cilk_for (int d = 0; d < depth; ++d) {
        cv::Mat image = cv::imread(images[d].c_str(), CV_LOAD_IMAGE_UNCHANGED);
        if (image.rows != rows || image.cols != cols) {
            ok[d] = false;
            printf("ERROR reading %s - images sizes don't match\n", images[d].c_str());
        } else {
            memcpy(((uint8_t*)data) + d * rows * cols, image.data, rows * cols);
        }
    }

    if (std::count(ok.begin(), ok.end(), false) > 0) {
        return Py_None;
    }
    npy_intp dims[3] = {depth, rows, cols};
    PyObject* res = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, data);
    Py_INCREF(res);
    return res;
}

static PyObject* read_probabilities_float(PyObject *dummy, PyObject *args) {
    std::vector< std::string > images;
    int depth, rows, cols;
    if (!read_init(args, images, depth, rows, cols)) {
        return Py_None;
    }

    float* data = new float[depth * rows * cols];
    std::vector< bool > ok(depth, true);
    cilk_for (int d = 0; d < depth; ++d) {
        cv::Mat image = cv::imread(images[d].c_str(), CV_LOAD_IMAGE_UNCHANGED);
        if (image.rows != rows || image.cols != cols) {
            ok[d] = false;
            printf("ERROR reading %s - images sizes don't match\n", images[d].c_str());
        } else {
            for (int p = 0; p < rows * cols; ++p) {
                data[d * rows * cols + p] = image.data[p] / 255.0;
            }
        }
    }
    if (std::count(ok.begin(), ok.end(), false) > 0) {
        return Py_None;
    }

    npy_intp dims[3] = {depth, rows, cols};
    PyObject* res = PyArray_SimpleNewFromData(3, dims, NPY_FLOAT32, data);
    Py_INCREF(res);
    return res;
}

static PyObject* read_int16_labels(PyObject* dummy, PyObject *args) {
    std::vector< std::string > images;
    int depth, rows, cols;
    if (!read_init(args, images, depth, rows, cols)) {
        return Py_None;
    }

    uint16_t* data = new uint16_t[depth * rows * cols];
    std::vector< bool > ok(depth, true);
    cilk_for (int d = 0; d < depth; ++d) {
        cv::Mat image = cv::imread(images[d].c_str(), CV_LOAD_IMAGE_UNCHANGED);
        if (image.rows != rows || image.cols != cols) {
            ok[d] = false;
            printf("ERROR reading %s - images sizes don't match\n", images[d].c_str());
        } else {
            memcpy(data + d * rows * cols, image.data, rows * cols * 2);
        }
    }

    if (std::count(ok.begin(), ok.end(), false) > 0) {
        return Py_None;
    }

    npy_intp dims[3] = {depth, rows, cols};
    PyObject* res = PyArray_SimpleNewFromData(3, dims, NPY_UINT16, data);

    if (TRANSPOSE) {
        res = PyArray_Transpose((PyArrayObject*)res, NULL);
    }
    Py_INCREF(res);
    return res;
}

static PyObject* read_rgb_labels(PyObject* dummy, PyObject *args) {
    std::vector< std::string > images;
    int depth, rows, cols;
    if (!read_init(args, images, depth, rows, cols)) {
        return Py_None;
    }

    uint32_t* data = new uint32_t[depth * rows * cols];
    std::vector< bool > ok(depth, true);
    cilk_for (int d = 0; d < depth; ++d) {
        cv::Mat image = cv::imread(images[d].c_str(), CV_LOAD_IMAGE_UNCHANGED);
        if (image.rows != rows || image.cols != cols) {
            ok[d] = false;
            printf("ERROR reading %s - images sizes don't match\n", images[d].c_str());
        } else {
            for (int p = 0; p < rows * cols; ++p) {
                uint32_t blu = image.data[3 * p + 0];
                uint32_t grn = image.data[3 * p + 1];
                uint32_t red = image.data[3 * p + 2];
                // opencv has bgr, we want rgb
                data[d * rows * cols + p] = ((red << 16) | (grn << 8) | blu); // keep opencv order
            }
        }
    }

    if (std::count(ok.begin(), ok.end(), false) > 0) {
        return Py_None;
    }

    npy_intp dims[3] = {depth, rows, cols};
    PyObject* res = PyArray_SimpleNewFromData(3, dims, NPY_UINT32, data);

    if (TRANSPOSE) {
        res = PyArray_Transpose((PyArrayObject*)res, NULL);
    }
    Py_INCREF(res);
    return res;
}

PyArrayObject* write_init(PyObject* args, std::string &path, std::string &prefix, int &depth, int &rows, int &cols) {
    const char* path_s = 0;
    const char* pref_s = 0;
    PyObject* obj = NULL;
    if (!PyArg_ParseTuple(args, "Os|s", &obj, &path_s, &pref_s)) {
        return NULL;
    }

    PyArrayObject* arr = (PyArrayObject*) obj;

    path = std::string(path_s);
    if (fs::exists(path)) {
        if (!fs::remove_all(path)) {
            printf("Cannot remove %s\n", path.c_str());
            return NULL;
        }
    }
    if (!fs::create_directory(path)) {
        printf("Cannot create %s\n", path.c_str());
        return NULL;
    }

    if (pref_s != 0) {
        prefix = std::string(pref_s) + "_";
    } else {
        prefix = "";
    }

    depth = PyArray_DIM(arr, 0);
    rows = PyArray_DIM(arr, 1);
    cols = PyArray_DIM(arr, 2);

    return PyArray_GETCONTIGUOUS(arr);
}

static PyObject* write_int_probabilities(PyObject *dummy, PyObject *args) {
    std::string dirPath, prefix;
    int depth, rows, cols;
    PyArrayObject* arr = write_init(args, dirPath, prefix, depth, rows, cols);
    if (arr == NULL) {
        return Py_None;
    }
    uint8_t* data = (uint8_t*) PyArray_DATA(arr);

    cilk_for (int d = 0; d < depth; ++d) {
        cv::Mat image(rows, cols, CV_8U);
        memcpy(image.data, data + d * rows * cols, rows * cols);

        boost::format fmt("%s/%s%04d.png");
        cv::imwrite(boost::str(fmt % dirPath % prefix % d), image);
    }
    return Py_None;
}

static PyObject* write_float_probabilities(PyObject *dummy, PyObject *args) {
    std::string dirPath, prefix;
    int depth, rows, cols;
    PyArrayObject* arr = write_init(args, dirPath, prefix, depth, rows, cols);
    if (arr == NULL) {
        return Py_None;
    }
    float* data = (float*) PyArray_DATA(arr);

    cilk_for (int d = 0; d < depth; ++d) {
        cv::Mat image_f(rows, cols, CV_32F, data + d * rows * cols);
        cv::Mat image;
        image_f.convertTo(image, CV_8U, 255);

        boost::format fmt("%s/%s%04d.png");
        cv::imwrite(boost::str(fmt % dirPath % prefix % d), image);
    }
    return Py_None;
}

PyArrayObject* getTranspose(PyArrayObject* arr) {
    arr = (PyArrayObject*)PyArray_Transpose(arr, NULL);
    PyArrayObject* res = PyArray_GETCONTIGUOUS(arr);
    arr = (PyArrayObject*)PyArray_Transpose(arr, NULL);
    return res;
}

static PyObject* write_labels_rgb(PyObject* dummy, PyObject *args) {
    std::string dirPath, prefix;
    int depth, rows, cols;
    PyArrayObject* arr = write_init(args, dirPath, prefix, depth, rows, cols);
    if (arr == NULL) {
        return Py_None;
    }

    if (TRANSPOSE) {
        arr = getTranspose(arr);
        std::swap(depth, cols);
    }
    uint32_t* data = (uint32_t*) PyArray_DATA(arr);

    cilk_for (int d = 0; d < depth; ++d) {
        cv::Mat image(rows, cols, CV_8UC3);
        for (int p = 0; p < rows * cols; ++p) {
            uint32_t id = data[d * rows * cols + p];
            // opencv has bgr, we want rgb
            image.data[3 * p + 0] = ((uint8_t)id & 255);            // cv::blue
            image.data[3 * p + 1] = ((uint8_t)(id >> 8) & 255);     // cv::green
            image.data[3 * p + 2] = ((uint8_t)(id >> 16) & 255);    // cv::red
        }

        boost::format fmt("%s/%s%04d.png");
        cv::imwrite(boost::str(fmt % dirPath % prefix % d), image);
    }
    return Py_None;
}

static PyObject* write_labels_int16(PyObject* dummy, PyObject *args) {
    std::string dirPath, prefix;
    int depth, rows, cols;
    PyArrayObject* arr = write_init(args, dirPath, prefix, depth, rows, cols);
    if (arr == NULL) {
        return Py_None;
    }

    if (TRANSPOSE) {
        arr = getTranspose(arr);
        std::swap(depth, cols);
    }
    uint16_t* data = (uint16_t*) PyArray_DATA(arr);

    cilk_for (int d = 0; d < depth; ++d) {
        cv::Mat image(rows, cols, CV_16U);
        memcpy(image.data, data + d * rows * cols, 2 * rows * cols);

        boost::format fmt("%s/%s%04d.png");
        cv::imwrite(boost::str(fmt % dirPath % prefix % d), image);
    }
    return Py_None;
}

static struct PyMethodDef methods[] = { 
        {"read_probabilities_int", read_probabilities_int, METH_VARARGS, "Reads probabilities from the given directory and returns in uint8 format"},
        {"read_probabilities_float", read_probabilities_float, METH_VARARGS, "Reads probabilities from the given directory and returns in float32 format"},
        {"read_rgb_labels", read_rgb_labels, METH_VARARGS, "Reads labels from the given directory. Convert from rgb to uint32."},
        {"read_int16_labels", read_int16_labels, METH_VARARGS, "Reads labels from the given directory (no conversion)"},
        {"write_int_probabilities", write_int_probabilities, METH_VARARGS, "Writes probabilities from the given int8 numpy 3d array to png's"},
        {"write_float_probabilities", write_float_probabilities, METH_VARARGS, "Writes probabilities from the given float numpy 3d array to png's"},
        {"write_labels_rgb", write_labels_rgb, METH_VARARGS, "Writes uint32 labels to rgb"},
        {"write_labels_int16", write_labels_int16, METH_VARARGS, "Writes uint16 labels to in one channel"},
        {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcilk_rw (void) {
    (void)Py_InitModule("cilk_rw", methods);
    import_array();
}

