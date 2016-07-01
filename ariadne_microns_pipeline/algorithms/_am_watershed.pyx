import numpy as np
cimport numpy as np

from libc.stdint cimport uint8_t, uint64_t, uint32_t
cdef extern from "connectivity.h":
    cdef cppclass WatershedConnectivity:
      WatershedConnectivity(int _nconnectivity) except +

cdef extern from "ws_alg.h" nogil:
    cdef void do_watershed(uint64_t depth, uint64_t rows, uint64_t cols, 
                           uint8_t *image, uint32_t *markers, 
                           uint64_t *index_Buffer, WatershedConnectivity *conn)

def watershed(np.ndarray[dtype=np.uint8_t, mode='c', ndim=3] image,
              np.ndarray[dtype=np.uint32_t, mode='c', ndim=3] markers,
              int connectivity=6):
    '''Perform a watershed with the given markers on the given image
    
    :param image: image to be watershedded (low values are filled first)
    :param markers: on input, markers of the seeds for the watershed, on output,
         the segmented volume.
    :param connectivity: 6 for 3-d six-connectivity, 4 for 2-d four-
        connectivity.
    '''
    cdef:
        np.ndarray[dtype=np.uint64_t, mode='c', ndim=3] index_buffer =\
            np.zeros((image.shape[0], image.shape[1], image.shape[2]),
                     np.uint64)
        WatershedConnectivity *c=new WatershedConnectivity(connectivity)
    assert image.shape[0] == markers.shape[0]
    assert image.shape[1] == markers.shape[1]
    assert image.shape[2] == markers.shape[2]
    
    with nogil:
        do_watershed(image.shape[0],
                     image.shape[1],
                     image.shape[2],
                     <uint8_t  *>image.data,
                     <uint32_t *>markers.data,
                     <uint64_t *>index_buffer.data,
                     c)
        
        
    