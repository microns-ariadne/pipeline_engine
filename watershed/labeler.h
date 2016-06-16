#include "stdint.h"
#include "coordinates_conversion.h"
#include "connectivity.h"
#include <iostream>

using namespace std;

class ComponentLabeler
{
    private:
        uint64_t *buffer;
        uint64_t bufSize;
        uint64_t lastIndex;
        WatershedConnectivity *conn;
    
    public:
    /* Assumes the size of this buffer if >= max queue length
     * just pass the temporary buffer, as used later in watershed
     */
    ComponentLabeler(WatershedConnectivity *_conn, uint64_t _bufSize, uint64_t *_buffer) {
        bufSize = _bufSize;
        buffer = _buffer;
        lastIndex = 0;
        conn = _conn;
    }
    
    uint64_t labelComponent(
        uint32_t *markers, uint64_t depth, uint64_t rows, uint64_t cols, 
		uint64_t startZ, uint64_t startY, uint64_t startX, 
		uint32_t currentLabel) {
	    
	    lastIndex = 0;

	    uint64_t currentLabelCount = 1;
	    uint64_t curIndex = 0;
	    buffer[lastIndex++] = XYZ_TO_INDEX(startX, startY, startZ);
	
        while (curIndex < lastIndex) {
            // cout << "curIndex = " << curIndex << endl;
            uint64_t imgIndex = buffer[curIndex++];
            int64_t z = INDEX_TO_Z(imgIndex);
            int64_t y = INDEX_TO_Y(imgIndex);
            int64_t x = INDEX_TO_X(imgIndex);

            for (int i = 0; i < conn->nconnectivity; i++) {
                int64_t nextz = z + conn->dz[i];
                int64_t nexty = y + conn->dy[i];
                int64_t nextx = x + conn->dx[i];
                int64_t nextIndex = XYZ_TO_INDEX(nextx, nexty, nextz);
                if (nextz < 0 || 
                    nextz >= depth || 
                    nexty < 0 || 
                    nextx < 0 || 
                    nexty >= rows || 
                    nextx >= cols) {
                    continue;
                }

                if (markers[nextIndex] != 1) {
                    continue;
                }

                markers[nextIndex] = currentLabel;
                buffer[lastIndex++] = nextIndex;
                currentLabelCount++;
            }
        }

        return currentLabelCount;
    }
    
    uint64_t unlabelComponent(uint32_t *markers) {
	    for (uint64_t index = 0; index < lastIndex; index++) {
	        markers[buffer[index]] = 0;
        }
    }
      
};
