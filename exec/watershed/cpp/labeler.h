#include "stdint.h"
#include "coordinates_conversion.h"
#include <iostream>

#define IS_3D_SEEDS

using namespace std;

class ComponentLabeler
{
 public:
  /* Assumes the size of this buffer if >= max queue length
   * just pass the temporary buffer, as used later in watershed
   */
  ComponentLabeler(uint64_t _bufSize, uint64_t *_buffer)
	{
	  bufSize = _bufSize;
	  buffer = _buffer;
	  lastIndex = 0;
	}

  uint64_t labelComponent(uint32_t *markers, uint64_t depth, uint64_t rows, uint64_t cols, 
						  uint64_t startZ, uint64_t startY, uint64_t startX, 
						  uint32_t currentLabel)
  {
	// cout << "here0" << endl;
	lastIndex = 0;
#ifdef IS_3D_SEEDS
	int connectivity = 6;
	int dy[] = {0, -1, 0, 0, 1, 0};
	int dx[] = {0, 0, -1, 1, 0, 0};
	int dz[] = {-1, 0, 0, 0, 0, 1};
#else
	int connectivity = 4;
    int dy[] = {0, -1, 1, 0};
    int dx[] = {-1, 0, 0, 1};
    int dz[] = {0, 0, 0, 0};
#endif
	// if (depth == 1) dz[4] = dz[5] = 0;

	uint64_t currentLabelCount = 1;
	uint64_t curIndex = 0;
	// cout << "here, alstIndex = " << lastIndex << endl;
	// cout << "startX " << startX << ",  " << startY << ", " << startZ << endl;
	buffer[lastIndex++] = XYZ_TO_INDEX(startX, startY, startZ);
	// cout << buffer << endl;
	// buffer[lastIndex++] = startZ * rows*cols + startY*cols + startX;
	while (curIndex < lastIndex)
	  {
		// cout << "curIndex = " << curIndex << endl;
		uint64_t imgIndex = buffer[curIndex++];
		int64_t z = INDEX_TO_Z(imgIndex);
		int64_t y = INDEX_TO_Y(imgIndex);
		int64_t x = INDEX_TO_X(imgIndex);

		for (int i = 0; i < connectivity; i++)
		  {
			int64_t nextz = z + dz[i];
			int64_t nexty = y + dy[i];
			int64_t nextx = x + dx[i];
			int64_t nextIndex = XYZ_TO_INDEX(nextx, nexty, nextz);
			if (nextz < 0 || nextz >= depth || nexty < 0 || nextx < 0 || 
				nexty >= rows || nextx >= cols)
			  continue;
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
  

  uint64_t unlabelComponent(uint32_t *markers)
  {
	for (uint64_t index; index < lastIndex; index++)
	  markers[buffer[index]] = 0;
  }

 private:
  uint64_t *buffer;
  uint64_t bufSize;
  uint64_t lastIndex;
};
