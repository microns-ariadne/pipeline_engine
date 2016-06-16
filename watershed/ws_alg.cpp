#include "ws_alg.h"
#include "ws_queue.h"
#include "coordinates_conversion.h"
#include <iostream>

using namespace std;

void do_watershed(
    uint64_t depth, uint64_t rows, uint64_t cols, 
	uint8_t *image, uint32_t *markers, uint64_t *indexBuffer, 
	WatershedConnectivity *conn) {
    
    int maxIndex = depth*rows*cols;
    WsQueue q(depth * rows * cols, image, indexBuffer);

    int64_t age = 1;
    for (int64_t x = 0; x < cols; x++) {
        for (int64_t y = 0; y < rows; y++) {
            for (int64_t z = 0; z < depth; z++) {
		        
		        int64_t index = XYZ_TO_INDEX(x,y,z);
                
		        if (markers[index] > 0) {
			        // check if at least 1 neighbor is not marked
			        // this elimiates dumping inner nodes on the queue
			        bool emptyNeighbor = false;
			        for (int c = 0; c < conn->nconnectivity && !emptyNeighbor; c++) {
			            int64_t nz = z + conn->dz[c];
			            int64_t ny = y + conn->dy[c];
			            int64_t nx = x + conn->dx[c];
                        if (nz < 0 || 
                            nz >= depth ||
                            nx < 0 || 
                            nx >= cols ||
                            ny >= rows || ny < 0) {
                            continue;
                            
                        } else {
                            int64_t nind = XYZ_TO_INDEX(nx,ny,nz);
                            // anything different than this one
                            if (markers[nind] != markers[index]) {
                                emptyNeighbor = true;
                                break;
                            }
                        }
                    }
                    
                    if (emptyNeighbor) {
    				    // cout << "enqueuing z, y, x: " << z+1 << " " << y+1 << " " << x+1 << endl;
    				    q.enqueue(index, image[index]);
    				    age++;
    				}
                }
			}
        }
    }

    uint64_t index = -1;
    // cout << "********* WATERSHED PROCESSING: " << endl;
    while ((index = q.dequeue()) != -1) {
        int64_t z = INDEX_TO_Z(index);
        int64_t y = INDEX_TO_Y(index);
        int64_t x = INDEX_TO_X(index);
        // cout << "z,y,x:  " << z+1 << " " << y+1 << " " << x+1 << "  -> " << (uint32_t)image[index] << " " << markers[index] << endl;

        for (int i = 0; i < conn->nconnectivity; i++) {
            int64_t nextz = z + conn->dz[i];
            int64_t nexty = y + conn->dy[i];
            int64_t nextx = x + conn->dx[i];

            if ((nextz < 0 || nextz >= depth) || 
                (nextx < 0 || nextx >= cols) ||  
                (nexty >= rows || nexty < 0)) {
                continue;
            }
            
		    // cout << "next xyz: " << nextx << " " << nexty << " " << nextz << endl;
		    int64_t nextIndex = XYZ_TO_INDEX(nextx, nexty, nextz);
		    // cout << "\t ni: " << nextIndex << endl;
		  
		    if (markers[nextIndex] > 0) {
			    continue;
		    }
            
            markers[nextIndex] = markers[index];
		    q.enqueue(nextIndex, image[nextIndex]);
		    age++;
		}
	}
	
}
