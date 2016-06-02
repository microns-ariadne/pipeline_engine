#ifndef __CONNECTIVITY_H__
#define __CONNECTIVITY_H__

#include <assert.h>

const int _global_dy[] = {0, -1, 0, 0, 1, 0};
const int _global_dx[] = {0, 0, -1, 1, 0, 0};
const int _global_dz[] = {-1, 0, 0, 0, 0, 1};

class WatershedConnectivity
{
 public:
  WatershedConnectivity(int _nconnectivity = 6)
	{
	  nconnectivity = _nconnectivity;
	  assert(nconnectivity == 6 || nconnectivity == 4);
	  dy = new int[nconnectivity];
	  dx = new int[nconnectivity];
	  dz = new int[nconnectivity];
	  
	  int starti = nconnectivity == 4 ? 1 : 0;
	  for (int i = starti; i < starti + nconnectivity; i++)
		{
		  dy[i-starti] = _global_dy[i];
		  dx[i-starti] = _global_dx[i];
		  dz[i-starti] = _global_dz[i];
		}
	}
  ~WatershedConnectivity()
	{
	  delete[] dy, dx, dz;
	}
  int nconnectivity;
  int *dy, *dx, *dz;
};

#endif //__CONNECTIVITY_H__
