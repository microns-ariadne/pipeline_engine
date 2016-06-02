#ifndef __COORDINATES_CONVERSION__
#define __COORDINATES_CONVERSION__

#define XYZ_TO_INDEX(X, Y, Z) (Z*(rows*cols) + Y*cols + X)
#define INDEX_TO_Z(INDEX) (INDEX/(rows*cols))
#define INDEX_TO_Y(INDEX) ((INDEX % (rows*cols))/cols)
#define INDEX_TO_X(INDEX) (INDEX%cols)

#endif //__COORDINATES_CONVERSION__
