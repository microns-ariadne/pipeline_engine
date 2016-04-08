#ifndef __WS_ALG__
#define __WS_ALG__

#include <queue>
#include "stdint.h"

#define IS_BG (1)
#define BG_VAL (255)
#define BG_MARKER (1000000)

void do_watershed(uint64_t depth, uint64_t rows, uint64_t cols, 
				  uint8_t *image, uint32_t *markers, 
				  uint64_t *index_Buffer, int nconnectivity=6);

#endif // __WS_ALG__
