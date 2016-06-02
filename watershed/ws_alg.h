#ifndef __WS_ALG__
#define __WS_ALG__

#include <queue>
#include "stdint.h"
#include "connectivity.h"
#include "ws_config.h"

void do_watershed(uint64_t depth, uint64_t rows, uint64_t cols, 
				  uint8_t *image, uint32_t *markers, 
				  uint64_t *index_Buffer, WatershedConnectivity *conn);

#endif // __WS_ALG__
