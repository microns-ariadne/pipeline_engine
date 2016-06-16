#ifndef __WS_QUEUE__
#define __WS_QUEUE__

#include <iostream>
#include "stdint.h"

/* Fast monotonic queue for keys 0 .. 255
 * returns (uint64_t)-1 on dequeue if empty
 */
class WsQueue
{
 public:
  WsQueue(uint64_t imgSize, uint8_t *img, uint64_t *_indexBuffer);
  void enqueue(uint64_t index, uint8_t value);
  // returns -1 on empty
  uint64_t dequeue();
  uint64_t size();
 private:
  void binCount(uint64_t imgSize, uint8_t *img);
  int minval = 0;
  uint64_t counts[256];
  // uint64_t *outstanding[256];
  uint64_t *indexBuffer;
  uint64_t start[256], end[256];
};

#endif //__WS_QUEUE__
