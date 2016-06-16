#include <cstring>
#include "ws_queue.h"

WsQueue::WsQueue(uint64_t imgSize, uint8_t *img, uint64_t *_indexBuffer)
{
  memset(start, 0, sizeof(start));
  memset(end, 0, sizeof(end));
  indexBuffer = _indexBuffer;
  minval = 0;
  binCount(imgSize, img);
  /*for (int i = 0; i < 256; i++)
	outstanding[i] = new uint64_t[counts[i]];*/
  for (int i = 1; i < 256; i++)
	end[i] = start[i] = start[i-1] + counts[i-1];
}

void WsQueue::enqueue(uint64_t index, uint8_t value)
{
  if (value < minval)
	minval = value;
  // outstanding[value][end[value]] = index;
  indexBuffer[end[value]] = index;
  end[value]++;
}

uint64_t WsQueue::dequeue()
{
  for (; minval < 256; minval++)
	if (start[minval] < end[minval]) {
	  // uint64_t ret = outstanding[minval][start[minval]];
	  uint64_t ret = indexBuffer[start[minval]];
	  start[minval]++;
	  return ret;
	}
  return (uint64_t)-1;
}

uint64_t WsQueue::size()
{
  uint64_t ret = 0;
  // std::cout << "in size: " << std::endl;
  for (int i = 0; i < 256; i++)
	{
	  // std::cout << "adding: " << end[i] << " " << start[i] << std::endl;
	  ret += (end[i] - start[i]);
	}
  // std::cout << "ret = " << ret << std::endl;
  return ret;
}

void WsQueue::binCount(uint64_t imgSize, uint8_t *img)
{
  memset(counts, 0, sizeof(counts));
  for (uint64_t index = 0; index < imgSize; index++)
	counts[img[index]]++;
}
