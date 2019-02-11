#ifndef MEMORY_MANAGER_HPP_
#define MEMORY_MANAGER_HPP_

#include <map>
#include <vector>


namespace storage {


struct mem_chunk {
  const void *data;
  size_t size;

  // latching or locking here?
};

}


#endif /* MEMORY_MANAGER_HPP_ */
