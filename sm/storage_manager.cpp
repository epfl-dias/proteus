

#include "sm/storage_manager.hpp"
#include "cm/comm_manager.hpp"
#include "topology/topology.hpp" // proteus

namespace storage {

void StorageManager::init() {}
void StorageManager::shutdown() {

  // TODO: clear all the mappings, delete all shared-memory regions.
}

void StorageManager::snapshot() {

  /*

          Ask OLTP to stop.
          Ask OLTP for # of records for all columns, epoch #
          set. then fork.

          Parent: ask OLTP to resume it's shit.
          Parent: set snapshot/epoch_id to zero

          Child: communicate with proteus..

  */
}

bool StorageManager::remove_shm(const std::string &key) {

  int ret = shm_unlink(key.c_str());
  if (ret != 0) {
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    return false;
  }
  return true;
}

bool StorageManager::alloc_shm(const std::string &key, const size_t size_bytes,
                               const size_t unit_size) {
  // std::cout << "[MemoryManager::alloc_shm] key: "<< key << std::endl;
  // std::cout << "[MemoryManager::alloc_shm] size_bytes: "<< size_bytes <<
  // std::endl; std::cout << "[MemoryManager::alloc_shm] numa_memset_id: "<<
  // numa_memset_id << std::endl;

  assert(key.length() <= 255);

  int shm_fd = shm_open(key.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);
  if (shm_fd == -1) {
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    return false;
  }

  if (ftruncate(shm_fd, size_bytes) < 0) { //== -1){
    shm_unlink(key.c_str());
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    return false;
  }

  void *mem_addr =
      mmap(NULL, size_bytes, PROT_WRITE | PROT_READ, MAP_SHARED, shm_fd, 0);
  if (!mem_addr) {
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    return false;
  }

  close(shm_fd);

  // Insert to vector
  struct mem_file file;
  file.path = key;
  file.size = size_bytes;
  file.unit_size = unit_size;
  file.num_records = 0;
  file.snapshot_epoch = 0;

  this->mappings.insert(std::pair<std::string, struct mem_file>(key, file));

  return true;
}

} // namespace storage
