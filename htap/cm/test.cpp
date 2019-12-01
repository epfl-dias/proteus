

#include <numa.h>
#include <numaif.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <vector>

int main() {
  pid_t pid_oltp = fork();

  if (pid_oltp == 0) {
    execl("/scratch/raza/htap/opt/aeolus/aeolus-server", "");
    assert(false && "OLTP process call failed");

  } else {
    return pid_oltp;
  }

  return 0;
}

void *map(std::string key, size_t size_bytes) {
  int shm_fd = shm_open(key.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);
  if (shm_fd == -1) {
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    assert(false);
  }

  if (ftruncate(shm_fd, size_bytes) < 0) {  //== -1){
    shm_unlink(key.c_str());
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    assert(false);
  }

  void *mem_addr =
      mmap(NULL, size_bytes, PROT_WRITE | PROT_READ, MAP_SHARED, shm_fd, 0);
  if (!mem_addr) {
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    assert(false);
  }

  close(shm_fd);

  return mem_addr;
}