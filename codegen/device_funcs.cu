#include <cstdio>
#include <cinttypes>


extern "C"{

__device__ void printi64(int64_t x){
    printf("%" PRId64 "\n", x);
}

}