#include "common/gpu/gpu-common.hpp"
#include "common/common.hpp"
#include <cassert>
#include "multigpu/numa_utils.cuh"
#include "util/raw-memory-manager.hpp"

void launch_kernel(CUfunction function, void ** args, dim3 gridDim, dim3 blockDim, cudaStream_t strm){
    gpu_run(cuLaunchKernel(function, gridDim.x, gridDim.y, gridDim.z,
                                 blockDim.x, blockDim.y, blockDim.z,
                                 0, (CUstream) strm, args, NULL));
}

void launch_kernel(CUfunction function, void ** args, dim3 gridDim, cudaStream_t strm){
    launch_kernel(function, args, gridDim, defaultBlockDim, strm);
}

void launch_kernel(CUfunction function, void ** args, cudaStream_t strm){
    launch_kernel(function, args, defaultGridDim, defaultBlockDim, strm);
}

extern "C" {
void launch_kernel(CUfunction function, void ** args){
    launch_kernel(function, args, defaultGridDim, defaultBlockDim, 0);
}

void launch_kernel_strm(CUfunction function, void ** args, cudaStream_t strm){
    launch_kernel(function, args, strm);
    gpu_run(cudaStreamSynchronize(strm));
}

void launch_kernel_strm_sized(CUfunction function, void ** args, cudaStream_t strm, unsigned int blockX, unsigned int gridX){
    launch_kernel(function, args, blockX, gridX, strm);
}

void launch_kernel_strm_single(CUfunction function, void ** args, cudaStream_t strm){
    launch_kernel_strm_sized(function, args, strm, 1, 1);
}
}

std::ostream& operator<<(std::ostream& out, const cpu_set_t& cpus) {
    long cores = sysconf(_SC_NPROCESSORS_ONLN);

    bool printed = false;

    for (int i = 0 ; i < cores ; ++i) if (CPU_ISSET(i, &cpus)) {
        if (printed) out << ",";
        printed = true;
        out << i;
    }

    return out;
}

mmap_file::mmap_file(std::string name, data_loc loc): loc(loc){
    time_block t("Topen (" + name + "): ");

    filesize = ::getFileSize(name.c_str());
    fd       = open(name.c_str(), O_RDONLY, 0);

    if (fd == -1){
        string msg("[Storage: ] Failed to open input file " + name);
        LOG(ERROR) << msg;
        throw runtime_error(msg);
    }

    //Execute mmap
    data     = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    assert(data != MAP_FAILED);

    // gpu_run(cudaHostRegister(data, filesize, 0));
    if (loc == PINNED){
        void * data2 = RawMemoryManager::mallocPinned(filesize);
        // void * data2 = cudaMallocHost_local_to_cpu(filesize);

        memcpy(data2, data, filesize);
        munmap(data, filesize);
        close (fd  );
        data = data2;
    }

    gpu_data = data;

    if (loc == GPU_RESIDENT){
        std::cout << "Dataset on device: " << get_device() << std::endl;
        gpu_run(cudaMalloc(&gpu_data,       filesize));
        gpu_run(cudaMemcpy( gpu_data, data, filesize, cudaMemcpyDefault));
        munmap(data, filesize);
        close (fd  );
    }
}

mmap_file::mmap_file(std::string name, data_loc loc, size_t bytes, size_t offset = 0): loc(loc), filesize(bytes){
    time_block t("Topen (" + name + ", " + std::to_string(offset) + ":" + std::to_string(offset + filesize) + "): ");

    size_t real_filesize = ::getFileSize(name.c_str());
    assert(offset + filesize <= real_filesize);
    fd       = open(name.c_str(), O_RDONLY, 0);

    if (fd == -1){
        string msg("[Storage: ] Failed to open input file " + name);
        LOG(ERROR) << msg;
        throw runtime_error(msg);
    }

    //Execute mmap
   {
        time_block t("Tmmap: ");
        data     = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, offset);
        assert(data != MAP_FAILED);
    }

    // gpu_run(cudaHostRegister(data, filesize, 0));
    if (loc == PINNED){
        void * data2;
        {
            time_block t("Talloc: ");


            // {
            //     time_block t("Tmmap_alloc: ");
            //     data2 = mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_HUGETLB | MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
            //     assert(data2 != MAP_FAILED);
            //     numa_tonode_memory(data2, filesize, numa_node_of_cpu(sched_getcpu()));
            //     // void * mem = numa_alloc_onnode(size, device);
            //     // assert(mem);
            // }
            // gpu_run(cudaHostRegister(data2, filesize, 0));
            data2 = RawMemoryManager::mallocPinned(filesize);
            // gpu_run(cudaMallocHost(&data2, filesize));
        }
        memcpy(data2, data, filesize);
        munmap(data, filesize);
        close (fd  );
        data = data2;
    }

    gpu_data = data;

    if (loc == GPU_RESIDENT){
        std::cout << "Dataset on device: " << get_device() << std::endl;
        gpu_run(cudaMalloc(&gpu_data,       filesize                   ));
        gpu_run(cudaMemcpy( gpu_data, data, filesize, cudaMemcpyDefault));
        munmap(data, filesize);
        close (fd  );
    }
}
// mmap_file::mmap_file(std::string name, data_loc loc, size_t bytes, size_t offset = 0): loc(loc), filesize(bytes){
//     time_block t("Topen (" + name + ", " + std::to_string(offset) + ":" + std::to_string(offset + filesize) + "): ");

//     size_t real_filesize = ::getFileSize(name.c_str());
//     assert(offset + filesize <= real_filesize);
//     fd       = open(name.c_str(), O_RDONLY, 0);

//     if (fd == -1){
//         string msg("[Storage: ] Failed to open input file " + name);
//         LOG(ERROR) << msg;
//         throw runtime_error(msg);
//     }

//     // gpu_run(cudaHostRegister(data, filesize, 0));
//     if (loc == PINNED){
//         void * data2 = cudaMallocHost_local_to_cpu(filesize);
        
//         lseek(fd, offset, SEEK_SET);

//         char * data3 = (char *) data2;
//         void * fptr = ((char *) data2) + filesize;
//         size_t rem  = filesize;
//         while (data3 < fptr){
//             ssize_t rc = read(fd, data3, rem);
//             std::cout << SSIZE_MAX << " " << rc << " " << filesize << " " << rem << std::endl;
//             assert(rc > 0);
//             data3 += rc;
//             rem   -= rc;
//         }

//         // memcpy(data2, data, filesize);
//         // munmap(data, filesize);
//         close (fd  );
//         data = data2;
//     } else {
//         //Execute mmap
//         data     = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, offset);
//         assert(data != MAP_FAILED);
//     }

//     gpu_data = data;

//     if (loc == GPU_RESIDENT){
//         std::cout << "Dataset on device: " << get_device() << std::endl;
//         gpu_run(cudaMalloc(&gpu_data,       filesize                   ));
//         gpu_run(cudaMemcpy( gpu_data, data, filesize, cudaMemcpyDefault));
//         munmap(data, filesize);
//         close (fd  );
//     }
// }

mmap_file::~mmap_file(){
    if (loc == GPU_RESIDENT) gpu_run(cudaFree(gpu_data));

    // gpu_run(cudaHostUnregister(data));
    // if (loc == PINNED)       gpu_run(cudaFreeHost(data));
    if (loc == PINNED)       RawMemoryManager::freePinned(data);

    if (loc == PAGEABLE){
        munmap(data, filesize);
        close (fd  );
    }
}

const void * mmap_file::getData() const{
    return gpu_data;
}

size_t mmap_file::getFileSize() const{
    return filesize;
}


extern "C"{
    int get_ptr_device(const void *p){
        return get_device(p);
    }

    int get_ptr_device_or_rand_for_host(const void *p){
        int dev = get_device(p);
        if (dev >= 0) return dev;
        return rand();
    }

    int get_rand_core_local_to_ptr(const void *p){
        int dev = get_device(p);
        cpu_set_t cset;
        if (dev >= 0) {
            cset = gpu_affinity[dev];
        } else {
            int node;
            void * tmp = (void *) p;
            move_pages(0, 1, &tmp, NULL, &node, MPOL_MF_MOVE);
            cset = cpu_numa_affinity[node];
         }
        while (true) {
            int r = rand();
            if (CPU_ISSET(r % cpu_cnt, &cset)) return r;
        }
    }

    void memcpy_gpu(void *dst, const void *src, size_t size, bool is_volatile){
        assert(!is_volatile);
#ifndef NCUDA
        cudaStream_t strm;
        gpu_run(cudaStreamCreate     (&strm));
        gpu_run(cudaMemcpyAsync      (dst, src, size, cudaMemcpyDefault, strm));
        gpu_run(cudaStreamSynchronize( strm));
        gpu_run(cudaStreamDestroy    ( strm));
#else
        memcpy(dst, src, size);
#endif
    }
}

int get_device(const void *p){
#ifndef NCUDA
    cudaPointerAttributes attrs;
    cudaError_t error = cudaPointerGetAttributes(&attrs, p);
    if (error == cudaErrorInvalidValue) return -1;
    gpu_run(error);
    return (attrs.memoryType == cudaMemoryTypeHost) ? -1 : attrs.device;
#else
    return -1;
#endif
}
