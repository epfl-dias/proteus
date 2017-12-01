#include "common/gpu/gpu-common.hpp"
#include "common/common.hpp"
#include <cassert>
#include "multigpu/numa_utils.cuh"

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
}

void launch_kernel_strm_single(CUfunction function, void ** args, cudaStream_t strm){
    launch_kernel(function, args, 1, 1, strm);
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
        void * data2 = cudaMallocHost_local_to_cpu(filesize);

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
    data     = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, offset);
    assert(data != MAP_FAILED);

    // gpu_run(cudaHostRegister(data, filesize, 0));
    if (loc == PINNED){
        void * data2 = cudaMallocHost_local_to_cpu(filesize);

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

mmap_file::~mmap_file(){
    if (loc == GPU_RESIDENT) gpu_run(cudaFree(gpu_data));

    // gpu_run(cudaHostUnregister(data));
    if (loc == PINNED)       cudaFreeHost_local_to_cpu(data, filesize);

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
        if (dev >= 0) return dev;
        int node;
        void * tmp = (void *) p;
        move_pages(0, 1, &tmp, NULL, &node, MPOL_MF_MOVE);
        cpu_set_t cset = cpu_numa_affinity[node];
        while (true) {
            int r = rand();
            if (CPU_ISSET(r % cpu_cnt, &cset)) return r;
        }
    }
}
