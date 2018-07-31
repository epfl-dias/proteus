// #include "buffer_manager.cuh"
#include <cinttypes>
#include <cstdint>
#include <cstdio>

// #include <cinttypes>

// __device__ __constant__ threadsafe_device_stack<int32_t *, (int32_t *) NULL> * pool;
// __device__ __constant__ int deviceId;
// __device__ __constant__ void * buff_start;
// __device__ __constant__ void * buff_end  ;

#include "common/gpu/gpu-common.hpp"

#include "multigpu/buffer_manager.cuh"
#include "topology/affinity_manager.hpp"
#include "util/raw-memory-manager.hpp"

#include <thread>
// #include <utmpx.h>
// #include <unistd.h>


template<typename T, typename... Args>
__host__ T * cuda_new(int dev, Args... args){
    if (dev >= 0){
        set_device_on_scope d(dev);
        T *tmp = new T(args...);
        T *res;
        gpu_run(cudaMalloc((void**) &res, sizeof(T)));
        gpu_run(cudaMemcpy(res, tmp, sizeof(T), cudaMemcpyDefault));
        gpu_run(cudaDeviceSynchronize());
        free(tmp);  //NOTE: bad practice ? we want to allocate tmp by new to
                    //      trigger initialization but we want to free the 
                    //      corresponding memory after moving to device 
                    //      without triggering the destructor
        return res;
    } else {
        T *tmp = new T(args...);
        T *res;
        gpu_run(cudaMallocHost((void**) &res, sizeof(T)));
        gpu_run(cudaMemcpy(res, tmp, sizeof(T), cudaMemcpyDefault));
        gpu_run(cudaDeviceSynchronize());
        free(tmp);  //NOTE: bad practice ? we want to allocate tmp by new to
                    //      trigger initialization but we want to free the 
                    //      corresponding memory after moving to device 
                    //      without triggering the destructor
        return res;
        // return new T(args...);
    }
}


template<typename T, typename... Args>
__host__ void cuda_delete(T *obj, Args... args){
    if (topology::getInstance().getGpuAddressed(obj)){
        T *tmp = (T *) malloc(sizeof(T));
        gpu_run(cudaDeviceSynchronize());
        gpu_run(cudaMemcpy(tmp, obj, sizeof(T), cudaMemcpyDefault));
        gpu_run(cudaFree(obj));
        delete tmp;
    } else {
        T *tmp = (T *) malloc(sizeof(T));
        gpu_run(cudaDeviceSynchronize());
        gpu_run(cudaMemcpy(tmp, obj, sizeof(T), cudaMemcpyDefault));
        gpu_run(cudaFreeHost(obj));
        delete tmp;
        // delete obj;
    }
}





extern "C"{

#ifndef NCUDA
__device__ void dprinti64(int64_t x){
    printf("%" PRId64 "\n", x);
}

__device__ int32_t * get_buffers(){
    uint32_t b = __ballot(1);
    uint32_t m = 1 << get_laneid();
    int32_t * ret;
    do {
        uint32_t leader = b & -b;

        if (leader == m) ret = buffer_manager<int32_t>::get_buffer();

        b ^= leader;
    } while (b);
    return ret;
}

void * get_dev_buffer(){
    return buffer_manager<int32_t>::h_get_buffer(topology::getInstance().getActiveGpu().id);
}

__device__ void release_buffers(int32_t * buff){
    uint32_t b = __ballot(buff != NULL);
    uint32_t m = 1 << get_laneid();
    do {
        uint32_t leader = b & -b;

        if (leader == m) buffer_manager<int32_t>::release_buffer(buff);

        b ^= leader;
    } while (b);
}

__device__ void dprinti(int32_t x){
    printf("%d\n", x);
}

__device__ void dprintptr(void * x){
    printf("%p\n", x);
}
#endif
}

void initializeModule(CUmodule & cudaModule){
#ifndef NCUDA
    CUdeviceptr ptr  ;
    size_t      bytes;
    void *      mem  ;

    gpu_run(cuModuleGetGlobal(&ptr, &bytes, cudaModule, "pool"));
    gpu_run(cudaMemcpyFromSymbol(&mem, pool      , sizeof(void   *)));
    gpu_run(cuMemcpyHtoD        (ptr , &mem      , sizeof(void   *)));

    gpu_run(cuModuleGetGlobal(&ptr, &bytes, cudaModule, "buff_start"));
    gpu_run(cudaMemcpyFromSymbol(&mem, buff_start, sizeof(void   *)));
    gpu_run(cuMemcpyHtoD        (ptr , &mem      , sizeof(void   *)));

    gpu_run(cuModuleGetGlobal(&ptr, &bytes, cudaModule, "buff_end"));
    gpu_run(cudaMemcpyFromSymbol(&mem, buff_end, sizeof(void   *)));
    gpu_run(cuMemcpyHtoD        (ptr , &mem      , sizeof(void   *)));

    gpu_run(cuModuleGetGlobal(&ptr, &bytes, cudaModule, "deviceId"));
    gpu_run(cudaMemcpyFromSymbol(&mem, deviceId  , sizeof(int)));
    gpu_run(cuMemcpyHtoD        (ptr , &mem      , sizeof(int)));
#endif
}


#ifndef NCUDA
__global__ void release_buffer_host(void **buff, int buffs){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    for (int i = 0 ; i < buffs ; ++i) buffer_manager<int32_t>::release_buffer((int32_t *) buff[i]);
}

__global__ void get_buffer_host(void **buff, int buffs){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    for (int i = 0 ; i < buffs ; ++i) buff[i] = buffer_manager<int32_t>::try_get_buffer();
}
#endif

int num_of_gpus;
int num_of_cpus;

inline int get_gpu_count(){
    return  num_of_gpus;
}

inline int get_cpu_numa_node_count(){
    return num_of_cpus;
}

int                                                 cpu_cnt;
// cpu_set_t                                          *gpu_affinity;
cpu_set_t                                          *cpu_numa_affinity;
int                                                *gpu_numa_node;

#if defined(__clang__) && defined(__CUDA__)
template<typename T>
__device__ T * buffer_manager<T>::get_buffer(){
    return pool->pop();
}

template<typename T>
__host__   T * buffer_manager<T>::get_buffer(){
    return get_buffer_numa(get_affinity());
}
#else
template<typename T>
__host__ __device__ T * buffer_manager<T>::get_buffer(){
#ifdef __CUDA_ARCH__
    return pool->pop();
#else
    return get_buffer_numa(get_affinity());
#endif
}
#endif

template<typename T>
__device__ T * buffer_manager<T>::try_get_buffer(){
#ifndef NCUDA
    T * b;
    bool got = pool->pop_if_nonempty(&b);
    if (!got) b = NULL;
    return b;
#else 
    return NULL;
#endif
}

#include <topology/topology.hpp>

template<typename T>
__host__ void buffer_manager<T>::init(int size, int h_size, size_t buff_buffer_size, size_t buff_keep_threshold){
    const topology &topo = topology::getInstance();
    // std::cout << topo << std::endl;

    uint32_t devices = topo.getGpuCount();
    buffer_manager<T>::h_size = h_size;
    
    uint32_t cores   = topo.getCoreCount();

    {

        //FIXME: Generalize
        uint32_t cpu_numa_nodes = topo.getCpuNumaNodeCount();
        uint32_t max_numa_id    = 0;
        for (const auto &n: topo.getCpuNumaNodes()){
            max_numa_id = std::max(max_numa_id, n.id);
        }

        // std::cout << "CPU numa nodes : " << cpu_numa_nodes << std::endl;
        // std::cout << "CPU cores      : " << cores << std::endl;
        // std::cout << "GPU devices    : " << devices << std::endl;

        terminating        = false;
        device_buffs_mutex = new mutex              [devices];
        device_buffs_cv    = new condition_variable [devices];
        device_buffs_thrds = new thread *           [devices];
        device_buffs_pool  = new vector<T *>        [devices];
        release_streams    = new cudaStream_t       [devices];

        h_pool             = new h_pool_t *         [cores  ];
        h_pool_numa        = new h_pool_t *         [max_numa_id + 1];

        h_d_pool           = new pool_t            *[devices];

        h_buff_start       = new void              *[devices];
        h_buff_end         = new void              *[devices];

        h_h_buff_start     = new void              *[max_numa_id + 1];

        device_buff        = new T **[devices];
        device_buff_size   = buff_buffer_size;
        keep_threshold     = buff_keep_threshold;

        buffer_cache.clear();

    }

    // gpu_run(cudaMallocHost(&tmp, device_buff_size*sizeof(buffer_t *)*devices));
    // for (int i = 0 ; i < devices ; ++i) {
        // device_buff[i] = tmp + device_buff_size*sizeof(buffer_t *)*i;
    // }

    // gpu_run(nvmlInit());
    // unsigned int device_count = 0;
    // gpu_run(nvmlDeviceGetCount(&device_count));
    // assert(device_count == devices && "NMVL disagrees with cuda about the number of GPUs");


    // gpu_affinity       = new cpu_set_t[devices];
    // cpu_numa_affinity  = new cpu_set_t[cpu_numa_nodes];
    // gpu_numa_node      = new int      [devices];

    // for (int j = 0 ; j < devices        ; ++j) CPU_ZERO(&gpu_affinity[j]);
    // for (int j = 0 ; j < cpu_numa_nodes ; ++j) CPU_ZERO(&cpu_numa_affinity[j]);


    // for (int j = 0 ; j < devices        ; ++j) std::cout << gpu_affinity[j]      << std::endl;
    // for (int j = 0 ; j < cpu_numa_nodes ; ++j) std::cout << cpu_numa_affinity[j] << std::endl;

    // diascld36
    // //FIXME: Generalize
    // for (int i = 0 ; i < 48 ; i += 2) CPU_SET(i, &gpu_affinity[0]);
    // for (int i = 1 ; i < 48 ; i += 2) CPU_SET(i, &gpu_affinity[1]);

    // //FIXME: Generalize
    // cpu_numa_affinity  = new cpu_set_t[cpu_numa_nodes];
    // for (int i = 0 ; i < 48 ; i += 2) CPU_SET(i, &cpu_numa_affinity[0]);
    // for (int i = 1 ; i < 48 ; i += 2) CPU_SET(i, &cpu_numa_affinity[1]);

//     for (int j = 0 ; j < devices ; ++j){
// #ifndef NCUDA
//         int sets = ((cores + 63) / 64);
//         uint64_t cpuSet[sets];
//         for (int i = 0 ; i < sets ; ++i) cpuSet[i] = 0;
//         nvmlDevice_t device;

//         nvmlDeviceGetHandleByIndex(j, &device);
//         nvmlDeviceGetCpuAffinity(device, sets, cpuSet);
//         for (int i = 0 ; i < sets ; ++i){
//             for (int k = 0 ; k < 64 ; ++k){
//                 if ((cpuSet[i] >> k) & 1){
//                     std::cout << "d" << j << " " << (64 * i + k) << std::endl;
//                     CPU_SET(64 * i + k, &gpu_affinity[j]);
//                 }
//             }
//         }
// #else
//         assert(false);
// #endif
//     }

    // for (int j = 0 ; j < devices        ; ++j) std::cout << gpu_affinity[j]      << std::endl;
    // for (int j = 0 ; j < cpu_numa_nodes ; ++j) std::cout << cpu_numa_affinity[j] << std::endl;

    // num_of_gpus = devices;
    // num_of_cpus = 0;

    // std::cout << devices << std::endl;
    // std::cout << cores   << std::endl;

    // for (int j = 0 ; j < cores ; ++j){
    //     std::cout << j << " " << numa_node_of_cpu(j) << std::endl;
    //     CPU_SET(j, &cpu_numa_affinity[numa_node_of_cpu(j)]);
    // }

    // for (int j = 0 ; j < cpu_numa_nodes ; ++j){
    //     std::cout << cpu_numa_affinity[j] << std::endl;
    //     if (CPU_COUNT(&cpu_numa_affinity[j]) > 0) ++num_of_cpus;
    // }



    //numa_node_of_cpu must be set prior to this
    // for (int j = 0 ; j < devices        ; ++j) gpu_numa_node[j] = calc_numa_node_of_gpu(j);

    // for (int i = 0 ; i < cores ; ++i){
    //     std::cout << "CPU " << i << " local to GPU ";
    //     for (int j = 0 ; j < devices ; ++j){
    //         if (CPU_ISSET(i, &gpu_affinity[j])) std::cout << j;
    //     }
    //     std::cout << std::endl;
    // }

    // for (int i = 0 ; i < cores ; ++i){
    //     std::cout << "CPU " << i << " local to NUMA ";
    //     for (int j = 0 ; j < cpu_numa_nodes ; ++j){
    //         if (CPU_ISSET(i, &cpu_numa_affinity[j])) std::cout << j;
    //     }
    //     std::cout << std::endl;
    // }


    // // diascld37
    // //FIXME: Generalize
    // for (int i = 0  ; i < 14 ; ++i) CPU_SET(i, &gpu_affinity[0]);
    // for (int i = 0  ; i < 14 ; ++i) CPU_SET(i, &gpu_affinity[1]);
    // for (int i = 14 ; i < 28 ; ++i) CPU_SET(i, &gpu_affinity[2]);
    // for (int i = 14 ; i < 28 ; ++i) CPU_SET(i, &gpu_affinity[3]);

    //FIXME: Generalize
    // cpu_numa_affinity  = new cpu_set_t[cpu_numa_nodes];
    // for (int i = 0  ; i < 14 ; ++i) CPU_SET(i, &cpu_numa_affinity[0]);
    // for (int i = 14 ; i < 28 ; ++i) CPU_SET(i, &cpu_numa_affinity[1]);

    mutex buff_cache;

    vector<thread> buffer_pool_constrs;
    for (const auto &gpu: topo.getGpus()){
        buffer_pool_constrs.emplace_back([gpu, size, &buff_cache]{
            uint32_t j = gpu.id;

            set_exec_location_on_scope d(j);

            T      *mem;
            size_t  pitch;
            gpu_run(cudaMallocPitch(&mem, &pitch, h_vector_size*sizeof(T), size));
            
            vector<T *> buffs;
            
            buffs.reserve(size);
            for (size_t i = 0 ; i < size ; ++i) {
                T        * m = (T *) (((char *) mem) + i*pitch);
                // buffer_t * b = cuda_new<buffer_t>(j, m, j);
                buffs.push_back(m);

                // cout << "Device " << j << " : data = " << m << endl;
                assert(topology::getInstance().getGpuAddressed(m)->id == j);
            }
            {   //FIXME: why are we including device buffers in the cache?
                lock_guard<mutex> guard(buff_cache);
                for (const auto b: buffs) buffer_cache[b] = 0;
            }
            
            pool_t * tmp =  cuda_new<pool_t>(j, size, buffs, j);
            gpu_run(cudaMemcpyToSymbol(pool      , &tmp, sizeof(pool_t *)));
            gpu_run(cudaMemcpyToSymbol(deviceId  ,   &j, sizeof(int     )));
            gpu_run(cudaMemcpyToSymbol(buff_start, &mem, sizeof(void   *)));
            void * e = (void *) (((char *) mem) + size*pitch);
            gpu_run(cudaMemcpyToSymbol(buff_end  ,   &e, sizeof(void   *)));

            h_d_pool    [j] = tmp;
            h_buff_start[j] = mem;
            h_buff_end  [j] = e  ;
            
            int greatest;
            int lowest;
            gpu_run(cudaDeviceGetStreamPriorityRange(&greatest, &lowest));
            // std::cout << greatest << " " << lowest << std::endl;
            gpu_run(cudaStreamCreateWithPriority(&(release_streams[j]), cudaStreamNonBlocking, lowest));

            T **bf;
            gpu_run(cudaMallocHost(&bf, std::max(device_buff_size, keep_threshold)*sizeof(T *)));
            device_buff[j] = bf;

            device_buffs_thrds[j] = new thread(dev_buff_manager, j);
        });
    }


    for (const auto &cpu: topo.getCpuNumaNodes()){
        buffer_pool_constrs.emplace_back([cpu, h_size, cores, &buff_cache]{
            set_exec_location_on_scope cu{cpu};
            const auto &topo = topology::getInstance();

            size_t bytes = h_vector_size * sizeof(T) * h_size;
            T      *mem = (T *) RawMemoryManager::mallocPinned(bytes);
            assert(mem);

            // T * mem;
            // gpu_run(cudaMallocHost(&mem, h_vector_size*sizeof(T)*h_size));
            printf("Memory at %p is at node %d (expected: %d)\n", mem, topo.getCpuNumaNodeAddressed(mem)->id, get_affinity().id);
            // assert(topo.getCpuNumaNodeAddressed(mem)->id == cpu.id); //FIXME: fails on power9, should reenable after we fix it

            h_h_buff_start[cpu.id] = mem;

            vector<T *> buffs;
            buffs.reserve(h_size);
            for (size_t j = 0 ; j < h_size ; ++j) {
                T        * m = mem + j * h_vector_size;
                // buffer_t * b = cuda_new<buffer_t>(-1, m, -1);
                buffs.push_back(m);

                m[0] = 0; //force allocation of first page of each buffer
                // cout << "NUMA " << topo.getCpuNumaNodeAddressed(m)->id << " : data = " << m << endl;
                // assert(topo.getCpuNumaNodeAddressed(m)->id == cpu.id); //FIXME: fails on power9, should reenable after we fix it
            }

            {
                lock_guard<mutex> guard(buff_cache);
                for (const auto b: buffs) buffer_cache[b] = 0;
            }

            h_pool_t *p         = new h_pool_t(h_size, buffs);
            
            h_pool_numa[cpu.id] = p;

            for (const auto &core: cpu.local_cores) h_pool[core] = p;
        });
    }

    // h_pool_t **numa_h_pools = new h_pool_t *[cpu_numa_nodes];

    // for (int i = 0 ; i < cores ; ++i) numa_node_inited[i] = NULL;

    // for (int i = 0 ; i < cores ; ++i){
    //     int numa_node = numa_node_of_cpu(i);

    //     if (!numa_node_inited[numa_node]){
    //         cpu_set_t cpuset;
    //         CPU_ZERO(&cpuset);
    //         CPU_SET(i, &cpuset);


    //         T      *mem;
    //         gpu_run(cudaMallocHost(&mem, buffer_t::capacity()*sizeof(T)*size));

    //         vector<buffer_t *> buffs;
    //         for (size_t i = 0 ; i < size ; ++i) buffs.push_back(cuda_new<buffer_t>(-1, mem + i * buffer_t::capacity(), -1));
    //         numa_node_inited[numa_node] = new h_pool_t(size, buffs);
    //     }
    //     h_pool[i] = numa_node_inited[numa_node];
    // }

    // T      *mem;
    // gpu_run(cudaMallocHost(&mem, buffer_t::capacity()*sizeof(T)*size));

    // vector<buffer_t *> buffs;
    // for (size_t i = 0 ; i < size ; ++i) buffs.push_back(cuda_new<buffer_t>(-1, mem + i * buffer_t::capacity(), -1));
    // h_pool = new h_pool_t(size, buffs);

    for (auto &t: buffer_pool_constrs) t.join();

    buffer_logger = new thread{buffer_manager<T>::log_buffers};
}

template<typename T>
__host__ void buffer_manager<T>::destroy(){
    int devices;
    gpu_run(cudaGetDeviceCount(&devices));

    // long cores = sysconf(_SC_NPROCESSORS_ONLN);
    // assert(cores > 0);

    // int cpu_numa_nodes = numa_num_task_nodes();

    terminating = true;

    buffer_logger->join();
    // device_buffs_mutex = new mutex              [devices];
    // device_buffs_pool  = new vector<buffer_t *> [devices];
    // release_streams    = new cudaStream_t       [devices];

    // h_pool             = new h_pool_t *         [cores  ];
    // h_pool_numa        = new h_pool_t *         [cpu_numa_nodes];

    // device_buff        = new buffer_t**[devices];
    // device_buff_size   = buff_buffer_size;
    // keep_threshold     = buff_keep_threshold;

    // gpu_affinity       = new cpu_set_t[devices];

    // mutex buff_cache;

    const auto &topo = topology::getInstance();

    vector<thread> buffer_pool_constrs;
    for (const auto &gpu: topo.getGpus()) {
#ifndef NCUDA
        buffer_pool_constrs.emplace_back([gpu]{
            uint32_t j = gpu.id;
            set_device_on_scope d(j);

            device_buffs_cv[j].notify_all();
            device_buffs_thrds[j]->join();

            std::unique_lock<std::mutex> lock(device_buffs_mutex[j]);

            size_t size = device_buffs_pool[j].size();
            assert(size <= keep_threshold);
            for (size_t i = 0 ; i < size ; ++i) device_buff[j][i] = device_buffs_pool[j][i];

            release_buffer_host<<<1, 1, 0, release_streams[j]>>>((void **) device_buff[j], size);
            gpu_run(cudaStreamSynchronize(release_streams[j]));

            pool_t *tmp;
            gpu_run(cudaMemcpyFromSymbol(&tmp, pool, sizeof(pool_t *)));
            cuda_delete(tmp);

            T * mem;
            gpu_run(cudaMemcpyFromSymbol(&mem, buff_start, sizeof(void   *)));
            gpu_run(cudaFree(mem));

            gpu_run(cudaStreamDestroy(release_streams[j]));

            gpu_run(cudaFreeHost(device_buff[j]));
        });
#else
        assert(false);
#endif
    }
    
    size_t h_size = buffer_manager<T>::h_size;

    for (const auto &cpu: topo.getCpuNumaNodes()){
        buffer_pool_constrs.emplace_back([cpu, h_size]{
            set_exec_location_on_scope cu{cpu};
            RawMemoryManager::freePinned(h_h_buff_start[cpu.id]);
            delete h_pool_numa[cpu.id];
        });
    }

    for (auto &t: buffer_pool_constrs) t.join();

    terminating        = false;
    delete[] device_buffs_mutex;
    delete[] device_buffs_cv   ;
    delete[] device_buffs_thrds;
    delete[] device_buffs_pool ;
    delete[] release_streams   ;

    delete[] h_pool            ;
    delete[] h_pool_numa       ;

    delete[] h_d_pool          ;

    delete[] h_buff_start      ;
    delete[] h_buff_end        ;

    delete[] h_h_buff_start    ;

    delete[] device_buff       ;

    buffer_cache.clear();
}

extern "C"{
    void * get_buffer(size_t bytes){
        assert(bytes <= sizeof(int32_t) * h_vector_size); //FIMXE: buffer manager should be able to allocate blocks of arbitary size
        return (void *) buffer_manager<int32_t>::h_get_buffer(-1);
    }

    void   release_buffer(void * buff){
        buffer_manager<int32_t>::release_buffer((int32_t *) buff);
    }
}

template<typename T>
void buffer_manager<T>::dev_buff_manager(int dev){
#ifndef NCUDA
    set_device_on_scope d(dev);

    while (true){
        bool sleep = false;
        int added = 0;
        {
            std::unique_lock<mutex> lk(device_buffs_mutex[dev]);
            
            device_buffs_cv[dev].wait(lk, [dev]{return device_buffs_pool[dev].empty() || terminating;});

            if (terminating) break;

            get_buffer_host<<<1, 1, 0, release_streams[dev]>>>((void **) device_buff[dev], device_buff_size);
            gpu_run(cudaStreamSynchronize(release_streams[dev]));


            for (size_t i = 0 ; i < device_buff_size ; ++i){
                if (device_buff[dev][i]) {
                    device_buffs_pool[dev].push_back(device_buff[dev][i]);
                    ++added;
                }
            }

            device_buffs_cv[dev].notify_all();

            sleep = device_buffs_pool[dev].empty();

            lk.unlock();
        }

        if (sleep) {
            std::cout << "Sleeping... (" << added << ")" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "Waking..." << std::endl;
        }
        // device_buffs_pool[dev].insert(device_buffs_pool[dev].end(), device_buff[dev], device_buff[dev]+device_buff_size);

        // lk.unlock();
    }
#endif
}
 
template<typename T>
__host__ void buffer_manager<T>::log_buffers(){
    return;
    const auto &topo = topology::getInstance();
    uint32_t devices = topo.getGpuCount();
    if (devices <= 0) return;

    uint32_t        cnts[devices];
    cudaStream_t    strs[devices];
    
    for (const auto &gpu: topo.getGpus()){
        set_device_on_scope d(gpu);
        gpu_run(cudaStreamCreateWithFlags(strs+gpu.id, cudaStreamNonBlocking));
    }

    char progress[]{"-\\|/"};
    size_t iter = 0;

    while (!terminating){
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        for (uint32_t i = 0 ; i < devices ; ++i){
            gpu_run(cudaMemcpyAsync(cnts + i, (void *) &(h_d_pool[i]->cnt), sizeof(decltype(pool_t::cnt)), cudaMemcpyDefault, strs[i]));
        }
        for (uint32_t i = 0 ; i < devices ; ++i) gpu_run(cudaStreamSynchronize(strs[i]));
        std::cerr << "\0337\033[H\r";
        for (uint32_t i = 0 ; i < 80 ; ++i) std::cerr << ' ';
        std::cerr << "\rBuffers on device: ";
        for (uint32_t i = 0 ; i < devices ; ++i) std::cerr << cnts[i] << "(+" << device_buffs_pool[i].size() << ") ";
        std::cerr << "\t\t" << progress[(iter++) % (sizeof(progress) - 1)]; //for null character
        std::cerr << "\0338";
        std::cerr.flush();
    }

    for (const auto &gpu: topo.getGpus()){
        set_device_on_scope d(gpu);
        gpu_run(cudaStreamDestroy(strs[gpu.id]));
    }
}

template<typename T>
__host__ inline T * buffer_manager<T>::h_get_buffer(int dev){
    if (dev >= 0){
        std::unique_lock<std::mutex> lock(device_buffs_mutex[dev]);

        device_buffs_cv[dev].wait(lock, [dev]{return !device_buffs_pool[dev].empty();});

        T * ret = device_buffs_pool[dev].back();
        device_buffs_pool[dev].pop_back();
        device_buffs_cv[dev].notify_all();
        return ret;
    } else {
        return get_buffer();
    }
}


template<typename T>
__host__ void buffer_manager<T>::overwrite(T * buff, const T * data, uint32_t N, cudaStream_t strm, bool blocking){
#ifndef NCUDA
    gpu_run(cudaMemcpyAsync(buff, data, N*sizeof(T), cudaMemcpyDefault, strm));
    if (blocking) gpu_run(cudaStreamSynchronize(strm));
#else
    memcpy(buff, data, N*sizeof(T));
#endif
}

template<typename T>
__host__ void buffer_manager<T>::overwrite_bytes(void * buff, const void * data, size_t bytes, cudaStream_t strm, bool blocking){
#ifndef NCUDA
    gpu_run(cudaMemcpyAsync(buff, data, bytes, cudaMemcpyDefault, strm));
    if (blocking) gpu_run(cudaStreamSynchronize(strm));
#else
    memcpy(buff, data, bytes);
#endif
}

template class buffer_manager<int32_t>;

#ifndef NCUDA
__global__ void GpuHashRearrange_acq_buffs(void   ** buffs){
    buffs[blockIdx.x] = get_buffers();
}
#endif

extern "C"{
void gpu_memset(void * dst, int32_t val, size_t size){
    cudaStream_t strm;
    gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
    gpu_run(cudaMemsetAsync(dst, val, size, strm));
    gpu_run(cudaStreamSynchronize(strm));
    gpu_run(cudaStreamDestroy(strm));
}
}
