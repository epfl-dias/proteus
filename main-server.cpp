/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/

#include "util/raw-memory-manager.hpp"
#include "util/raw-pipeline.hpp"
#include "storage/raw-storage-manager.hpp"
#include "util/gpu/gpu-raw-context.hpp"
#include "plan/plan-parser.hpp"

//https://stackoverflow.com/a/25829178/1237824
std::string trim(const std::string& str){
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first) return str;
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

//https://stackoverflow.com/a/7756105/1237824
bool starts_with(const std::string& s1, const std::string& s2) {
    return s2.size() <= s1.size() && s1.compare(0, s2.size(), s2) == 0;
}

constexpr size_t clen(const char* str){
    return (*str == 0) ? 0 : clen(str + 1) + 1;
}

const char * catalogJSON = "inputs/plans/catalog.json";

void executePlan(const char *label, const char *planPath, const char *catalogJSON){
    int devices = get_num_of_gpus();
    // for (int i = 0 ; i < devices ; ++i) {
    //     gpu_run(cudaSetDevice(i));
    //     gpu_run(cudaProfilerStart());
    // }
    // __itt_resume();
    {
        RawCatalog     * catalog = &RawCatalog::getInstance();
        CachingService * caches  = &CachingService::getInstance();
        catalog->clear();
        caches->clear();
    }

    gpu_run(cudaSetDevice(0));

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        GpuRawContext * ctx   = new GpuRawContext(label, false);
        CatalogParser catalog = CatalogParser(catalogJSON, ctx);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, label, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    //just to be sure...
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaDeviceSynchronize());
    }
    
    {
        time_block     t("Texecute w sync: ");

        {
            time_block t("Texecute       : ");

            for (RawPipeline * p: pipelines) {
                nvtxRangePushA("pip");
                {
                    time_block t("T: ");
    
                    p->open();
                    p->consume(0);
                    p->close();
                }
                nvtxRangePop();
            }
        }

        //just to be sure...
        for (int i = 0 ; i < devices ; ++i) {
            gpu_run(cudaSetDevice(i));
            gpu_run(cudaDeviceSynchronize());
        }
    }

    // __itt_pause();
    // for (int i = 0 ; i < devices ; ++i) {
    //     gpu_run(cudaSetDevice(i));
    //     gpu_run(cudaProfilerStop());
    // }

    gpu_run(cudaSetDevice(0));
}

void executePlan(const char *label, const char *planPath){
    executePlan(label, planPath, catalogJSON);
}

class unlink_upon_exit{
    size_t query;
    std::string label_prefix;

    std::string last_label;
public:
    unlink_upon_exit(): 
        query(0), 
        label_prefix("raw_server_" + std::to_string(getpid()) + "_q"), 
        last_label(""){}

    unlink_upon_exit(size_t unique_id): 
        query(0), 
        label_prefix("raw_server_" + std::to_string(unique_id) + "_q"), 
        last_label(""){}

    ~unlink_upon_exit(){
        if (last_label != "") shm_unlink(last_label.c_str());
    }

    std::string get_label(){
        return last_label;
    }

    std::string inc_label(){
        if (query > 0) shm_unlink(last_label.c_str());
        last_label = label_prefix + std::to_string(query++);
        return last_label;
    }
};

void thread_warm_up(){}

/**
 * Protocol:
 * 
 * Communication is done over stdin/stdout
 * Command spans at most one line
 * Every line either starts with a command keyword or it should be IGNORED and 
 *      considered a comment
 * Input commands:
 *
 *      quit 
 *          Kills the raw-jit-executor engine
 *
 *      execute plan <plan_description>
 *          Executes the plan described from the <plan_description>
 *          It will either result in an error command send back, or a result one
 *
 *          Valid plan descriptions:
 *
 *              from file <file_path>
 *                  Reads the plan from the file pointed by the <file_path>
 *                  The file path is either an absolute path, or a path relative
 *                  to the current working directory
 *
 *     echo <object_to_echo>
 *          Switched on/off the echoing of types of results. When switched on,
 *          in general, replies with the specific type of object that were
 *          to be written in files, are also echoed to stdout
 *
 *          Valid to-echo-objects:
 *              results (on/off)
 *                  Prints results in output as well.
 *                  Use with causion! Results may be binary or contain new lines
 *                  with keywords!
 *                  Default: off
 *
 * Output commands:
 *      ready
 *          Send to the client when the raw-jit-executor is ready to start
 *          receiving commands
 *      error [(<reason>)]
 *          Specifies that a previous command or the engine failed.
 *          The optional (<reason>) specified in parenthesis a human-readable
 *          explanation of the error. The error may be fatal or not.
 *      result <result_description>
 *          Specifies the result of the previous command, if any
 *          
 *          Valid result descriptions:
 *              in file <file_path>
 *                  The result is saved in file pointed by the <file_path>
 *                  The file path is either an absolute path, or a path relative
 *                  to the current working directory
 *              echo
 *                  The following line/lines are results printed into stdout
 */
int main(int argc, char* argv[]){
    // Initialize Google's logging library.
    google::InitGoogleLogging(argv[0]);
    LOG(INFO)<< "Starting up server...";

    bool echo = false;

    google::InstallFailureSignalHandler();

    LOG(INFO)<< "Warming up GPUs...";
    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaFree(0));
    }

    gpu_run(cudaFree(0));

    // gpu_run(cudaDeviceSetLimit(cudaLimitStackSize, 40960));

    LOG(INFO)<< "Warming up threads...";

    std::vector<std::thread> thrds;
    for (int i = 0 ; i < 32 ; ++i) thrds.emplace_back(thread_warm_up);
    for (auto &t: thrds) t.join();

    // srand(time(0));

    LOG(INFO)<< "Initializing codegen...";

    RawPipelineGen::init();

    LOG(INFO)<< "Initializing memory manager...";
    RawMemoryManager::init();

    gpu_run(cudaSetDevice(0));
    LOG(INFO)<< "Eagerly loading files in memory...";


    //FIXME: remove, we should be loading files lazily
    {
        auto load = [](string filename){
            StorageManager::load(filename, PINNED);
        };
        
        load("inputs/ssbm100/customer.csv.c_city");
        load("inputs/ssbm100/customer.csv.c_custkey");
        load("inputs/ssbm100/customer.csv.c_nation");
        load("inputs/ssbm100/customer.csv.c_region");

        load("inputs/ssbm100/date.csv.d_datekey");
        load("inputs/ssbm100/date.csv.d_weeknuminyear");
        load("inputs/ssbm100/date.csv.d_year");
        load("inputs/ssbm100/date.csv.d_yearmonth");
        load("inputs/ssbm100/date.csv.d_yearmonthnum");
        load("inputs/ssbm100/date.csv.d_date");

        load("inputs/ssbm100/lineorder.csv.lo_custkey");
        load("inputs/ssbm100/lineorder.csv.lo_discount");
        load("inputs/ssbm100/lineorder.csv.lo_extendedprice");
        load("inputs/ssbm100/lineorder.csv.lo_orderdate");
        load("inputs/ssbm100/lineorder.csv.lo_partkey");
        load("inputs/ssbm100/lineorder.csv.lo_quantity");
        load("inputs/ssbm100/lineorder.csv.lo_revenue");
        load("inputs/ssbm100/lineorder.csv.lo_suppkey");
        load("inputs/ssbm100/lineorder.csv.lo_supplycost");

        load("inputs/ssbm100/part.csv.p_brand1");
        load("inputs/ssbm100/part.csv.p_category");
        load("inputs/ssbm100/part.csv.p_mfgr");
        load("inputs/ssbm100/part.csv.p_partkey");
        load("inputs/ssbm100/part.csv.p_size");
        
        load("inputs/ssbm100/supplier.csv.s_city");
        load("inputs/ssbm100/supplier.csv.s_nation");
        load("inputs/ssbm100/supplier.csv.s_region");
        load("inputs/ssbm100/supplier.csv.s_suppkey");
        load("inputs/ssbm100/supplier.csv.s_name");

        // load("inputs/ssbm100/date.csv.d_datekey");
        // load("inputs/ssbm100/date.csv.d_year");
        // load("inputs/ssbm100/supplier.csv.s_suppkey");
        // load("inputs/ssbm100/supplier.csv.s_nation");
        // load("inputs/ssbm100/customer.csv.c_custkey");
        // load("inputs/ssbm100/customer.csv.c_region");
        // load("inputs/ssbm100/part.csv.p_partkey");
        // load("inputs/ssbm100/part.csv.p_category");
        // load("inputs/ssbm100/lineorder.csv.lo_custkey");
        // load("inputs/ssbm100/lineorder.csv.lo_partkey");
        // load("inputs/ssbm100/lineorder.csv.lo_suppkey");
        // load("inputs/ssbm100/lineorder.csv.lo_orderdate");
        // load("inputs/ssbm100/lineorder.csv.lo_revenue");
        // load("inputs/ssbm100/lineorder.csv.lo_supplycost");
    }
    gpu_run(cudaSetDevice(0));
    LOG(INFO)<< "Finished initialization";
    std::cout << "ready" << std::endl;
    std::string line;
    std::string prefix("--foo=");

    {
        unlink_upon_exit uue;

        while (std::getline(std::cin, line)) {
            std::string cmd = trim(line);

            LOG(INFO)<< "Command received: " << cmd;

            if (cmd == "quit") {
                std::cout << "quiting..." << std::endl;
                break;
            } else if (starts_with(cmd, "execute plan ")){
                if (starts_with(cmd, "execute plan from file ")){
                    constexpr size_t prefix_size = clen("execute plan from file ");
                    std::string plan  = cmd.substr(prefix_size);
                    std::string label = uue.inc_label();
                    executePlan(label.c_str(), plan.c_str());

                    if (echo){
                        std::cout << "result echo" << std::endl;
                        /* current */
                        int fd2 = shm_open(label.c_str(), O_RDONLY, S_IRWXU);
                        if (fd2 == -1) {
                            throw runtime_error(string(__func__) + string(".open (output): ")+label);
                        }
                        struct stat statbuf;
                        if (fstat(fd2, &statbuf)) {
                            fprintf(stderr, "FAILURE to stat test results! (%s)\n", std::strerror(errno));
                            assert(false);
                        }
                        size_t fsize2 = statbuf.st_size;
                        char *currResultBuf = (char*) mmap(NULL, fsize2, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE, fd2, 0);
                        fwrite(currResultBuf, sizeof(char), fsize2, stdout);
                        std::cout << std::endl;
                        // shm_unlink(label.c_str());
                        munmap(currResultBuf, fsize2);
                    }

                    std::cout << "result in file /dev/shm/" << label << std::endl;
                } else {
                    std::cout << "error (command not supported)" << std::endl;
                }
            } else if (starts_with(cmd, "echo")){
                if (cmd == "echo results on"){
                    echo = true;
                } else if (cmd == "echo results off"){
                    echo = false;
                } else {
                    std::cout << "error (unknown echo, please specify what to echo)" << std::endl;
                }
            } else if (starts_with(cmd, "codegen")){
                if (cmd == "codegen print on"){
                    print_generated_code = true;
                } else if (cmd == "codegen print off"){
                    print_generated_code = false;
                } else if (cmd == "codegen print query"){
                    std::cout << print_generated_code << std::endl;
                } else {
                    std::cout << "error (unknown codegen option, please specify what to echo)" << std::endl;
                }
            }
        }
    }
    LOG(INFO)<< "Shutting down...";

    LOG(INFO)<< "Unloading files...";
    StorageManager::unloadAll();

    LOG(INFO)<< "Shuting down memory manager...";
    RawMemoryManager::destroy();

    LOG(INFO)<< "Shut down finished";
    return 0;
}
