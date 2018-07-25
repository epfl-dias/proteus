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
#ifndef EXCHANGE_HPP_
#define EXCHANGE_HPP_

#include "operators/operators.hpp"
#include "util/gpu/gpu-raw-context.hpp"
#include <queue>
#include <mutex>
#include <stack>
#include <condition_variable>
#include <atomic>
#include <thread>

class Exchange;

extern "C"{
    void * acquireBuffer    (int target, Exchange * xch);
    void * try_acquireBuffer(int target, Exchange * xch);
    void   releaseBuffer    (int target, Exchange * xch, void * buff);
    void   freeBuffer       (int target, Exchange * xch, void * buff);
}

class Exchange : public UnaryRawOperator {
public:
    Exchange(   RawOperator * const             child,
                GpuRawContext * const           context,
                int                             numOfParents,
                const vector<RecordAttribute*> &wantedFields,
                int                             slack,
                expressions::Expression       * hash = NULL,
                bool                            numa_local = true,
                bool                            rand_local_cpu = false,
                int                             producers = 1) :
                    UnaryRawOperator(child), 
                    context(context), 
                    numOfParents(numOfParents),
                    wantedFields(wantedFields),
                    slack(slack),
                    hashExpr(hash),
                    numa_local(numa_local),
                    rand_local_cpu(rand_local_cpu),
                    producers(producers),
                    remaining_producers(producers),
                    need_cnt(false){
        assert((!hash || !numa_local) && "Just to make it more clear that hash has precedence over numa_local");
        
        free_pool           = new std::stack<void *>     [numOfParents];
        free_pool_mutex     = new std::mutex             [numOfParents];
        free_pool_cv        = new std::condition_variable[numOfParents];

        ready_pool          = new std::queue<void *>     [numOfParents];
        ready_pool_mutex    = new std::mutex             [numOfParents];
        ready_pool_cv       = new std::condition_variable[numOfParents];
        
        // int devices = get_num_of_gpus();
        int devices = numa_num_task_nodes();

        for (int i = 0 ; i < numOfParents ; ++i){
            target_processors.emplace_back(i % devices);
        }
    }

    virtual ~Exchange()                                             { LOG(INFO)<<"Collapsing Exchange operator";}

    virtual void produce();
    virtual void consume(RawContext* const context, const OperatorState& childState);
    virtual bool isFiltering() const {return false;}

protected:
    virtual void generate_catch();

    void   fire         (int target, RawPipelineGen * pipGen);

private:
    void * acquireBuffer(int target, bool polling = false);
    void   releaseBuffer(int target, void * buff);
    void   freeBuffer   (int target, void * buff);
    bool   get_ready    (int target, void * &buff);

    friend void * acquireBuffer    (int target, Exchange * xch);
    friend void * try_acquireBuffer(int target, Exchange * xch);
    friend void   releaseBuffer    (int target, Exchange * xch, void * buff);
    friend void   freeBuffer       (int target, Exchange * xch, void * buff);

protected:
    void open (RawPipeline * pip);
    void close(RawPipeline * pip);

    const vector<RecordAttribute *> wantedFields;

    const int                       slack;
    const int                       numOfParents;
    int                             producers;
    std::atomic<int>                remaining_producers;
    GpuRawContext * const           context;

    llvm::Type                    * params_type;
    RawPipelineGen                * catch_pip  ;

    std::vector<std::thread>        firers;

    std::queue<void *>            * ready_pool;
    std::mutex                    * ready_pool_mutex;
    std::condition_variable       * ready_pool_cv;

    std::stack<void *>            * free_pool;
    std::mutex                    * free_pool_mutex;
    std::condition_variable       * free_pool_cv;

    std::mutex                      init_mutex;

    std::vector<exec_location>      target_processors;

    expressions::Expression       * hashExpr;
    bool                            numa_local;
    bool                            rand_local_cpu;

    bool                            need_cnt;
};

#endif /* EXCHANGE_HPP_ */


