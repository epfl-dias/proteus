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

#ifndef ASYNC_CONTAINERS_HPP_
#define ASYNC_CONTAINERS_HPP_

#include <mutex>
#include <stack>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <iostream>

#include "common/gpu/gpu-common.hpp"

template<typename T>
class AsyncStackSPSC{
private:
    std::mutex              m   ;
    std::condition_variable cv  ;
    
    std::vector<T>          data;

    std::atomic<bool>       terminating;
public:
    AsyncStackSPSC(): terminating(false){}

    void close(){
        nvtxRangePushA("AsyncStack_o");
        terminating = true;
        cv.notify_all();

        nvtxRangePushA("AsyncStack_t");
        std::unique_lock<std::mutex> lock(m);

        cv.wait(lock, [this](){return data.empty();});

        lock.unlock();
        nvtxRangePop();
        nvtxRangePop();
    }

    void push(const T &x){
        assert(!terminating);
        std::unique_lock<std::mutex> lock(m);
        data.emplace_back(x);
        cv.notify_all();
        lock.unlock();
    }

    bool pop(T &x){
        std::unique_lock<std::mutex> lock(m);

        if (data.empty()){
            cv.wait(lock, [this](){return !data.empty() || (data.empty() && terminating);});
        }

        if (data.empty()){
            assert(terminating);
            lock.unlock();

            cv.notify_all();
            return false;
        }

        x = data.back();
        data.pop_back();

        lock.unlock();
        return true;
    }

    T pop_unsafe(){
        T x = data.back();
        data.pop_back();
        return x;
    }
};

template<typename T>
class AsyncQueueSPSC{
private:
    std::mutex              m   ;
    std::condition_variable cv  ;
    
    std::queue<T>           data;

    std::atomic<bool>       terminating;
public:
    AsyncQueueSPSC(): terminating(false){}

    void close(){
        nvtxRangePushA("AsyncQueue_o");
        terminating = true;
        cv.notify_all();

        nvtxRangePushA("AsyncQueue_t");
        std::unique_lock<std::mutex> lock(m);

        cv.wait(lock, [this](){return data.empty();});

        lock.unlock();
        nvtxRangePop();
        nvtxRangePop();
    }

    void push(const T &x){
        assert(!terminating);
        std::unique_lock<std::mutex> lock(m);
        data.emplace(x);
        cv.notify_all();
        lock.unlock();
    }

    bool pop(T &x){
        std::unique_lock<std::mutex> lock(m);

        if (data.empty()){
            cv.wait(lock, [this](){return !data.empty() || (data.empty() && terminating);});
        }

        if (data.empty()){
            assert(terminating);
            lock.unlock();

            cv.notify_all();
            return false;
        }

        x = data.front();
        data.pop();

        lock.unlock();
        return true;
    }

    T pop_unsafe(){
        T x = data.front();
        data.pop();
        return x;
    }
};

// template<typename T, size_t size>
// class AsyncQueueSPSC_spin{
// private:
//     volatile uint32_t front;
//     volatile uint32_t back ; //TODO: should we put them in different cache lines ?

//     volatile bool     terminating;
//     volatile T data[size];
// public:
//     AsyncQueueSPSC(): front(0), back(-1), terminating(false){}

//     void close(){
//         nvtxRangePushA("AsyncQueue_o");
//         terminating = true;

//         while (f > back + 1);
//         nvtxRangePop();
//     }

//     void push(const T &x){
//         // assert(!terminating);
//         uint32_t f = front;
//         while (f - back - 1 >= size);

//         data[f % size] = x;

//         front = f + 1;
//     }

//     bool pop(T &x){
//         uint32_t b = back;

//         while (front <= b + 1 && !terminating);

//         x = data[b % size];

//         return true;
//     }

//     T pop_unsafe(){
//         T x = data.front();
//         data.pop();
//         return x;
//     }
// };


#endif /* ASYNC_CONTAINERS_HPP_ */

