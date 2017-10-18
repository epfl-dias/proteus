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

#include <mutex>
#include <stack>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <iostream>

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
        terminating = true;
        cv.notify_all();

        std::unique_lock<std::mutex> lock(m);

        cv.wait(lock, [this](){return data.empty();});

        lock.unlock();
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

        cv.wait(lock, [this](){return !data.empty() || (data.empty() && terminating);});

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
        terminating = true;
        cv.notify_all();

        std::unique_lock<std::mutex> lock(m);

        cv.wait(lock, [this](){return data.empty();});

        lock.unlock();
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

        cv.wait(lock, [this](){return !data.empty() || (data.empty() && terminating);});

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
};

