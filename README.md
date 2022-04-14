Proteus: Just-In-Time Executor
================

The executor of Pelago, a prototype in situ DBMS employing Just-In-Time operators.

Setting up git hooks
========

To setup the git hooks run:
```sh
git config core.hooksPath .githooks
```
This enables a variety of automatic notifications and configurations at commit-time, including formatting your committed code, checking conformance with licenses, worktree skip-lists, etc.

Allocate HugePages
========
```
echo 76800 | sudo tee /sys/devices/system/node/node{0,1}/hugepages/hugepages-2048kB/nr_hugepages
```

Update include paths in Clion after LLVM update
========

Clion lazily updates the include paths during remote deployment, use the resync with remote hosts to force a refresh: https://www.jetbrains.com/help/clion/remote-projects-support.html#resync


Testing
========
For our c++ code we use [GTest](https://github.com/google/googletest) for our functional testing and [googlebenchmark](https://github.com/google/benchmark) for performance testing. 
cpp integration tests live in `tests`. cpp Unit tests are in `tests` subdirectories in the component they test, and likewise performance tests are in `perftests` subdirectories. 
The planner uses junit. 