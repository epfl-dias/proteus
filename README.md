Proteus: Just-In-Time Executor
================

The executor of Pelago, a prototype in situ DBMS employing Just-In-Time operators.

Setting up git hooks
========

To setup the git hooks run:
```
git config core.hooksPath .githooks
```


Allocate HugePages
========
```
echo 76800 | sudo tee /sys/devices/system/node/node{0,1}/hugepages/hugepages-2048kB/nr_hugepages
```

Update include paths in Clion after LLVM update
========

Clion lazily updates the include paths during remote deployment, use the resync with remote hosts to force a refresh: https://www.jetbrains.com/help/clion/remote-projects-support.html#resync
