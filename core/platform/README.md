# Platform
Platform is responsible for providing unified APIs to the underlying system, such as the kernel and the hardware devices.

## Topology
The [`topology`](topology/topology.hpp) discovers and provides a single point of entrance to devices.
During platform initialization, it detects the different devices and memories in the system such as CPUs, GPUs, NUMA nodes and IB devices.

### Devices
Each device type may or may not be a compute unit ([`cu`](topology/topology.hpp)), as well as it may or may not have a local memory ([`numanode`](topology/topology.hpp)).

Currently the provided devices are: [`cpunumanode`](topology/topology.hpp), [`gpunode`](topology/topology.hpp), [`ib`](network/infiniband/devices/ib.hpp)

### Affinity
Execution flows may jump across devices, thus affinities are described as a set of "approved" compute units per compute unit type.
For example, an execution flow (thread) may be allowed to execute on CPU1 and GPU2 and depending on the current execution unit, it will either be running on CPU1 or GPU2 (the intersection of the currently used device type and the affinity of that flow). 

## Space Management
Space management is (currently) based on singletons (some can consider the `topology` singleton as a special hardware-space component).
Space management is usually relying on `topology` and thus space management initialization follows as a second platform initialization phase.
Thus, topology is not allowed to call into space management components.

### Memory Management
[`Memory management`](memory/memory-manager.hpp) is responsible for allocating memory on different NUMA nodes by translating requests from the upper layers to the corresponding NUMA node allocations.

The upper layers may request memory attached to a specific device type and the memory managers are redirecting these allocations to the nearest (to the current processing affinity) device. 

### Block Management (old: Buffer Management)
[`Block management`](memory/block-manager.hpp) relies on memory management and it works on fixed-size blocks that are optimized for copying across numa nodes: blocks are working as a staging area that is specially handled to guarantee optimal transfer times over interconnects such as PCIe or IB.

### Storage Management
[`Storage management`](storage/storage-manager.hpp) is responsible for pulling, pinning and unpinning data from storage devices into (different types of) memory.
Different [`data_loc`](storage/mmap-file.hpp) policies are available to specify how the loaded data are distributed across the NUMA nodes.

## Component diagram 

```plantuml
digraph Platform {
    graph [rankdir = TB];
    compound=true;

    node[shape=record];
    Topology[label="{Topology|{<cpu>CPUs|<gpu>GPUs|<ib>IBs} }"];
    
    subgraph clusterSpace{
        label = "Space Management";
        
        BlockManager -> MemoryManager;
        StorageManager -> MemoryManager;
        InfiniBandManager -> BlockManager;
        InfiniBandManager -> MemoryManager;
    };

    Execution[label="Execution Layer"];

    MemoryManager -> Topology [ltail=clusterSpace];
    Execution -> Topology [ltail=clusterExecution];
    Execution -> InfiniBandManager [ltail=clusterExecution,lhead=clusterSpace];
}
```
