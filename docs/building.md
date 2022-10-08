## With docker (recommended)
We provide a docker container for our toolchain which has the required dependencies to configure and build proteus. (TODO link to new dockerhub profile )
#### Requirements:
Host system ubuntu 18.04 (we have tested extensively on 18.04, 20.04/22.04 should work, but it is not guaranteed) with docker + [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian) installed.

#### Building with the command line in docker
- clone this repo to a convenient location e.g
  ```sh
  usero@server01:~$ git clone git@github.com:epfl-dias/proteus.git
  ```
- Start the container and mount the source directory into the container:
  ```sh
  user@server01:~$ docker run -it --gpus all -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) --mount type=bind,source="$(pwd)"/proteus,target=/proteus --security-opt seccomp="$(pwd)"/proteus/tools/docker/proteus-build/seccomp.json --cap-add=SYS_NICE --cap-add=IPC_LOCK chapeiro/pelago-build:cuda11.3-llvm14.0.0
  ```
  - *Note* Like with most docker images, avoid using the `latest` tag, newer versions of the container may have  versions of LLVM and cuda which are incompatible with this commit.
  - *Note* In order to run benchmarks prepared datasets need to also be mounted into the docker container. In the case of SSB100(0) with `--mount type=bind,source=/path/to/ssb100,target=/proteus/tests/inputs/ssb100`
  - *Note* If you are using docker 19.03, use the alternative seccomp file: `seccomp-19-03.json`
  
- Setup docker environment for interactive use
  ```sh
  root@8eae36dca703:/# /proteus/tools/docker/utilities/enter-docker-env.sh
  ```
  This script will set the UID/GID inside the docker container to match the UID/GID of the host so that you will have permissions to write to the bind-mounted source directory.

- Configure CMake:
  ```sh
  user@8eae36dca703:/# cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=opt/ -S /proteus -B /proteus/cmake-build-debug
  ```
  - *Note* `Debug` is the only supported build type, it is not a traditional debug build, it is still an optimised build.
  - *Note* the first time doing this will take a couple of minutes, it will appear to hang after
    ```sh
    -- Found Python: /usr/bin/python3.6 (found version "3.6.9") found components: Interpreter
    ```
but it will eventually continue
- Build proteus
  ```sh
  user@8eae36dca703:/# cd proteus/cmake-build-debug
  user@8eae36dca703:/proteus/cmake-build-debug# make install -j 96
  ```
*Note* set -j to the available number of cores on your system for build parallelism
*Note* We use install as the preferred target to ensure all internal dependencies are up-to-date after any source modifications. Running binaries from their build location may lead to unexpected results. 
Binaries can then be run from the install location (`/proteus/opt/pelago`), all binaries should be run from this folder as they depend on the path to find datasets.

## Without Docker
#### Requirements:
- Ubuntu 18.04 (20.04/22.04 *should* work but are not tested in ci).
- Dependencies: CMake fetches and builds most dependencies, however there are some which must be installed at the system level. 
See the list in `docker/pelago-build/requirements.txt`, in addition to this list we also require LLVM and CMake. For the current minimum CMake version see the top of the root `CMakeLists.txt`. The current required LLVM version can be found in `cmake/modules/proteus-prolog.cmake`
