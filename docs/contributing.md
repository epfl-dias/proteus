# Reporting bugs
- Please use the [GitHub issues tab](https://github.com/epfl-dias/proteus/issues) to report bugs.
- Use a clear and descriptive title for the issue to identify the problem.
- Describe the exact steps which reproduce the problem in as many details as possible. 
- Provide a detailed description of your environment. e.g kernel/distribution version, whether you are running in docker or on bare metal and which commit you are using.

# How to submit changes
Please open a PR on the GitHub repo. Please follow our git conventions, see the git section below.

# Runtime Environment
## Hugepages
Proteus requires Linux hugepages. To allocate hugepages:
```sh
echo 76800 | sudo tee /sys/devices/system/node/node{0,1}/hugepages/hugepages-2048kB/nr_hugepages
```
You may need to vary the number of huge pages based on your system's memory.
We recommend using 80-90% of system memory if Proteus is the only resource intensive process running on the server (e.g. for benchmarking).
You may also need to change `node{0,1}` based on the number of numa nodes in your system. 

# Development Environment
See [building.md](building.md) on how to build Proteus.

## Git
### Conventions
We follow a fairly standard set of git conventions:
- We maintain a linear history. Please rebase on main before opening a pull request. 
- Please keep individual commits to manageable sizes. Each commit should be as self-contained as possible and under 500 lines.
- Commit messages should have a title starting with a tag, e.g `[storage] move storage into its own library`  (The current list of tags can be found in `.githooks/commit-msg`).
- Any non-trivial commit should have a message body detailing the reasoning/background of the changes, e.g describing pros/cons of alternate approaches, how the committer decided on this approach or links to external documentation/bug trackers where appropriate. 

### Setting up git hooks

To setup the git hooks run:
```sh
git config core.hooksPath .githooks
```
This enables a variety of automatic notifications and configurations at commit-time, including formatting your committed code, checking conformance with licenses, worktree skip-lists, etc.

Furthermore, it's highly recommended to run the following to use our predefined git config:
```sh
git config --local include.path .config/diascld/.gitconfig
```

## Clion
While you can of course use any editor to develop proteus, the editor of choice at DIAS is CLion. We commit a basic CLion project configuration in `.idea`.

### Update include paths in Clion after LLVM update
Clion lazily updates the include paths during remote deployment, use the resync with remote hosts to force a refresh: https://www.jetbrains.com/help/clion/remote-projects-support.html#resync

## CMake
Proteus uses CMake for the build system. We try to adhere to modern CMake principles. 

### Dependencies
Dependencies are broadly broken up into 1) system dependencies and 2) dependencies build via CMake.
However, there are also some dependencies where it is sometimes preferable to build them ourselves, but outside of CMake.
In particular, LLVM and CMake, since both are required to build dependencies and Proteus in CMake, and it is necessary to have fine-grained control over the LLVM and CMake versions. i.e distro repos may not always have the appropriate versions available. 
In general, we try to minimize system dependencies and where possible prefer to build dependencies ourselves in CMake. 


#### LLVM & CMake
You can build Proteus with an LLVM/CMake you compiled yourself, or a system install of LLVM/CMake from distro repos if it is the appropriate version. 
If you are using a self compiled LLVM/CMake you need to update your `PATH` and `LD_LIBRARY_PATH` environment variable before configuring CMake to build Proteus. e.g 
```sh
export LD_LIBRARY_PATH='PATH_TO_LLVM_INSTALL/lib:$LD_LIBRARY_PATH'
export PATH="PATH_TO_LLVM_INSTALL/bin:PATH_TO_CMAKE_INSTALL/bin:$PATH"
```
Because Proteus uses LLVM for runtime code generation, `LD_LIBRARY_PATH` must also be set so that Proteus can find the LLVM shared libraries can be found at runtime. 
In our CLion configs you may find reference to `/scratch/pelago/llvm-14/opt/`. This is the path of our internal toolchain, where we install both LLVM and CMake. 

### Editing CMake configurations with user-specific settings
Sometimes you want to invoke `cmake` with different flags or have multiple profiles.
CMake allows that through a `CMakeUserPresets.json` file that you can add locally with user-specific settings.

For example, if you want to produce verbose makefiles or use a pre-configured CLion/Gateway configuration, you can add in the project root the following `CMakeUserPresets.json` file:

```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "Proteus - User",
      "inherits": "Proteus",
      "cacheVariables": {
        "CMAKE_VERBOSE_MAKEFILE": "ON"
      },
      "vendor": {
        "jetbrains.com/clion": {
          "toolchain": "diascld00"
        }
      }
    }
  ],
  "buildPresets": [
    {
      "name": "Proteus - User - Build",
      "inherits": "Proteus Build",
      "configurePreset": "Proteus - User"
    }
  ]
}

```
You should *NOT* commit this file, to avoid conflicts with other users (it's already in our gitignore).
Furthermore, you should not depend on any user-specific settings to run/compile Proteus and any time you think something is broken, you should first verify that any user-specific settings in that file do not cause the issue.


# Testing
For our c++ code we use [GTest](https://github.com/google/googletest) for our functional testing and [googlebenchmark](https://github.com/google/benchmark) for performance testing.
cpp integration tests live in `tests`. cpp Unit tests are in `tests` subdirectories in the component they test, and likewise performance tests are in `perftests` subdirectories.
The planner uses junit. 