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

Editing CMake configurations with user-specific settings
========
Sometimes you want to invoke `cmake` with different flags or have multiple profiles.
CMake allows that through a `CMakeUserPresets.json` file that you can add locally with user-specific settings.
For example, if you want to produce verbose makefiles, you can add in the project root the following `CMakeUserPresets.json` file:
```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "Proteus - User",
      "inherits": "Proteus",
      "cacheVariables": {
        "CMAKE_VERBOSE_MAKEFILE": "ON"
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
