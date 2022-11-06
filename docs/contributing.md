# Reporting bugs
Please use the [GitHub issues tab](https://github.com/epfl-dias/proteus/issues) to report bugs.

# Submitting changes
Please open a PR on the GitHub repo. 
Before putting in the effort to make changes, please open an issue to discuss the proposed changes.
This minimizes the chances of a PR hitting an unexpected blocker.
Please also follow our git conventions, see the git section below.

# Building
Proteus is a conventional CMake project. For details on required dependencies and building see [building.md](building.md).

# Runtime Environment
## Hugepages
Proteus requires Linux hugepages. To allocate hugepages:
```sh
echo 76800 | sudo tee /sys/devices/system/node/node{0,1}/hugepages/hugepages-2048kB/nr_hugepages
```
You may need to vary the number of huge pages based on your system's memory.
We recommend using 80-90% of system memory if Proteus is the only resource intensive process running on the server (e.g. for benchmarking).
You may also need to change `node{0,1}` based on the number of NUMA nodes in your system. 


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

