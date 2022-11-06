# Testing
For our c++ code we use [GTest](https://github.com/google/googletest) for our functional testing and [googlebenchmark](https://github.com/google/benchmark) for performance testing.
cpp integration tests live in `tests`. cpp Unit tests are in `tests` subdirectories in the component they test, and likewise performance tests are in `perftests` subdirectories.
The planner uses junit. 

## CI
Internal to DIAS, we use GitLab for CI, see `.gitlab-ci.yml` for how tests are invoked.
Unfortunately, we cannot use GitHub Actions for CI as we have to use self-hosted runners due to the hardware requirements of Proteus.