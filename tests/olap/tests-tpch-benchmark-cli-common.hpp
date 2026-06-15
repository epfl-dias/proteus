#ifndef PROTEUS_TESTS_TPCH_BENCHMARK_CLI_COMMON_HPP
#define PROTEUS_TESTS_TPCH_BENCHMARK_CLI_COMMON_HPP

#include "gtest/gtest.h"

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace
{
struct TpchBenchmarkCase
{
    std::string compression_type;
    std::filesystem::path dataset_path;
    int query;
    std::optional<int> tile_size;
    bool allow_post_result_abort;
};

struct ProcessResult
{
    int exit_code = -1;
    std::string output;
};

inline std::optional<std::string> FindTpchBenchmarkBinary()
{
    if (const char* env = std::getenv("PROTEUS_BENCHMARK_TPCH_BIN"))
    {
        std::filesystem::path bin(env);
        if (std::filesystem::exists(bin))
        {
            return bin.string();
        }
    }

    const std::filesystem::path cwd = std::filesystem::current_path();
    const std::vector<std::filesystem::path> candidates = {
        cwd / "proteus-benchmark-tpch",
        cwd / "apps" / "benchmarks" / "olap" / "benchmark-tpch" /
            "proteus-benchmark-tpch",
        cwd.parent_path() / "apps" / "benchmarks" / "olap" /
            "benchmark-tpch" / "proteus-benchmark-tpch",
        cwd.parent_path().parent_path() / "apps" / "benchmarks" / "olap" /
            "benchmark-tpch" / "proteus-benchmark-tpch",
    };

    for (const auto& candidate : candidates)
    {
        if (std::filesystem::exists(candidate))
        {
            return candidate.string();
        }
    }
    return std::nullopt;
}

inline bool HasNvidiaGpu()
{
    std::error_code ec;
    const std::filesystem::path nvidia_gpus_dir = "/proc/driver/nvidia/gpus";
    if (std::filesystem::exists(nvidia_gpus_dir, ec) &&
        std::filesystem::is_directory(nvidia_gpus_dir, ec))
    {
        for (const auto& entry :
             std::filesystem::directory_iterator(nvidia_gpus_dir, ec))
        {
            if (entry.is_directory(ec))
            {
                return true;
            }
        }
    }
    return std::filesystem::exists("/dev/nvidia0", ec) ||
        std::filesystem::exists("/dev/nvidiactl", ec);
}

inline std::string QueryName(const int query)
{
    return "Q" + std::to_string(query);
}

inline std::vector<std::string> BuildTpchArgs(const TpchBenchmarkCase& test_case)
{
    std::vector<std::string> args = {
        "--cpu_buffers=0.01",
        "--print_generated_code=false",
        "--benchmark_format=console",
        "--compression_type=" + test_case.compression_type,
        "--benchmark_filter=TPCHGPUOnly/GPU_" + QueryName(test_case.query),
    };
    if (test_case.compression_type != "NONE")
    {
        args.emplace_back("--decom_op=true");
        args.emplace_back("--tile_size=" +
                          std::to_string(test_case.tile_size.value_or(512)));
    }
    return args;
}

inline ProcessResult RunProcessCapture(const std::string& binary,
                                       const std::vector<std::string>& args)
{
    std::vector<char*> argv;
    argv.reserve(args.size() + 2);
    argv.push_back(const_cast<char*>(binary.c_str()));
    for (const auto& arg : args)
    {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    int pipe_fds[2];
    if (pipe(pipe_fds) == -1)
    {
        return {};
    }

    const pid_t pid = fork();
    if (pid == -1)
    {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return {};
    }

    if (pid == 0)
    {
        close(pipe_fds[0]);
        dup2(pipe_fds[1], STDOUT_FILENO);
        dup2(pipe_fds[1], STDERR_FILENO);
        close(pipe_fds[1]);

        if (std::getenv("CUDA_VISIBLE_DEVICES") == nullptr)
        {
            setenv("CUDA_VISIBLE_DEVICES", "0", 1);
        }
        execv(binary.c_str(), argv.data());
        _exit(127);
    }

    close(pipe_fds[1]);
    std::string output;
    char buf[4096];
    while (true)
    {
        const ssize_t read_bytes = read(pipe_fds[0], buf, sizeof(buf));
        if (read_bytes > 0)
        {
            output.append(buf, static_cast<size_t>(read_bytes));
            continue;
        }
        if (read_bytes == 0)
        {
            break;
        }
        if (errno == EINTR)
        {
            continue;
        }
        break;
    }
    close(pipe_fds[0]);

    int status = 0;
    ProcessResult res;
    res.output = std::move(output);
    if (waitpid(pid, &status, 0) == -1)
    {
        return res;
    }
    if (WIFEXITED(status))
    {
        res.exit_code = WEXITSTATUS(status);
    }
    else if (WIFSIGNALED(status))
    {
        res.exit_code = 128 + WTERMSIG(status);
    }
    else
    {
        res.exit_code = status;
    }
    return res;
}

inline std::string BuildCommand(const std::string& binary,
                                const std::vector<std::string>& args)
{
    std::string command = binary;
    for (const auto& arg : args)
    {
        command += " ";
        command += arg;
    }
    return command;
}

inline void PointTpchInputAt(const std::filesystem::path& dataset_path)
{
    std::error_code ec;
    std::filesystem::create_directories("inputs", ec);
    ASSERT_FALSE(ec) << "Failed to create inputs directory: " << ec.message();

    const std::filesystem::path tpch_link = "inputs/tpch100";
    if (std::filesystem::exists(tpch_link, ec) ||
        std::filesystem::is_symlink(tpch_link, ec))
    {
        ASSERT_TRUE(std::filesystem::is_symlink(tpch_link, ec))
        << tpch_link << " exists and is not a symlink";
        std::filesystem::remove(tpch_link, ec);
        ASSERT_FALSE(ec) << "Failed to remove " << tpch_link << ": "
                         << ec.message();
    }

    std::filesystem::create_directory_symlink(dataset_path, tpch_link, ec);
    ASSERT_FALSE(ec) << "Failed to point " << tpch_link << " at "
                     << dataset_path << ": " << ec.message();
}

inline std::string TpchBenchmarkLineNeedle(const int query)
{
    return "TPCHGPUOnly/GPU_" + QueryName(query) + "/100/1/iterations";
}

inline std::string TpchCaseName(const testing::TestParamInfo<int>& info)
{
    return "q" + std::to_string(info.param);
}
} // namespace

class TpchBenchmarkCliBase : public ::testing::TestWithParam<int>
{
protected:
    void RunAndCheckCompletes(TpchBenchmarkCase test_case)
    {
        ASSERT_TRUE(HasNvidiaGpu()) << "No NVIDIA GPU detected";
        const auto binary = FindTpchBenchmarkBinary();
        ASSERT_TRUE(binary.has_value())
        << "proteus-benchmark-tpch binary not found; set "
           "PROTEUS_BENCHMARK_TPCH_BIN";
        ASSERT_TRUE(std::filesystem::exists(test_case.dataset_path))
        << "Missing TPC-H dataset path: " << test_case.dataset_path;

        PointTpchInputAt(test_case.dataset_path);

        const auto args = BuildTpchArgs(test_case);
        const auto run = RunProcessCapture(*binary, args);
        ASSERT_NE(run.exit_code, -1)
        << "Failed to launch: " << BuildCommand(*binary, args);

        const auto benchmark_line = TpchBenchmarkLineNeedle(test_case.query);
        ASSERT_NE(run.output.find(benchmark_line), std::string::npos)
        << "Benchmark did not complete query " << QueryName(test_case.query)
        << " for " << test_case.compression_type << "\nCommand: "
        << BuildCommand(*binary, args) << "\n=== Captured output begin ===\n"
        << run.output << "\n=== Captured output end ===";

        if (!test_case.allow_post_result_abort)
        {
            ASSERT_EQ(run.exit_code, 0)
            << "Non-zero exit code from: " << BuildCommand(*binary, args)
            << "\n=== Captured output begin ===\n"
            << run.output << "\n=== Captured output end ===";
        }
    }
};

inline const std::vector<int>& BuildTpchQueries()
{
    static const std::vector<int> queries = {1, 3, 5, 6};
    return queries;
}

#endif  // PROTEUS_TESTS_TPCH_BENCHMARK_CLI_COMMON_HPP
