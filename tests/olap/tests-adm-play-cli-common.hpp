#ifndef PROTEUS_TESTS_ADM_PLAY_CLI_COMMON_HPP
#define PROTEUS_TESTS_ADM_PLAY_CLI_COMMON_HPP

#include "gtest/gtest.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gflags/gflags.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

DECLARE_int32(tile_size);

namespace
{
    // Toggle compression families and SF1000 group without commenting out
    // instantiations. `false` means the related suite gets zero parameters.
    const bool test_none = true;
    const bool test_rfor = true;
    const bool test_rfor_decomp = true;
    const bool test_rfor_decomp_reg = true;
    const bool test_rfor_pre_decomp = true;
    const bool test_flbp = true;
    const bool test_flbp_decomp = true;
    const bool test_flbp_decomp_reg = true;
    const bool test_flbp_pre_decomp = true;
    const bool test_one_kernel = true;
    const bool test_1000 = false;


    struct AdmPlayCase
    {
        int scale_factor;
        int num_drives;
        std::string compression_type;
        std::string bench_flag;
        bool in_gpu;
        int query;
        bool decom_op;
        bool decom_reg;
        bool precompute_rl;
        bool pre_decom;
        std::optional<bool> one_kernel;
        std::optional<int> tile_size;
        std::optional<int> threads_per_block;
        std::optional<int> items_per_thread;
        std::string label;
        bool sorted = false;
    };

    struct ProcessResult
    {
        int exit_code = -1;
        std::string output;
    };

    inline std::optional<std::string> FindAdmPlayBinary()
    {
        if (const char* env = std::getenv("PROTEUSADM_PLAY_BIN"))
        {
            std::filesystem::path bin(env);
            if (std::filesystem::exists(bin))
            {
                return bin.string();
            }
        }

        const std::filesystem::path cwd = std::filesystem::current_path();
        const std::vector<std::filesystem::path> candidates = {
            cwd / "proteusadm-play",
            cwd / "apps" / "standalones" / "adm" / "adm-play" / "proteusadm-play",
            cwd.parent_path() / "apps" / "standalones" / "adm" / "adm-play" /
            "proteusadm-play",
            cwd.parent_path().parent_path() / "apps" / "standalones" / "adm" /
            "adm-play" / "proteusadm-play",
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

    inline int EffectiveTileSize(const AdmPlayCase& test_case)
    {
        return test_case.tile_size.value_or(FLAGS_tile_size);
    }

    inline std::vector<std::string> MissingPaths(
        const std::vector<std::string>& paths)
    {
        std::vector<std::string> missing;
        std::error_code ec;
        for (const auto& path : paths)
        {
            if (!std::filesystem::exists(path, ec))
            {
                missing.push_back(path);
            }
        }
        return missing;
    }

    inline std::string JoinPaths(const std::vector<std::string>& paths)
    {
        std::string joined;
        for (size_t i = 0; i < paths.size(); ++i)
        {
            if (i > 0)
            {
                joined += ", ";
            }
            joined += paths[i];
        }
        return joined;
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

    inline std::optional<std::vector<std::string>> RequiredPaths(
        const AdmPlayCase& test_case)
    {
        if ((test_case.decom_op || test_case.decom_reg || test_case.pre_decom) &&
            test_case.compression_type != "RFOR" &&
            test_case.compression_type != "FLBP")
        {
            return std::nullopt;
        }

        if (test_case.pre_decom && !test_case.in_gpu)
        {
            return std::nullopt;
        }

        if (test_case.decom_reg && !test_case.in_gpu)
        {
            return std::nullopt;
        }

        if (test_case.precompute_rl && !test_case.decom_reg)
        {
            return std::nullopt;
        }

        if (test_case.precompute_rl && test_case.compression_type != "RFOR")
        {
            return std::nullopt;
        }

        if (test_case.compression_type == "NONE")
        {
            if (test_case.scale_factor == 100 && test_case.num_drives == 1)
            {
                return std::vector<std::string>{
                    "/nvme31/ssbm100/uncompressed",
                };
            }
            if (test_case.scale_factor == 1000 && test_case.num_drives == 1)
            {
                return std::vector<std::string>{
                    "/nvme16/nicholso/data/sbm1000",
                };
            }
            if (test_case.scale_factor == 1000 && test_case.num_drives == 2)
            {
                return std::vector<std::string>{
                    "/nvme18/nicholso/data/sbm1000_0_2",
                    "/nvme23/nicholso/data/sbm1000_1_2",
                };
            }
            if (test_case.scale_factor == 1000 && test_case.num_drives == 4)
            {
                return std::vector<std::string>{
                    "/nvme25/nicholso/data/sbm1000_0_4",
                    "/nvme18/nicholso/data/sbm1000_1_4",
                    "/nvme28/nicholso/data/sbm1000_2_4",
                    "/nvme22/nicholso/data/sbm1000_3_4",
                };
            }
            if (test_case.scale_factor == 1000 && test_case.num_drives == 8)
            {
                return std::vector<std::string>{
                    "/nvme16/nicholso/data/sbm1000_0_8",
                    "/nvme19/nicholso/data/sbm1000_1_8",
                    "/nvme28/nicholso/data/sbm1000_2_8",
                    "/nvme27/nicholso/data/sbm1000_3_8",
                    "/nvme28/nicholso/data/sbm1000_4_8",
                    "/nvme20/nicholso/data/sbm1000_5_8",
                    "/nvme21/nicholso/data/sbm1000_6_8",
                    "/nvme24/nicholso/data/sbm1000_7_8",
                };
            }
        }

        if (test_case.compression_type == "RFOR")
        {
            const auto effective_tile_size = EffectiveTileSize(test_case);
            if (test_case.scale_factor == 100 && test_case.num_drives == 1)
            {
                if (effective_tile_size != 512 && effective_tile_size != 1024)
                {
                    throw std::runtime_error(
                        "--tile_size must be 512 or 1024 for RFOR tests");
                }
                if (test_case.sorted)
                {
                    return std::vector<std::string>{
                        (effective_tile_size == 1024)
                            ? (test_case.decom_reg
                                   ? "/nvme29/boffa/data/compressed_rfor_ssbm100_sorted_decom_reg"
                                   : "/nvme29/boffa/data/compressed_rfor_ssbm100_sorted_decom_op")
                            : (test_case.decom_reg
                                   ? "/nvme29/boffa/data/compressed_rfor_ssbm100_sorted_decom_reg_512"
                                   : "/nvme29/boffa/data/compressed_rfor_ssbm100_sorted_decom_op_512"),
                    };
                }
                if (test_case.in_gpu)
                {
                    const auto rfor_tile_size = effective_tile_size;
                    return std::vector<std::string>{
                        (rfor_tile_size == 1024)
                            ? (test_case.decom_reg
                                   ? "/nvme31/ssbm100/compressed_rfor_ssbm100_decom_reg"
                                   : "/nvme31/ssbm100/compressed_rfor_ssbm100_decom_op")
                            : (test_case.decom_reg
                                   ? "/nvme31/ssbm100/compressed_rfor_ssbm100_decom_reg_512"
                                   : "/nvme31/ssbm100/compressed_rfor_ssbm100_decom_op_512"),
                    };
                }
                if (test_case.decom_op || test_case.pre_decom || test_case.decom_reg)
                {
                    return std::vector<std::string>{
                        (effective_tile_size == 1024)
                            ? (test_case.decom_reg
                                   ? "/nvme31/ssbm100/compressed_rfor_ssbm100_decom_reg"
                                   : "/nvme31/ssbm100/compressed_rfor_ssbm100_decom_op")
                            : (test_case.decom_reg
                                   ? "/nvme31/ssbm100/compressed_rfor_ssbm100_decom_reg_512"
                                   : "/nvme31/ssbm100/compressed_rfor_ssbm100_decom_op_512"),
                    };
                }
                if (effective_tile_size != 1024)
                {
                    throw std::runtime_error(
                        "No SF100 non-metadata RFOR dataset configured for "
                        "tile_size=" + std::to_string(effective_tile_size));
                }
                return std::vector<std::string>{
                    "/nvme31/ssbm100/compressed_rfor_ssbm100",
                };
            }
            if (test_case.scale_factor == 1000 && test_case.num_drives == 1)
            {
                if (test_case.decom_reg)
                {
                    return std::nullopt;
                }
                return std::vector<std::string>{
                    (test_case.decom_op || test_case.pre_decom)
                        ? "/nvme25/boffa/data/compressed_rfor_ssbm1000_decom_op"
                        : "/nvme25/duric/data/compressed_rfor_ssbm1000",
                };
            }
            if (test_case.scale_factor == 1000 && test_case.num_drives == 8)
            {
                if (test_case.decom_reg)
                {
                    return std::nullopt;
                }
                if (test_case.decom_op || test_case.pre_decom)
                {
                    return std::vector<std::string>{
                        "/nvme16/boffa/data/compressed_rfor_ssbm1000_decom_op_0_8",
                        "/nvme22/boffa/data/compressed_rfor_ssbm1000_decom_op_1_8",
                        "/nvme26/boffa/data/compressed_rfor_ssbm1000_decom_op_2_8",
                        "/nvme29/boffa/data/compressed_rfor_ssbm1000_decom_op_3_8",
                        "/nvme18/boffa/data/compressed_rfor_ssbm1000_decom_op_4_8",
                        "/nvme21/boffa/data/compressed_rfor_ssbm1000_decom_op_5_8",
                        "/nvme28/boffa/data/compressed_rfor_ssbm1000_decom_op_6_8",
                        "/nvme27/boffa/data/compressed_rfor_ssbm1000_decom_op_7_8",
                    };
                }
                return std::vector<std::string>{
                    "/nvme16/boffa/data/compressed_rfor_ssbm1000_0_8",
                    "/nvme22/boffa/data/compressed_rfor_ssbm1000_1_8",
                    "/nvme26/boffa/data/compressed_rfor_ssbm1000_2_8",
                    "/nvme29/boffa/data/compressed_rfor_ssbm1000_3_8",
                    "/nvme18/boffa/data/compressed_rfor_ssbm1000_4_8",
                    "/nvme21/boffa/data/compressed_rfor_ssbm1000_5_8",
                    "/nvme28/boffa/data/compressed_rfor_ssbm1000_6_8",
                    "/nvme27/boffa/data/compressed_rfor_ssbm1000_7_8",
                };
            }
        }

        if (test_case.compression_type == "FLBP")
        {
            if (test_case.scale_factor == 100 && test_case.num_drives == 1)
            {
                return std::vector<std::string>{
                    "/nvme31/ssbm100/compressed_flbp_ssbm100_decom_op",
                };
            }
            return std::nullopt;
        }

        return std::nullopt;
    }

    inline std::vector<std::string> BuildArgs(const AdmPlayCase& test_case)
    {
        std::vector<std::string> args = {
            test_case.bench_flag,
            "--scale_factor=" + std::to_string(test_case.scale_factor),
            "--num_iterations=1",
            "--server_number=49",
            "--num_drives=" + std::to_string(test_case.num_drives),
            "--query=" + std::to_string(test_case.query),
            "--compression_type=" + test_case.compression_type,
            "--tile_size=" + std::to_string(EffectiveTileSize(test_case)),
            "--warm_up=false",
            "--print_res=true",
            "--emit_tuple_oid_bindings=true",
        };
        if (test_case.threads_per_block.has_value())
        {
            args.emplace_back("--threads_per_block=" +
                std::to_string(*test_case.threads_per_block));
        }
        if (test_case.items_per_thread.has_value())
        {
            args.emplace_back("--items_per_thread=" +
                std::to_string(*test_case.items_per_thread));
        }
        if (test_case.in_gpu)
        {
            args.emplace_back("--in_gpu");
        }
        if (test_case.decom_op)
        {
            args.emplace_back("--decom_op");
        }
        if (test_case.decom_reg)
        {
            args.emplace_back("--decom_reg");
        }
        if (test_case.precompute_rl)
        {
            args.emplace_back("--precompute_rl");
        }
        if (test_case.pre_decom)
        {
            args.emplace_back("--pre_decom");
        }
        if (test_case.one_kernel.has_value())
        {
            args.emplace_back(std::string("--one_kernel=") +
                (*test_case.one_kernel ? "true" : "false"));
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

            setenv("CUDA_VISIBLE_DEVICES", "1", 1);
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
        if (waitpid(pid, &status, 0) == -1)
        {
            ProcessResult res;
            res.exit_code = -1;
            res.output = std::move(output);
            return res;
        }

        ProcessResult res;
        res.output = std::move(output);
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

    inline std::string Trim(std::string_view s)
    {
        size_t begin = 0;
        while (begin < s.size() &&
            std::isspace(static_cast<unsigned char>(s[begin])))
        {
            ++begin;
        }
        size_t end = s.size();
        while (end > begin &&
            std::isspace(static_cast<unsigned char>(s[end - 1])))
        {
            --end;
        }
        return std::string(s.substr(begin, end - begin));
    }

    inline bool IsTimestampLogLine(std::string_view line)
    {
        if (line.size() < 10)
        {
            return false;
        }
        const char severity = line[0];
        if (severity != 'I' && severity != 'W' && severity != 'E' &&
            severity != 'F')
        {
            return false;
        }
        for (size_t i = 1; i <= 8; ++i)
        {
            if (!std::isdigit(static_cast<unsigned char>(line[i])))
            {
                return false;
            }
        }
        return std::isspace(static_cast<unsigned char>(line[9]));
    }

    inline bool IsPostResultNoiseLine(std::string_view line)
    {
        if (line.empty())
        {
            return false;
        }
        if (IsTimestampLogLine(line))
        {
            return true;
        }
        if (line.rfind("query,", 0) == 0)
        {
            return true;
        }
        if (line.rfind("Process finished with exit code", 0) == 0)
        {
            return true;
        }
        if (line.rfind("Catalog cleared!", 0) == 0)
        {
            return true;
        }
        if (line.rfind("=======================================================>", 0) == 0 ||
            line.rfind("------------------------------------------------------->", 0) == 0)
        {
            return true;
        }
        return false;
    }

    inline bool IsInlineNoiseLine(std::string_view line)
    {
        if (IsPostResultNoiseLine(line))
        {
            return true;
        }
        if (line.rfind("warning:", 0) == 0 || line.rfind("remark:", 0) == 0 ||
            line.rfind("ptxas", 0) == 0)
        {
            return true;
        }
        return false;
    }

    inline std::vector<std::string> ExtractAllQueryResults(const std::string& output)
    {
        static constexpr std::string_view kBeginMarker = "<QUERY RESULT BEGIN>";
        static constexpr std::string_view kBeginMarkerLegacy = "QUERY RESULT BEGIN";
        static constexpr std::string_view kEndMarker = "<END QUERY RESULT>";

        std::istringstream marker_stream(output);
        std::string line;
        std::vector<std::string> marker_results;
        std::string current_result;
        bool capturing = false;

        while (std::getline(marker_stream, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }

            if (!capturing)
            {
                if (line.find(kBeginMarker) != std::string::npos ||
                    line.find(kBeginMarkerLegacy) != std::string::npos)
                {
                    capturing = true;
                    current_result.clear();
                }
                continue;
            }

            if (line.find(kEndMarker) != std::string::npos)
            {
                marker_results.push_back(Trim(current_result));
                capturing = false;
                current_result.clear();
                continue;
            }

            if (IsInlineNoiseLine(line))
            {
                continue;
            }

            if (!current_result.empty())
            {
                current_result.push_back('\n');
            }
            current_result += line;
        }

        if (!marker_results.empty())
        {
            return marker_results;
        }

        static constexpr std::string_view kLegacyMarker = "QUERY RESULT:";
        std::istringstream stream(output);
        current_result.clear();
        std::vector<std::string> all_results;
        capturing = false;

        while (std::getline(stream, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }

            const size_t marker_pos = line.find(kLegacyMarker);
            if (marker_pos != std::string::npos)
            {
                if (capturing && !current_result.empty())
                {
                    all_results.push_back(current_result);
                }
                current_result = Trim(
                    std::string_view(line).substr(marker_pos + kLegacyMarker.size()));
                capturing = true;
                continue;
            }

            if (!capturing)
            {
                continue;
            }

            if (IsPostResultNoiseLine(line))
            {
                if (!current_result.empty())
                {
                    all_results.push_back(current_result);
                }
                current_result.clear();
                capturing = false;
                continue;
            }

            if (line.empty())
            {
                continue;
            }

            if (!current_result.empty())
            {
                current_result.push_back('\n');
            }
            current_result += Trim(line);
        }

        if (capturing && !current_result.empty())
        {
            all_results.push_back(current_result);
        }

        return all_results;
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

    inline std::string ConfigName(const AdmPlayCase& test_case)
    {
        std::string name = "sf" + std::to_string(test_case.scale_factor);
        name += "_q" + std::to_string(test_case.query);
        if (!test_case.label.empty())
        {
            name += "_" + test_case.label;
        }
        else
        {
            name += "_" + test_case.compression_type;
            if (test_case.in_gpu)
            {
                name += "_ingpu";
            }
            else
            {
                name += "_nvmegpu";
            }
            if (test_case.decom_op)
            {
                name += "_decomop";
            }
            else
            {
                name += "_nodecomop";
            }
            if (test_case.decom_reg)
            {
                name += "_decomreg";
            }
            if (test_case.precompute_rl)
            {
                name += "_precomputerl";
            }
            if (test_case.pre_decom)
            {
                name += "_predecom";
            }
            if (test_case.one_kernel.has_value())
            {
                name += (*test_case.one_kernel) ? "_onekernel1" : "_onekernel0";
            }
            if (test_case.tile_size.has_value())
            {
                name += "_t" + std::to_string(*test_case.tile_size);
            }
            if (test_case.threads_per_block.has_value() &&
                test_case.items_per_thread.has_value())
            {
                name += "_" + std::to_string(*test_case.threads_per_block) +
                    "x" + std::to_string(*test_case.items_per_thread);
            }
        }
        return name;
    }


    inline const std::vector<int>& BuildAllQueries()
    {
        // static const std::vector<int> queries = {
        //     11, 12, 13, 21, 22, 23, 31, 32, 33, 34, 41, 42, 43
        // };
        // static const std::vector<int> queries = {
        //   11, 12, 13};
        // static const std::vector<int> queries = {
        //   21, 22, 23};
        // static const std::vector<int> queries = {
        //   31, 32, 33, 34};
        // static const std::vector<int> queries = {
        //   41, 42, 43};
        static const std::vector queries = {
         11, 21, 31, 41};
        return queries;
    }

    inline const std::vector<int>& BuildNoQueries()
    {
        static const std::vector<int> queries = {};
        return queries;
    }

    inline const std::vector<int>& BuildQueriesIfEnabled(bool enabled)
    {
        return enabled ? BuildAllQueries() : BuildNoQueries();
    }

    inline const std::vector<int>& BuildSf1000QueriesIfEnabled(
        bool compression_family_enabled)
    {
        return (test_1000 && compression_family_enabled)
                   ? BuildAllQueries()
                   : BuildNoQueries();
    }

    inline AdmPlayCase BuildCase(int scale_factor, int query,
                                 const std::string& compression_type, bool in_gpu,
                                 bool decom_op, bool pre_decom = false,
                                 std::optional<bool> one_kernel = std::nullopt,
                                 bool decom_reg = false,
                                 bool precompute_rl = false,
                                 std::optional<int> tile_size = std::nullopt,
                                 std::optional<int> threads_per_block = std::nullopt,
                                 std::optional<int> items_per_thread = std::nullopt,
                                 std::string label = {})
    {
        return {
            scale_factor, 1, compression_type, "--bench_varybw_gpu_ssb",
            in_gpu, query, decom_op, decom_reg, precompute_rl, pre_decom,
            one_kernel, tile_size, threads_per_block, items_per_thread,
            std::move(label)
        };
    }

    inline AdmPlayCase BuildSortedCase(int scale_factor, int query,
                                       const std::string& compression_type,
                                       bool in_gpu, bool decom_op,
                                       bool pre_decom = false,
                                       std::optional<bool> one_kernel = std::nullopt,
                                       bool decom_reg = false,
                                       bool precompute_rl = false,
                                       std::optional<int> tile_size = std::nullopt)
    {
        auto c = BuildCase(scale_factor, query, compression_type, in_gpu,
                           decom_op, pre_decom, one_kernel, decom_reg,
                           precompute_rl, tile_size);
        c.sorted = true;
        return c;
    }

    inline AdmPlayCase BuildDecomRegCase(int scale_factor, int query,
                                         std::optional<bool> one_kernel =
                                             std::nullopt)
    {
        return BuildCase(scale_factor, query, "RFOR", true, false, false,
                         one_kernel, true, true);
    }

    inline AdmPlayCase BuildFlbpDecomRegCase(int scale_factor, int query,
                                             std::optional<bool> one_kernel =
                                                 std::nullopt)
    {
        return BuildCase(scale_factor, query, "FLBP", true, false, false,
                         one_kernel, true, false);
    }

    inline void AppendCasesForQueries(std::vector<AdmPlayCase>& cases, bool enabled,
                                      const std::string& label,
                                      const std::string& compression_type,
                                      bool decom_op, bool pre_decom = false,
                                      bool decom_reg = false,
                                      bool precompute_rl = false,
                                      std::optional<int> tile_size = std::nullopt,
                                      std::optional<int> threads_per_block =
                                          std::nullopt,
                                      std::optional<int> items_per_thread =
                                          std::nullopt)
    {
        if (!enabled)
        {
            return;
        }
        for (const int query : BuildAllQueries())
        {
            cases.emplace_back(BuildCase(
                100, query, compression_type, true, decom_op, pre_decom,
                std::nullopt, decom_reg, precompute_rl, tile_size,
                threads_per_block, items_per_thread, label));
        }
    }

    // Paper-config case builders, split per codec so each per-codec test binary
    // only instantiates its own cases. The original BuildPaperConfigCases() is
    // the union of these three.

    inline std::vector<AdmPlayCase> BuildPaperConfigCasesNone()
    {
        std::vector<AdmPlayCase> cases;
        AppendCasesForQueries(cases, test_none, "NONE", "NONE", false);
        return cases;
    }

    inline std::vector<AdmPlayCase> BuildPaperConfigCasesRfor()
    {
        std::vector<AdmPlayCase> cases;
        AppendCasesForQueries(cases, test_rfor_decomp,
                              "RFOR_decom_op_t512_64x8", "RFOR", true, false,
                              false, false, 512, 64, 8);
        AppendCasesForQueries(cases, test_rfor_decomp,
                              "RFOR_decom_op_t512_128x4", "RFOR", true, false,
                              false, false, 512, 128, 4);
        AppendCasesForQueries(cases, test_rfor_decomp,
                              "RFOR_decom_op_t512_256x2", "RFOR", true, false,
                              false, false, 512, 256, 2);
        AppendCasesForQueries(cases, test_rfor_decomp,
                              "RFOR_decom_op_t1024_512x2", "RFOR", true,
                              false, false, false, 1024, 512, 2);
        AppendCasesForQueries(cases, test_rfor_decomp,
                              "RFOR_decom_op_t1024_256x4", "RFOR", true,
                              false, false, false, 1024, 256, 4);
        AppendCasesForQueries(cases, test_rfor_decomp,
                              "RFOR_decom_op_t1024_128x8", "RFOR", true,
                              false, false, false, 1024, 128, 8);
        AppendCasesForQueries(cases, test_rfor_decomp_reg,
                              "RFOR_decom_reg_t512_64x8", "RFOR", false,
                              false, true, true, 512, 64, 8);
        AppendCasesForQueries(cases, test_rfor_decomp_reg,
                              "RFOR_decom_reg_t512_128x4", "RFOR", false,
                              false, true, true, 512, 128, 4);
        AppendCasesForQueries(cases, test_rfor_decomp_reg,
                              "RFOR_decom_reg_t512_256x2", "RFOR", false,
                              false, true, true, 512, 256, 2);
        AppendCasesForQueries(cases, test_rfor_decomp_reg,
                              "RFOR_decom_reg_t1024_512x2", "RFOR", false,
                              false, true, true, 1024, 512, 2);
        AppendCasesForQueries(cases, test_rfor_decomp_reg,
                              "RFOR_decom_reg_t1024_256x4", "RFOR", false,
                              false, true, true, 1024, 256, 4);
        AppendCasesForQueries(cases, test_rfor_decomp_reg,
                              "RFOR_decom_reg_t1024_128x8", "RFOR", false,
                              false, true, true, 1024, 128, 8);
        AppendCasesForQueries(cases, test_rfor_pre_decomp,
                              "RFOR_pre_decom_t512_64x8", "RFOR", false, true,
                              false, false, 512, 64, 8);
        AppendCasesForQueries(cases, test_rfor_pre_decomp,
                              "RFOR_pre_decom_t512_128x4", "RFOR", false, true,
                              false, false, 512, 128, 4);
        AppendCasesForQueries(cases, test_rfor_pre_decomp,
                              "RFOR_pre_decom_t512_256x2", "RFOR", false, true,
                              false, false, 512, 256, 2);
        AppendCasesForQueries(cases, test_rfor_pre_decomp,
                              "RFOR_pre_decom_t1024_512x2", "RFOR", false,
                              true, false, false, 1024, 512, 2);
        AppendCasesForQueries(cases, test_rfor_pre_decomp,
                              "RFOR_pre_decom_t1024_256x4", "RFOR", false,
                              true, false, false, 1024, 256, 4);
        AppendCasesForQueries(cases, test_rfor_pre_decomp,
                              "RFOR_pre_decom_t1024_128x8", "RFOR", false,
                              true, false, false, 1024, 128, 8);
        return cases;
    }

    inline std::vector<AdmPlayCase> BuildPaperConfigCasesFlbp()
    {
        std::vector<AdmPlayCase> cases;
        AppendCasesForQueries(cases, test_flbp_decomp,
                              "FLBP_decom_op_t1024_512x2", "FLBP", true,
                              false, false, false, 1024, 512, 2);
        AppendCasesForQueries(cases, test_flbp_decomp,
                              "FLBP_decom_op_t1024_256x4", "FLBP", true, false,
                              false, false, 1024, 256, 4);
        AppendCasesForQueries(cases, test_flbp_decomp,
                              "FLBP_decom_op_t1024_128x8", "FLBP", true, false,
                              false, false, 1024, 128, 8);
        AppendCasesForQueries(cases, test_flbp_decomp,
                              "FLBP_decom_op_t1024_64x16", "FLBP", true, false,
                              false, false, 1024, 64, 16);
        AppendCasesForQueries(cases, test_flbp_decomp_reg,
                              "FLBP_decom_reg_t1024_512x2", "FLBP", false,
                              false, true, false, 1024, 512, 2);
        AppendCasesForQueries(cases, test_flbp_decomp_reg,
                              "FLBP_decom_reg_t1024_256x4", "FLBP", false,
                              false, true, false, 1024, 256, 4);
        AppendCasesForQueries(cases, test_flbp_decomp_reg,
                              "FLBP_decom_reg_t1024_128x8", "FLBP", false,
                              false, true, false, 1024, 128, 8);
        AppendCasesForQueries(cases, test_flbp_decomp_reg,
                              "FLBP_decom_reg_t1024_64x16", "FLBP", false,
                              false, true, false, 1024, 64, 16);
        AppendCasesForQueries(cases, test_flbp_pre_decomp,
                              "FLBP_pre_decom_t1024_512x2", "FLBP", false,
                              true, false, false, 1024, 512, 2);
        AppendCasesForQueries(cases, test_flbp_pre_decomp,
                              "FLBP_pre_decom_t1024_256x4", "FLBP", false,
                              true, false, false, 1024, 256, 4);
        AppendCasesForQueries(cases, test_flbp_pre_decomp,
                              "FLBP_pre_decom_t1024_128x8", "FLBP", false,
                              true, false, false, 1024, 128, 8);
        AppendCasesForQueries(cases, test_flbp_pre_decomp,
                              "FLBP_pre_decom_t1024_64x16", "FLBP", false,
                              true, false, false, 1024, 64, 16);
        return cases;
    }

    inline std::filesystem::path GoldenOutputPathForQuery(int query)
    {
        const int family = query / 10;
        const int variant = query % 10;
        return "ssb_q" + std::to_string(family) + "_" + std::to_string(variant) +
            "_par_cpy_plan";
    }

    inline std::vector<std::filesystem::path> GoldenOutputCandidatesForQuery(int query)
    {
        const auto file_name = GoldenOutputPathForQuery(query);
        const std::filesystem::path source_tree_path =
            std::filesystem::path(__FILE__).parent_path() / "outputs" /
            "tests-gpu-ssb" / file_name;
        return {
            std::filesystem::path(TEST_OUTPUTS) / "tests-gpu-ssb" / file_name,
            source_tree_path,
            std::filesystem::path("tests/olap/outputs/tests-gpu-ssb") / file_name,
        };
    }

    inline std::string JoinFsPaths(const std::vector<std::filesystem::path>& paths)
    {
        std::vector<std::string> as_strings;
        as_strings.reserve(paths.size());
        for (const auto& path : paths)
        {
            as_strings.push_back(path.string());
        }
        return JoinPaths(as_strings);
    }

    inline std::optional<std::string> ReadGoldenOutputForQuery(int query)
    {
        for (const auto& path : GoldenOutputCandidatesForQuery(query))
        {
            std::error_code ec;
            if (!std::filesystem::exists(path, ec))
            {
                continue;
            }

            std::ifstream in(path);
            if (!in.is_open())
            {
                continue;
            }

            std::ostringstream buffer;
            buffer << in.rdbuf();
            return Trim(buffer.str());
        }
        return std::nullopt;
    }

    inline std::string QueryCaseName(const testing::TestParamInfo<int>& info)
    {
        return "q" + std::to_string(info.param);
    }

    inline std::string AdmPlayCaseName(const testing::TestParamInfo<AdmPlayCase>& info)
    {
        std::string name =
            "q" + std::to_string(info.param.query) + "_" + ConfigName(info.param);
        std::replace_if(name.begin(), name.end(),
                        [](const char c)
                        {
                            return !std::isalnum(
                                static_cast<unsigned char>(c));
                        },
                        '_');
        return name;
    }
} // namespace

class AdmPlayCliHelpers
{
protected:
    void EnsureEnvironment()
    {
        if (!binary_.has_value())
        {
            binary_ = FindAdmPlayBinary();
        }
        ASSERT_TRUE(binary_.has_value())
        << "proteusadm-play binary not found; set PROTEUSADM_PLAY_BIN";
        ASSERT_TRUE(HasNvidiaGpu()) << "No NVIDIA GPU detected";
    }

    void EnsureRequiredPaths(const AdmPlayCase& test_case)
    {
        const auto required_paths = RequiredPaths(test_case);
        ASSERT_TRUE(required_paths.has_value())
        << "Missing path mapping for compression_type="
        << test_case.compression_type << " scale_factor="
        << test_case.scale_factor << " num_drives=" << test_case.num_drives
        << " decom_op=" << test_case.decom_op
        << " decom_reg=" << test_case.decom_reg
        << " precompute_rl=" << test_case.precompute_rl
        << " pre_decom=" << test_case.pre_decom
        << " tile_size=" << EffectiveTileSize(test_case)
        << " threads_per_block="
        << test_case.threads_per_block.value_or(0)
        << " items_per_thread="
        << test_case.items_per_thread.value_or(0)
        << " label=" << test_case.label;

        const auto missing_paths = MissingPaths(*required_paths);
        ASSERT_TRUE(missing_paths.empty())
        << "Missing dataset paths: " << JoinPaths(missing_paths);
    }

    void ReadGoldenOutputOrFail(int query, std::string& golden_output)
    {
        const auto golden = ReadGoldenOutputForQuery(query);
        ASSERT_TRUE(golden.has_value())
        << "Missing golden output for query=" << query << " in any of: "
        << JoinFsPaths(GoldenOutputCandidatesForQuery(query));
        golden_output = *golden;
    }

    void RunCaseAndExtractSingleResult(const AdmPlayCase& test_case,
                                       std::string& result)
    {
        ASSERT_TRUE(binary_.has_value());
        const auto run_args = BuildArgs(test_case);
        const auto run = RunProcessCapture(*binary_, run_args);
        ASSERT_NE(run.exit_code, -1)
        << "Failed to launch: " << BuildCommand(*binary_, run_args);
        ASSERT_EQ(run.exit_code, 0)
        << "Non-zero exit code from: " << BuildCommand(*binary_, run_args)
        << "\n=== Captured output begin ===\n"
        << run.output
        << "\n=== Captured output end ===";

        const auto parsed_results = ExtractAllQueryResults(run.output);
        ASSERT_EQ(parsed_results.size(), 1u)
        << "Expected exactly one QUERY RESULT block for config="
        << ConfigName(test_case) << " but got " << parsed_results.size()
        << "\n=== Captured output begin ===\n"
        << run.output
        << "\n=== Captured output end ===";

        result = parsed_results.front();
    }

    void RunAndCheckAgainstGolden(const AdmPlayCase& test_case)
    {
        EnsureEnvironment();
        EnsureRequiredPaths(test_case);

        std::string golden_output;
        ReadGoldenOutputOrFail(test_case.query, golden_output);

        std::string current_result;
        RunCaseAndExtractSingleResult(test_case, current_result);
        EXPECT_EQ(current_result, golden_output)
        << "Mismatch against golden output for config="
        << ConfigName(test_case) << " candidates="
        << JoinFsPaths(GoldenOutputCandidatesForQuery(test_case.query));
    }

    [[maybe_unused]] void RunAndCheckProducesResult(const AdmPlayCase& test_case)
    {
        EnsureEnvironment();
        EnsureRequiredPaths(test_case);

        std::string current_result;
        RunCaseAndExtractSingleResult(test_case, current_result);
        EXPECT_FALSE(current_result.empty())
        << "Empty QUERY RESULT for config=" << ConfigName(test_case);
    }

private:
    std::optional<std::string> binary_;
};

class AdmPlayCliCorrectnessBase : public ::testing::TestWithParam<int>,
                                  protected AdmPlayCliHelpers
{
};

class AdmPlayCliPaperConfigBase
    : public ::testing::TestWithParam<AdmPlayCase>,
      protected AdmPlayCliHelpers
{
};

#endif  // PROTEUS_TESTS_ADM_PLAY_CLI_COMMON_HPP
