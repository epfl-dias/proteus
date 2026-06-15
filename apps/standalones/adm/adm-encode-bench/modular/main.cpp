// ---------------------------------------------------------------------------
// modular_ssb11 — SSB Query 1.1 benchmark with modular compressed execution.
//
// Supports RFOR_GLOBAL, RFOR_SHARED, FL_BITPACK, FL_DELTA, UNCOMPRESSED_ROW, and ALL modes.
// All column data is given as binary int32 files (one value per element).
//
// Usage:
//   modular_ssb11 [--mode RFOR_GLOBAL|RFOR_SHARED|FL_BITPACK|FL_DELTA|UNCOMPRESSED_ROW|ALL]
//                 [--name LABEL]
//                 [--runs N]
//                 [--block_threads BT]     (FL_BITPACK: 32/64/128/256; default 128)
//                 [--items_per_thread IPT]  (FL_BITPACK: paired with BT; default 8)
//                 <lo_orderdate_file> <lo_quantity_file>
//                 <lo_extendedprice_file> <lo_discount_file>
//
// Results are written to <source>/results/[LABEL_]<YYYY-MM-DD_HH-MM-SS>.csv.
// Stdout emits only PASS/FAIL.
// ---------------------------------------------------------------------------

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>

#include "runner/run_config.cuh"
#include "runner/kernel_runner.cuh"
#include "runner/metrics.cuh"
#include "mode/rfor_global.cuh"
#include "mode/rfor_shared.cuh"
#include "mode/fl_bitpack.cuh"
#include "mode/fl_delta.cuh"
#include "mode/uncompressed_row.cuh"

// csv_utils.h lives in the sibling compressed-block-to-tuple directory
// (added to include path by CMakeLists.txt)
#include "csv_utils.h"

using namespace std;

// 3-column CSV helper: group,metric,value
static void csvOut(const string& group, const string& metric, const string& value, ostream& os) {
    os << group << "," << metric << "," << value << "\n";
}
template<typename T>
static void csvOut(const string& group, const string& metric, const T& value, ostream& os) {
    csvOut(group, metric, toCsvString(value), os);
}

static size_t getFileSize(const string& path) {
    ifstream f(path, ios::binary | ios::ate);
    if (!f.is_open()) return 0;
    return static_cast<size_t>(f.tellg());
}

static bool loadColumnInto(const string& path, size_t num_elements, vector<int>& out) {
    ifstream f(path, ios::binary);
    if (!f.is_open()) return false;
    out.resize(num_elements);
    f.read(reinterpret_cast<char*>(out.data()), num_elements * sizeof(int));
    return static_cast<size_t>(f.gcount()) == num_elements * sizeof(int);
}

static ExecMode parseMode(const string& s) {
    if (s == "RFOR_GLOBAL")       return ExecMode::RFOR_GLOBAL;
    if (s == "RFOR_SHARED")       return ExecMode::RFOR_SHARED;
    if (s == "FL_BITPACK")        return ExecMode::FL_BITPACK;
    if (s == "FL_DELTA")          return ExecMode::FL_DELTA;
    if (s == "UNCOMPRESSED_ROW")  return ExecMode::UNCOMPRESSED_ROW;
    throw invalid_argument("Unknown mode: " + s);
}


// ---------------------------------------------------------------------------
// LPT scheduler simulation.
// Assigns blocks to SM slots using Longest Processing Time First and returns
// the critical-path (total, decomp, query, reduce) time.
// ---------------------------------------------------------------------------
struct LPTResult { double total, decomp, query, reduce; };

static LPTResult longestProcessingTimeFirst(
    const vector<nsecs>& decomp_ns,
    const vector<nsecs>& query_ns,
    const vector<nsecs>& reduce_ns,
    int max_blocks,
    int tiles_per_block)
{
    const size_t n_tiles = decomp_ns.size();
    using T4 = tuple<nsecs, nsecs, nsecs, nsecs>;
    const size_t tpb = static_cast<size_t>(tiles_per_block);
    const size_t n   = (n_tiles + tpb - 1) / tpb;
    vector<T4> blocks(n, T4{0, 0, 0, 0});
    for (size_t i = 0; i < n_tiles; ++i) {
        const size_t g = i / tpb;
        get<0>(blocks[g]) += decomp_ns[i] + query_ns[i] + reduce_ns[i];
        get<1>(blocks[g]) += decomp_ns[i];
        get<2>(blocks[g]) += query_ns[i];
        get<3>(blocks[g]) += reduce_ns[i];
    }
    sort(blocks.begin(), blocks.end(), greater<T4>());

    vector<T4> slots(max_blocks, T4{0, 0, 0, 0});
    for (const auto& b : blocks) {
        auto it = min_element(slots.begin(), slots.end());
        get<0>(*it) += get<0>(b);
        get<1>(*it) += get<1>(b);
        get<2>(*it) += get<2>(b);
        get<3>(*it) += get<3>(b);
    }
    const auto& crit = *max_element(slots.begin(), slots.end());
    return { static_cast<double>(get<0>(crit)),
             static_cast<double>(get<1>(crit)),
             static_cast<double>(get<2>(crit)),
             static_cast<double>(get<3>(crit)) };
}

static void reportResults(
    const vector<RunResult>& results,
    size_t                   total_data_size,
    size_t                   num_tuples,
    int                      max_blocks,
    int                      tiles_per_block,
    double                   peak_bw_gbs,
    ostream&                 out)
{
    const double total_gb = static_cast<double>(total_data_size) / 1e9;

    if (!results.empty())
        csvOut("perf", "revenue", results[0].revenue, out);

    // Collect per-run stats over measurement runs (skip warmup at index 0).
    vector<float> v_elapsed_ms, v_tput_gbs, v_tput_mtuples;
    vector<float> v_lpt_decomp, v_lpt_query, v_lpt_reduce, v_lpt_total;

    int best_r = 1;
    for (int r = 1; r < static_cast<int>(results.size()); r++) {
        const float ms = results[r].elapsed_ms;
        const double s = ms * 1e-3;
        v_elapsed_ms.push_back(ms);
        v_tput_gbs.push_back(static_cast<float>(total_gb / s));
        v_tput_mtuples.push_back(static_cast<float>(
            static_cast<double>(num_tuples) / s / 1e6));

        const auto lpt = longestProcessingTimeFirst(
            results[r].block_decompress_ns,
            results[r].block_query_ns,
            results[r].block_reduce_ns,
            max_blocks, tiles_per_block);
        v_lpt_decomp.push_back(static_cast<float>(lpt.decomp));
        v_lpt_query.push_back(static_cast<float>(lpt.query));
        v_lpt_reduce.push_back(static_cast<float>(lpt.reduce));
        v_lpt_total.push_back(static_cast<float>(lpt.total));

        if (results[r].elapsed_ms < results[best_r].elapsed_ms) best_r = r;
    }

    if (v_elapsed_ms.empty()) return;

    auto emit_mean = [&](const string& metric, vector<float> v) {
        csvOut("perf", metric + "_mean", computeStats(v).mean, out);
    };

    emit_mean("elapsed_ms",         v_elapsed_ms);
    emit_mean("throughput_gbs",     v_tput_gbs);
    emit_mean("throughput_mtuples", v_tput_mtuples);
    emit_mean("lpt_decompress_ns",  v_lpt_decomp);
    emit_mean("lpt_query_ns",       v_lpt_query);
    emit_mean("lpt_reduce_ns",      v_lpt_reduce);
    emit_mean("lpt_total_ns",       v_lpt_total);

    const auto lpt_best = longestProcessingTimeFirst(
        results[best_r].block_decompress_ns,
        results[best_r].block_query_ns,
        results[best_r].block_reduce_ns,
        max_blocks, tiles_per_block);
    if (lpt_best.total > 0) {
        csvOut("perf", "decompress_pct", lpt_best.decomp / lpt_best.total * 100.0, out);
        csvOut("perf", "query_pct",      lpt_best.query  / lpt_best.total * 100.0, out);
        csvOut("perf", "reduce_pct",     lpt_best.reduce / lpt_best.total * 100.0, out);
    }
    if (peak_bw_gbs > 0) {
        const double best_tput_gbs = total_gb / (results[best_r].elapsed_ms * 1e-3);
        csvOut("perf", "bw_utilization_pct", best_tput_gbs / peak_bw_gbs * 100.0, out);
    }
}

template<typename MC>
static bool runMode(
    const char*                          mode_name,
    const vector<vector<int>>&           col_data,
    size_t                               num_tuples,
    size_t                               total_data_size,
    const LaunchConfig&                  cfg,
    cub::CachingDeviceAllocator&         allocator,
    const cudaDeviceProp&                props,
    double                               peak_bw_gbs,
    ostream&                             out)
{
    const int num_SMs = props.multiProcessorCount;

    csvOut("kernel", "mode", string(mode_name), out);

    // Compress and upload.
    auto t0 = chrono::high_resolution_clock::now();
    typename MC::ColData compressed = MC::compress(col_data, num_tuples, allocator);
    const double compress_s = chrono::duration<double>(
        chrono::high_resolution_clock::now() - t0).count();
    csvOut("perf", "compression_time_s", compress_s, out);

    const size_t compressed_bytes  = MC::compressed_size_bytes(compressed);
    const double compression_ratio = static_cast<double>(total_data_size)
                                   / static_cast<double>(compressed_bytes);
    csvOut("perf", "uncompressed_bytes",  total_data_size,   out);
    csvOut("perf", "compressed_bytes",    compressed_bytes,  out);
    csvOut("perf", "compression_ratio",   compression_ratio, out);

    // Kernel config.
    const int regs = MC::kernel_regs();
    csvOut("kernel", "block_threads",        MC::BT,      out);
    csvOut("kernel", "items_per_thread",     MC::IPT,     out);
    csvOut("kernel", "registers_per_thread", regs,        out);
    csvOut("kernel", "warps_per_block",      MC::BT / 32, out);

    // Smem breakdown.
    const auto [decomp_smem, staging_smem, cub_smem] = MC::smem_breakdown();
    csvOut("smem", "decomp_smem_bytes",  decomp_smem,     out);
    csvOut("smem", "staging_smem_bytes", staging_smem,    out);
    csvOut("smem", "cub_smem_bytes",     cub_smem,        out);
    csvOut("smem", "total_smem_bytes",   MC::smem_size(), out);

    // Occupancy.
    const size_t n_chunks = MC::num_chunks(compressed);
    const int    max_bpsm = MC::occupancy_blocks_per_sm();
    const size_t n_blocks = MC::grid_blocks(n_chunks);
    const int    max_blks = max_bpsm * num_SMs;
    const size_t n_waves  = max_blks > 0
        ? (n_blocks + static_cast<size_t>(max_blks) - 1) / static_cast<size_t>(max_blks)
        : 0;

    const OccupancyLimits lim =
        computeOccupancyLimits(props, MC::smem_size(), MC::BT, regs);
    const int    active_warps = max_bpsm * (MC::BT / 32);
    const double occ = static_cast<double>(active_warps)
                     / (props.maxThreadsPerMultiProcessor / 32);

    csvOut("occupancy", "max_blocks_per_SM",     max_bpsm,              out);
    csvOut("occupancy", "active_warps_per_SM",   active_warps,          out);
    csvOut("occupancy", "theoretical_occupancy", occ,                   out);
    csvOut("occupancy", "smem_limit_blocks",     lim.smem_limit_blocks, out);
    csvOut("occupancy", "warp_limit_blocks",     lim.warp_limit_blocks, out);
    csvOut("occupancy", "reg_limit_blocks",      lim.reg_limit_blocks,  out);
    csvOut("occupancy", "binding_constraint",    string(lim.binding),   out);

    // Grid launch.
    const double wave_eff = (n_waves > 0 && max_blks > 0)
        ? static_cast<double>(n_blocks) / (static_cast<double>(n_waves) * max_blks)
        : 0.0;
    csvOut("grid", "num_blocks",      n_blocks, out);
    csvOut("grid", "max_blocks",      max_blks, out);
    csvOut("grid", "num_waves",       n_waves,  out);
    csvOut("grid", "wave_efficiency", wave_eff, out);

    // Run kernel.
    vector<RunResult> results;
    try {
        results = run_mode<MC>(compressed, cfg, allocator);
    } catch (const exception& e) {
        csvOut("error", "launch_failed", string(e.what()), out);
        return false;
    }

    reportResults(results, total_data_size, num_tuples,
                  max_blks, MC::tiles_per_block(), peak_bw_gbs, out);
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // -----------------------------------------------------------------------
    // Argument parsing
    // -----------------------------------------------------------------------
    constexpr int NUM_COLS = 4;
    string mode_selector;
    string run_name;
    string output_dir;
    int    num_runs        = 5;
    int    block_threads   = 128;
    int    items_per_thread = 8;
    int    first_col_arg   = 1;

    for (int i = 1; i < argc; i++) {
        const string arg(argv[i]);
        if      (arg == "--mode"             && i + 1 < argc) mode_selector    = argv[++i];
        else if (arg == "--name"             && i + 1 < argc) run_name         = argv[++i];
        else if (arg == "--output_dir"       && i + 1 < argc) output_dir       = argv[++i];
        else if (arg == "--runs"             && i + 1 < argc) num_runs         = stoi(argv[++i]);
        else if (arg == "--block_threads"    && i + 1 < argc) block_threads    = stoi(argv[++i]);
        else if (arg == "--items_per_thread" && i + 1 < argc) items_per_thread = stoi(argv[++i]);
        else { first_col_arg = i; break; }
    }

    if (mode_selector.empty()) {
        cerr << "FAIL: --mode is required. "
             << "Valid: RFOR_GLOBAL|RFOR_SHARED|FL_BITPACK|FL_DELTA|UNCOMPRESSED_ROW\n";
        return 1;
    }
    if (argc - first_col_arg < NUM_COLS) {
        cerr << "FAIL: Usage: " << argv[0]
             << " --mode RFOR_GLOBAL|RFOR_SHARED|FL_BITPACK|FL_DELTA|UNCOMPRESSED_ROW"
                " [--name LABEL] [--output_dir DIR] [--runs N]"
                " [--block_threads BT] [--items_per_thread IPT]"
                " <lo_orderdate> <lo_quantity> <lo_extendedprice> <lo_discount>\n";
        return 1;
    }

    // Output file: [output_dir or <source>/results]/<name>.csv
    auto _now = std::chrono::system_clock::now();
    std::time_t _t = std::chrono::system_clock::to_time_t(_now);
    char _ts[32];
    std::strftime(_ts, sizeof(_ts), "%Y-%m-%d_%H-%M-%S", std::localtime(&_t));
    const string ts_prefix = run_name.empty() ? _ts : run_name;
    const string dir = output_dir.empty()
        ? string(MODULAR_RESULTS_DIR) + "/results"
        : output_dir;
    const string output_file = dir + "/" + ts_prefix + ".csv";

    // Create parent directory if needed, then open output file.
    {
        auto parent = std::filesystem::path(output_file).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
    }
    ofstream fout(output_file);
    if (!fout.is_open()) {
        cerr << "FAIL: cannot open output file: " << output_file << "\n";
        return 1;
    }
    // CSV header.
    fout << "group,metric,value\n";

    ExecMode mode;
    try {
        mode = parseMode(mode_selector);
    } catch (const invalid_argument& e) {
        cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // Load column data
    // -----------------------------------------------------------------------
    vector<string> col_files(NUM_COLS);
    for (int c = 0; c < NUM_COLS; c++) col_files[c] = argv[first_col_arg + c];

    const size_t per_col_bytes = getFileSize(col_files[0]);
    if (per_col_bytes == 0 || per_col_bytes % sizeof(int) != 0) {
        cerr << "FAIL: first column file is empty or not int-aligned\n";
        return 1;
    }
    const size_t num_tuples      = per_col_bytes / sizeof(int);
    const size_t total_data_size = per_col_bytes * NUM_COLS;

    vector<vector<int>> col_data(NUM_COLS);
    for (int c = 0; c < NUM_COLS; c++) {
        if (!loadColumnInto(col_files[c], num_tuples, col_data[c])) {
            cerr << "FAIL: could not load column file: " << col_files[c] << "\n";
            return 1;
        }
    }

    // -----------------------------------------------------------------------
    // GPU device info + hardware metrics
    // -----------------------------------------------------------------------
    cub::CachingDeviceAllocator allocator(true, false);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    const double peak_bw_gbs = peakBandwidthGBs(props);

    csvOut("hardware", "gpu_name",         string(props.name),                    fout);
    csvOut("hardware", "num_SMs",          props.multiProcessorCount,             fout);
    csvOut("hardware", "max_warps_per_SM", props.maxThreadsPerMultiProcessor / 32,fout);
    csvOut("hardware", "peak_bw_gbs",      peak_bw_gbs,                           fout);
    csvOut("hardware", "l2_cache_bytes",   props.l2CacheSize,                     fout);
    csvOut("query",    "query",            string("SSB Q1.1"),                    fout);
    csvOut("query",    "num_tuples",       num_tuples,                            fout);
    csvOut("query",    "num_runs",         num_runs,                              fout);

    LaunchConfig cfg;
    cfg.num_cols         = NUM_COLS;
    cfg.num_runs         = num_runs;
    cfg.block_threads    = block_threads;
    cfg.items_per_thread = items_per_thread;

    // -----------------------------------------------------------------------
    // Dispatch
    // -----------------------------------------------------------------------
    bool ok = true;

#define RUN(Config, Name) \
    ok = runMode<Config>(Name, col_data, num_tuples, total_data_size, cfg, allocator, props, peak_bw_gbs, fout)

    if (mode == ExecMode::FL_BITPACK) {
        const int bt  = cfg.block_threads;
        const int ipt = cfg.items_per_thread;
        // Use local alias to avoid macro comma issue with template args.
#define FLBP(BT, IPT) { using C = FLBitPackConfig<BT, IPT>; RUN(C, "FL_BITPACK"); }
        if      (bt == 32  && ipt == 32) FLBP(32,  32)
        else if (bt == 64  && ipt == 16) FLBP(64,  16)
        else if (bt == 128 && ipt ==  8) FLBP(128,  8)
        else if (bt == 256 && ipt ==  4) FLBP(256,  4)
        else if (bt == 512 && ipt ==  2) FLBP(512,  2)
        else if (bt == 1024 && ipt ==  1) FLBP(1024,  1)
        else {
            cerr << "FAIL: FL_BITPACK requires BT*IPT==1024 and BT%32==0. "
                 << "Valid pairs: (32,32) (64,16) (128,8) (256,4).\n";
            return 1;
        }
#undef FLBP
    } else if (mode == ExecMode::RFOR_SHARED) {
        const int bt  = cfg.block_threads;
        const int ipt = cfg.items_per_thread;
#define RFOR_S(BT, IPT) { using C = RFORSharedConfig<BT, IPT>; RUN(C, "RFOR_SHARED"); }
        if      (bt == 64  && ipt == 4) RFOR_S(64,  4)
        else if (bt == 64  && ipt == 8) RFOR_S(64,  8)
        else if (bt == 128 && ipt == 4) RFOR_S(128, 4)
        else if (bt == 128 && ipt == 8) RFOR_S(128, 8)
        else if (bt == 256 && ipt == 4) RFOR_S(256, 4)
        else if (bt == 256 && ipt == 8) RFOR_S(256, 8)
        else {
            cerr << "FAIL: RFOR_SHARED unsupported (BT,IPT). "
                 << "Valid pairs: (64,4) (64,8) (128,4) (128,8) (256,4) (256,8).\n";
            return 1;
        }
#undef RFOR_S
    } else if (mode == ExecMode::FL_DELTA)        { RUN(FLDeltaConfig,        "FL_DELTA");        }
      else if (mode == ExecMode::RFOR_GLOBAL)      { RUN(RFORGlobalConfig,     "RFOR_GLOBAL");     }
      else if (mode == ExecMode::UNCOMPRESSED_ROW) { RUN(UncompressedRowConfig,"UNCOMPRESSED_ROW");}

#undef RUN

    if (!ok) {
        cerr << "FAIL: mode " << mode_selector << " failed\n";
        return 1;
    }

    cout << "PASS: results written to " << output_file << "\n"
         << "Run the command below to download the output file into local repo:" << "\n"
         << "  scp anagupta@diascld49.iccluster.epfl.ch:" << output_file
         << " scratchpad/results/\n";
    return 0;
}
