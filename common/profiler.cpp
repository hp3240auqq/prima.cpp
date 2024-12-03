#include "log.h"
#include "profiler.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
#elif defined(__linux__)
    #include <unistd.h>
    #include <sys/sysinfo.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <dirent.h>
#elif defined(__APPLE__) && defined(__MACH__)
    #include <sys/sysctl.h>
    #include <sys/param.h>
    #include <sys/mount.h>
    #include <mach/mach.h>
    #include <unistd.h>
#endif

#ifdef GGML_USE_METAL
    #include "ggml-metal.h"
#endif

#ifdef GGML_USE_CUDA
    #include "ggml-cuda.h"
    #include <cuda_runtime.h>
#endif

#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <sys/types.h>
#include <vector>
#include <inttypes.h>
#include <thread>
#include <random>
#include <regex>


const char * device_name() {
    static char device_name[256];

#if defined(_WIN32) || defined(_WIN64)
    DWORD size = sizeof(device_name);
    if (GetComputerNameA(device_name, &size) == 0) {
        strncpy(device_name, "Unknown Windows Device", sizeof(device_name));
    }
#elif defined(__linux__)
    if (gethostname(device_name, sizeof(device_name)) != 0) {
        strncpy(device_name, "Unknown Linux Device", sizeof(device_name));
    }
#elif defined(__APPLE__) && defined(__MACH__)
    if (gethostname(device_name, sizeof(device_name)) != 0) {
        strncpy(device_name, "Unknown Mac Device", sizeof(device_name));
    }
#else
    strncpy(device_name, "Unknown Device", sizeof(device_name));
#endif

    return device_name;
}

uint32_t device_cpu_cores() {
    unsigned int core_count = 1; // default to 1 in case of failure

#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    core_count = sysinfo.dwNumberOfProcessors;
#elif defined(__linux__)
    core_count = sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(__APPLE__) && defined(__MACH__)
    int mib[4];
    size_t len = sizeof(core_count);

    mib[0] = CTL_HW;
    mib[1] = HW_AVAILCPU;

    if (sysctl(mib, 2, &core_count, &len, NULL, 0) != 0 || core_count < 1) {
        mib[1] = HW_NCPU; // total number of cpus
        if (sysctl(mib, 2, &core_count, &len, NULL, 0) != 0 || core_count < 1) {
            core_count = 1; // default to 1 if sysctl fails
        }
    }
#endif

    return core_count;
}

static float device_flops(struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t, enum profiler_backend_type btype, int n_threads) {
    const int n_repeat = 1;
    const int n_embd   = llama_n_embd(model);
    std::vector<float> matrix_A(n_embd * n_embd, 1.0f); 
    std::vector<float> matrix_B(n_embd * n_embd, 1.0f / n_embd);

    ggml_backend_t backend = NULL;
    switch (btype) {
        case PROFILER_BACKEND_TYPE_CPU:
            backend = ggml_backend_cpu_init();
            break;
        case PROFILER_BACKEND_TYPE_METAL:
#ifdef GGML_USE_METAL
            backend = ggml_backend_metal_init();
#endif
            break;
        case PROFILER_BACKEND_TYPE_CUDA:
#ifdef GGML_USE_CUDA
            backend = ggml_backend_cuda_init(0);
#endif
            break;
    }

    if (!backend) {
        LOG_INF("%s: ggml backend init failed\n", __func__);
        return 0.0f;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 2 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, src0t, n_embd, n_embd);
    struct ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, src1t, n_embd, n_embd);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(tensor_a, matrix_A.data(), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, matrix_B.data(), 0, ggml_nbytes(tensor_b));

    struct ggml_cgraph  * gf         = NULL;
    struct ggml_context * ctx_cgraph = NULL;
    struct ggml_tensor  * cur        = NULL;
    {
        struct ggml_init_params params0 = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (n_repeat + 2) + ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
        };
        ctx_cgraph = ggml_init(params0);

        gf = ggml_new_graph(ctx_cgraph);
        cur = ggml_mul_mat(ctx_cgraph, tensor_a, tensor_b);
        for (int i = 0; i < n_repeat - 1; i++) {
            cur = ggml_mul_mat(ctx_cgraph, tensor_a, cur);
        }
        ggml_build_forward_expand(gf, cur);
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

    // warm-up
    // ggml_backend_graph_compute(backend, gf);

    const int64_t t_start = ggml_time_us();
    ggml_backend_graph_compute(backend, gf);
    const int64_t t_end = ggml_time_us();

    double elapsed_seconds = ((double)t_end - (double)t_start) / 1e6; // convert to seconds
    double flops = (2.0 * n_repeat * (double)n_embd * (double)n_embd * (double)n_embd) / elapsed_seconds / 1e9; // convert to GFLOPS

    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    return (float)flops;
}

float device_cpu_flops(struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t, int n_threads) {
    return device_flops(model, src0t, src1t, PROFILER_BACKEND_TYPE_CPU, n_threads);
}

float device_metal_flops(struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t) {
#ifdef GGML_USE_METAL
    return device_flops(model, src0t, src1t, PROFILER_BACKEND_TYPE_METAL, 4);
#endif

    (void)model;
    (void)src0t;
    (void)src1t;
    return 0.0f;
}

float device_cuda_flops(struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t) {
#ifdef GGML_USE_CUDA
    return device_flops(model, src0t, src1t, PROFILER_BACKEND_TYPE_CUDA, 4);
#endif

    (void)model;
    (void)src0t;
    (void)src1t;
    return 0.0f;
}

float device_inp_embd_delay(struct llama_model * model, enum ggml_type src0t, int n_tokens, int n_threads) {
    const int n_vocab = llama_n_vocab(model);
    const int n_embd  = llama_n_embd(model);
    
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        LOG_INF("%s: ggml backend init failed\n", __func__);
        return 0.0f;
    }

    size_t ctx_size = 0;
    ctx_size += 2 * ggml_tensor_overhead(); // tensors

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * tok_embd   = ggml_new_tensor_2d(ctx, src0t, n_embd, n_vocab);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<int32_t> matrix_A(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        matrix_A[i] = i % n_vocab;
    }

    const size_t embd_size = n_vocab * n_embd;
    void * matrix_B = nullptr;

    // quantization and dequantization functions
    ggml_type_traits_t qfns = ggml_internal_get_type_traits(src0t);
    if (!qfns.from_float || !qfns.to_float) {
        LOG_INF("Unsupported or uninitialized quantization type: %d\n", src0t);
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return 0.0f;
    }

    size_t QK_K = 0; 
    switch (src0t) {
        case GGML_TYPE_F32: {
            matrix_B = malloc(embd_size * sizeof(float));
            float * matrix_B_f32 = static_cast<float *>(matrix_B);
            for (size_t i = 0; i < embd_size; ++i) {
                matrix_B_f32[i] = static_cast<float>(rand()) / RAND_MAX;
            }
            break;
        }
        case GGML_TYPE_F16: {
            matrix_B = malloc(embd_size * sizeof(ggml_fp16_t));
            std::vector<float> temp_f32(embd_size);
            for (size_t i = 0; i < embd_size; ++i) {
                temp_f32[i] = static_cast<float>(rand()) / RAND_MAX;
            }
            ggml_fp32_to_fp16_row(temp_f32.data(), static_cast<ggml_fp16_t *>(matrix_B), embd_size);
            break;
        }
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
            QK_K = 256;
            matrix_B = malloc((embd_size / QK_K) * ggml_type_size(src0t));
            break;
        default:
            LOG_INF("Unsupported type: %d\n", src0t);
            ggml_free(ctx);
            ggml_backend_buffer_free(buffer);
            ggml_backend_free(backend);
            return 0.0f;
    }

    ggml_backend_tensor_set(inp_tokens, matrix_A.data(), 0, ggml_nbytes(inp_tokens));
    ggml_backend_tensor_set(tok_embd, matrix_B, 0, ggml_nbytes(tok_embd));

    struct ggml_cgraph  * gf         = NULL;
    struct ggml_context * ctx_cgraph = NULL;
    {
        struct ggml_init_params params0 = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 3 + ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
        };
        ctx_cgraph = ggml_init(params0);

        gf = ggml_new_graph(ctx_cgraph);
        struct ggml_tensor * cur = ggml_get_rows(ctx_cgraph, tok_embd, inp_tokens);
        ggml_build_forward_expand(gf, cur);
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

    // warm-up
    // ggml_backend_graph_compute(backend, gf);

    const int64_t t_start = ggml_time_us();
    ggml_backend_graph_compute(backend, gf);
    const int64_t t_end = ggml_time_us();

    double elapsed_ms = ((double)t_end - (double)t_start) / 1e3; // convert to ms

    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    return (float)elapsed_ms;
}

static bool device_is_docker_container() {
#if defined(__linux__)
    struct stat buffer;
    if (stat("/.dockerenv", &buffer) == 0) {
        return true;
    }

    std::ifstream cgroup_file("/proc/1/cgroup");
    std::string line;
    while (std::getline(cgroup_file, line)) {
        if (line.find("docker") != std::string::npos || 
            line.find("containerd") != std::string::npos) {
            return true;
        }
    }
    cgroup_file.close();
#endif

    return false;
}

static uint64_t device_host_physical_memory(bool available) {
    uint64_t memory = 0;

#if defined(_WIN32) || defined(_WIN64)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    if (available) {
        memory = status.ullAvailPhys;
    } else {
        memory = status.ullTotalPhys;
    }

#elif defined(__linux__)
    if (available) {
        // read available memory from /proc/meminfo
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        if (meminfo.is_open()) {
            while (std::getline(meminfo, line)) {
                if (line.find("MemAvailable:") == 0) {
                    std::istringstream iss(line);
                    std::string key;
                    uint64_t kb;
                    iss >> key >> kb;
                    memory = kb * 1024;
                    break;
                }
            }
            meminfo.close();
        }
    } else {
        // get total memory using sysinfo
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            memory = info.totalram * info.mem_unit;
        }
    }

#elif defined(__APPLE__) && defined(__MACH__)
    if (available) {
        mach_port_t host = mach_host_self();
        vm_statistics64_data_t vm_stats;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

        if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
            memory = (vm_stats.free_count + vm_stats.inactive_count) * sysconf(_SC_PAGESIZE);
        }
    } else {
        int mib[2];
        size_t len = sizeof(memory);
        mib[0] = CTL_HW;
        mib[1] = HW_MEMSIZE;
        sysctl(mib, 2, &memory, &len, NULL, 0);
    }
#endif

    return memory;
}

static uint64_t device_cgroup_physical_memory(bool available) {
    uint64_t memory_info   = 0;
    const char * file_path = nullptr;
    
    std::ifstream cgroup_file("/proc/cgroups");
    bool is_cgroup_v2 = false;
    if (cgroup_file.is_open()) {
        std::string line;
        while (std::getline(cgroup_file, line)) {
            if (line.find("0") != std::string::npos) {
                is_cgroup_v2 = true;
                break;
            }
        }
        cgroup_file.close();
    }

    if (is_cgroup_v2) {
        file_path = available
                        ? "/sys/fs/cgroup/memory.current" 
                        : "/sys/fs/cgroup/memory.max";    
    } else {
        file_path = available
                        ? "/sys/fs/cgroup/memory/memory.usage_in_bytes" 
                        : "/sys/fs/cgroup/memory/memory.limit_in_bytes"; 
    }

    std::ifstream file(file_path);
    if (file.is_open()) {
        std::string line;
        if (std::getline(file, line)) {
            try {
                memory_info = std::stoull(line);
            } catch (const std::exception &e) {
                memory_info = 0;
            }
        }
        file.close();
    } else {
        memory_info = 0;
    }

    return memory_info;
}

uint64_t device_physical_memory(bool available) {
    if (device_is_docker_container()) {
        uint64_t memory_total = device_cgroup_physical_memory(false);
        if (available) {
            uint64_t memory_usage = device_cgroup_physical_memory(true);
            return memory_total - memory_usage;
        }
        return memory_total;
    } else {
        return device_host_physical_memory(available);
    }
}

static uint64_t device_host_swap_memory(bool available) {
    uint64_t swap_memory = 0;

#if defined(_WIN32) || defined(_WIN64)
    PERFORMANCE_INFORMATION performance_info;
    performance_info.cb = sizeof(performance_info);
    if (GetPerformanceInfo(&performance_info, sizeof(performance_info))) {
        if (available) {
            swap_memory = (performance_info.PageFileTotal - performance_info.PageFileUsage) * performance_info.PageSize;
        } else {
            swap_memory = performance_info.PageFileTotal * performance_info.PageSize;
        }
    }
#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    uint64_t total_swap = 0;
    uint64_t free_swap = 0;

    if (meminfo.is_open()) {
        while (std::getline(meminfo, line)) {
            if (line.find("SwapTotal:") == 0) {
                std::istringstream iss(line);
                std::string key;
                uint64_t kb;
                iss >> key >> kb;
                total_swap = kb * 1024;
            } else if (line.find("SwapFree:") == 0) {
                std::istringstream iss(line);
                std::string key;
                uint64_t kb;
                iss >> key >> kb;
                free_swap = kb * 1024;
            }
        }
        meminfo.close();
    }

    if (available) {
        swap_memory = free_swap;
    } else {
        swap_memory = total_swap;
    }

#elif defined(__APPLE__) && defined(__MACH__)
    int mib[2] = {CTL_VM, VM_SWAPUSAGE};
    struct xsw_usage swap;
    size_t len = sizeof(swap);

    if (sysctl(mib, 2, &swap, &len, NULL, 0) == 0) {
        if (available) {
            swap_memory = swap.xsu_avail;
        } else {
            swap_memory = swap.xsu_total;
        }
    }
#endif

    return swap_memory;
}

static uint64_t device_cgroup_swap_memory(bool available) {
    if (available) return 0;

#if defined(__linux__)
    const char * file_path = nullptr;
    uint64_t swap_limit    = 0;

    std::ifstream cgroup_file("/proc/cgroups");
    bool is_cgroup_v2 = false;
    if (cgroup_file.is_open()) {
        std::string line;
        while (std::getline(cgroup_file, line)) {
            if (line.find("0") != std::string::npos) {
                is_cgroup_v2 = true;
                break;
            }
        }
        cgroup_file.close();
    }

    if (is_cgroup_v2) {
        file_path = "/sys/fs/cgroup/memory.swap.max"; 
    } else {
        file_path = "/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes"; 
    }

    std::ifstream mem_swap_file(file_path);
    if (mem_swap_file.is_open()) {
        std::string line;
        if (std::getline(mem_swap_file, line)) {
            try {
                swap_limit = std::stoull(line);
            } catch (const std::exception &e) {
                swap_limit = 0;
            }
        }
        mem_swap_file.close();
    }

    return swap_limit;
#else
    return 0; // Unsupported on non-Linux platforms
#endif
}

uint64_t device_swap_memory(bool available) {
    if (device_is_docker_container()) {
        return device_cgroup_swap_memory(available);
    } else {
        return device_host_swap_memory(available);
    }
}

static size_t get_page_size() {
    size_t page_size = 0;

#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    page_size = si.dwPageSize;
#elif defined(__APPLE__) || defined(__linux__)
    page_size = sysconf(_SC_PAGESIZE);
#endif

    return page_size;
}

static std::string get_default_device_path() {
#ifdef __linux__
    // find the first block device under /sys/block
    const std::string block_path = "/sys/block/";
    DIR * dir = opendir(block_path.c_str());
    if (!dir) {
        LOG_INF("Unable to open %s\n", block_path.c_str());
        return "";
    }
    struct dirent * entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] != '.') { // ignore hidden files/directories
            std::string device = entry->d_name;
            closedir(dir);
            return "/dev/" + device;
        }
    }
    closedir(dir);
    LOG_INF("No block devices found in %s\n", block_path.c_str());
    return "";
#elif __APPLE__
    // use the root device as a default
    return "/";
#elif _WIN32
    // use the default drive (usually C:)
    char volume_name[MAX_PATH];
    if (GetVolumeInformation("C:\\", volume_name, sizeof(volume_name), NULL, NULL, NULL, NULL, 0)) {
        return "C:\\";
    } else {
        LOG_INF("Failed to determine default volume\n");
        return "";
    }
#else
    LOG_INF("Unsupported platform\n");
    return "";
#endif
}

static size_t get_default_readahead_size() {
    const std::string device_path = get_default_device_path();

#ifdef __linux__
    std::string device = device_path.empty() ? get_default_device_path() : device_path;
    if (device.empty()) return 0;

    // read from sysfs
    std::string sysfs_path = "/sys/block/" + device.substr(device.find_last_of("/") + 1) + "/queue/read_ahead_kb";
    std::ifstream file(sysfs_path);
    if (file.is_open()) {
        size_t read_ahead_kb;
        file >> read_ahead_kb;
        file.close();
        return read_ahead_kb * 1024; // convert to bytes
    } else {
        return 0;
    }
#elif __APPLE__
    // use statfs to determine default block size
    struct statfs stats;
    std::string path = device_path.empty() ? "/" : device_path;
    if (statfs(path.c_str(), &stats) == 0) {
        return stats.f_iosize; // return in bytes
    } else {
        LOG_INF("statfs failed\n");
        return 0;
    }
#elif _WIN32
    // use GetDiskFreeSpace to get default cluster size
    std::string drive = device_path.empty() ? "C:\\" : device_path;
    DWORD sectorsPerCluster, bytesPerSector, numberOfFreeClusters, totalNumberOfClusters;
    if (GetDiskFreeSpace(drive.c_str(), &sectorsPerCluster, &bytesPerSector, &numberOfFreeClusters, &totalNumberOfClusters)) {
        return sectorsPerCluster * bytesPerSector; // return in bytes
    } else {
        LOG_INF("GetDiskFreeSpace failed\n");
        return 0;
    }
#else
    LOG_INF("Unsupported platform\n");
    return 0;
#endif
}

static void external_fio_impl(float * read_bw, float * write_bw, bool op_rand, int n_threads) {
    const char * test_file = "fio_test";
    const char * fio_conf_template = R"(
[global]
ioengine=%s
direct=1
time_based=1
runtime=2
size=500M
group_reporting=1

[read-job]
rw=%s
bs=%s
filename=%s
numjobs=%d

[write-job]
rw=%s
bs=%s
filename=%s
numjobs=%d
)";

    size_t page_size = get_page_size();
    if (page_size == 0) {
        LOG_INF("Unable to get system page size, use 4KB by default\n");
        page_size = 4 * 1024;
    }
    // format the page size as a readable string (e.g., "16k" or "4k")
    char page_size_str[8];
    if (page_size >= 1024) {
        snprintf(page_size_str, sizeof(page_size_str), "%zuk", page_size / 1024);
    } else {
        snprintf(page_size_str, sizeof(page_size_str), "%zu", page_size);
    }

    size_t readahead_size = get_default_readahead_size();
    if (readahead_size == 0) {
        LOG_INF("Unable to get system readahead size, use 128KB by default\n");
        readahead_size = 128 * 1024;
    }
    // format the readahead size as a readable string (e.g., "128k" or "1m")
    char readahead_str[8];
    if (readahead_size >= 1024 * 1024) {
        snprintf(readahead_str, sizeof(readahead_str), "%zuM", readahead_size / 1024 / 1024);
    } else if (readahead_size >= 1024) {
        snprintf(readahead_str, sizeof(readahead_str), "%zuk", readahead_size / 1024);
    } else {
        snprintf(readahead_str, sizeof(readahead_str), "%zu",  readahead_size);
    }

    const char * read_type  = op_rand ? "randread" : "read";
    const char * write_type = op_rand ? "randwrite" : "write";
    const char * block_size = op_rand ? page_size_str : readahead_str;

    const char * ioengine    = "posixaio";
    bool retry_with_sync     = false;
    const char * output_file = "fio_output.log";
    const char * conf_file   = "config.fio";

    do {
        char fio_conf[1024];
        snprintf(fio_conf, sizeof(fio_conf), fio_conf_template, ioengine,
                 read_type,  block_size, test_file, n_threads,
                 write_type, block_size, test_file, n_threads);

        
        std::ofstream conf(conf_file);
        if (!conf) {
            LOG_INF("Error: Unable to create configuration file\n");
            return;
        }
        conf << fio_conf;
        conf.close();

        std::string command = "fio " + std::string(conf_file) + " > " + std::string(output_file) + " 2>&1";
        int ret = std::system(command.c_str());

        if (ret == 0) {
            retry_with_sync = false; // Execution succeeded
        } else {
            LOG_INF("Engine posixaio not loadable, retrying with sync engine\n");
            ioengine = "sync";
            retry_with_sync = true;
        }
    } while (retry_with_sync);

    // parse fio output
    std::ifstream result(output_file);
    if (!result) {
        LOG_INF("Error: Failed to open fio output file\n");
        return;
    }
    *read_bw = 0.0f;
    *write_bw = 0.0f;
    
    std::string line;
    std::regex read_regex(R"(READ: bw=([0-9.]+)([a-zA-Z/]+))");
    std::regex write_regex(R"(WRITE: bw=([0-9.]+)([a-zA-Z/]+))");
    std::smatch match;

    while (std::getline(result, line)) {
        if (std::regex_search(line, match, read_regex)) {
            float value = std::stof(match[1]);
            std::string unit = match[2];
            if (unit == "MiB/s") {
                *read_bw = value * 1024.0f * 1024.0f / 1e9;  // convert MiB/s to GB/s
            } else if (unit == "MB/s") {
                *read_bw = value / 1000.0f;  // convert MB/s to GB/s
            }
        } else if (std::regex_search(line, match, write_regex)) {
            float value = std::stof(match[1]);
            std::string unit = match[2];
            if (unit == "MiB/s") {
                *write_bw = value * 1024.0f * 1024.0f / 1e9;  // convert MiB/s to GB/s
            } else if (unit == "MB/s") {
                *write_bw = value / 1000.0f;  // convert MB/s to GB/s
            }
        }
    }

    // clean up temporary files
    std::remove(test_file);
    std::remove(conf_file);
    std::remove(output_file);
}

void device_disk_rnd_bw(float * read_rnd_bw, float * write_rnd_bw, int n_threads) {
    external_fio_impl(read_rnd_bw, write_rnd_bw, true, n_threads);
}

void device_disk_seq_bw(float * read_seq_bw, float * write_seq_bw, int n_threads) {
    external_fio_impl(read_seq_bw, write_seq_bw, false, n_threads);
}

float device_memory_bw(int n_thread) {
    size_t buffer_size = 5L * 1024 * 1024; // 5 MiB
    std::vector<std::thread> thread_pool;
    std::vector<double> results(n_thread);
    std::vector<char *> buffers(n_thread);

    for (int i = 0; i < n_thread; ++i) {
        buffers[i] = new char[buffer_size];
    }

    auto memory_bw_test = [](char * buffer, size_t size, double & result) {
        // read test
        volatile char temp = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < size; i += 64) {
            temp += buffer[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        result = size / elapsed.count() / 1e9; // GB/s
    };

    for (int i = 0; i < n_thread; ++i) {
        thread_pool.emplace_back(memory_bw_test, buffers[i], buffer_size, std::ref(results[i]));
    }
    for (auto & t : thread_pool) {
        t.join();
    }

    double bandwidth = 0.0f;
    for (double result : results) {
        bandwidth += result;
    }

    for (int i = 0; i < n_thread; ++i) {
        delete[] buffers[i];
    }

    return static_cast<float>(bandwidth);
}

static float device_read_vram_bw(struct llama_model * model, enum profiler_backend_type btype) {
    const int n_embd = llama_n_embd(model) * 2;
    std::vector<float> matrix_A(n_embd * n_embd, 1.0f);

    ggml_backend_t backend = NULL;
    switch (btype) {
        case PROFILER_BACKEND_TYPE_METAL:
#ifdef GGML_USE_METAL
            backend = ggml_backend_metal_init();
#endif
            break;
        case PROFILER_BACKEND_TYPE_CUDA:
#ifdef GGML_USE_CUDA
            backend = ggml_backend_cuda_init(0);
#endif
            break;
        case PROFILER_BACKEND_TYPE_CPU:
            break;
    }

    if (!backend) {
        LOG_INF("%s: ggml backend init failed\n", __func__);
        return 0.0f;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
    tensor_a->op = GGML_OP_READ;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(tensor_a, matrix_A.data(), 0, ggml_nbytes(tensor_a));

    struct ggml_cgraph  * gf         = NULL;
    struct ggml_context * ctx_cgraph = NULL;
    {
        struct ggml_init_params params0 = {
            /*.mem_size   =*/ ggml_tensor_overhead() + ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
        };
        ctx_cgraph = ggml_init(params0);

        gf = ggml_new_graph(ctx_cgraph);
        ggml_build_forward_expand(gf, tensor_a);
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    const int64_t t_start = ggml_time_us();
    ggml_backend_graph_compute(backend, gf);
    const int64_t t_end = ggml_time_us();

    double elapsed_s = ((double)t_end - (double)t_start) / 1e6;
    size_t total_bytes = n_embd * n_embd * sizeof(float);
    float bandwidth = (total_bytes / elapsed_s) / 1e9; // GB/s

    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    return bandwidth;
}

float device_metal_read_vram_bw(struct llama_model * model) {
#ifdef GGML_USE_METAL
    return device_read_vram_bw(model, PROFILER_BACKEND_TYPE_METAL);
#endif

    (void)model;
    return 0.0f;
}

float device_cuda_read_vram_bw(struct llama_model * model) {
#ifdef GGML_USE_CUDA
    return device_read_vram_bw(model, PROFILER_BACKEND_TYPE_CUDA);
#endif

    (void)model;
    return 0.0f;
}

int device_has_metal(void) {
    return ggml_cpu_has_metal();
}

int device_has_cuda(void) {
    return ggml_cpu_has_cuda();
}

int device_has_vulkan(void) {
    return ggml_cpu_has_vulkan();
}

int device_has_kompute(void) {
    return ggml_cpu_has_kompute();
}

int device_has_gpublas(void) {
    return ggml_cpu_has_gpublas();
}

int device_has_blas(void) {
    return ggml_cpu_has_blas();
}

int device_has_sycl(void) {
    return ggml_cpu_has_sycl();
}

void device_get_props(struct llama_model * model, int device, struct ggml_backend_dev_props * props) {
    ggml_backend_buffer_type_t buft_type;
    if (device == -1) { // type cpu
        buft_type = ggml_backend_cpu_buffer_type();
    } else { // type gpu
        buft_type = llama_dev_buffer_type(model, device);
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft_type);
    ggml_backend_dev_get_props(dev, props);
}

static float device_compute_delay(struct device_info & dev_info, int n_layers, const struct llama_context_params cparams) {
    struct model_flops n_flops   = dev_info.model_flops;
    struct cpu_props cpu         = dev_info.cpu_props;
    const int n_gpu_layers       = cparams.n_gpu_layers;
    double gpu_latency_per_layer = 0.0f;
    double cpu_latency_per_layer = 0.0f;

#ifdef GGML_USE_CUDA
    struct gpu_props gpu = dev_info.gpu_props;

    gpu_latency_per_layer += (double)n_flops.layer_f32_f32  / (double)gpu.cuda_flops_f32_f32 / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_f16_f32  / (double)gpu.cuda_flops_f16_f32 / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q4k_f32  / (double)gpu.cuda_flops_q4k_f32 / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q6k_f32  / (double)gpu.cuda_flops_q6k_f32 / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q80_f32  / (double)gpu.cuda_flops_q80_f32 / 1e9;
#elif GGML_USE_METAL
    struct gpu_props gpu = dev_info.gpu_props;

    gpu_latency_per_layer += (double)n_flops.layer_f32_f32  / (double)gpu.metal_flops_f32_f32 / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_f16_f32  / (double)gpu.metal_flops_f16_f32 / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q4k_f32  / (double)gpu.metal_flops_q4k_f32 / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q6k_f32  / (double)gpu.metal_flops_q6k_f32 / 1e9;
    gpu_latency_per_layer += (double)n_flops.layer_q80_f32  / (double)gpu.metal_flops_q80_f32 / 1e9;
#endif

    cpu_latency_per_layer += (double)n_flops.layer_f32_f32  / (double)cpu.flops_f32_f32 / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_f16_f32  / (double)cpu.flops_f16_f32 / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_q4k_f32  / (double)cpu.flops_q4k_f32 / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_q6k_f32  / (double)cpu.flops_q6k_f32 / 1e9;
    cpu_latency_per_layer += (double)n_flops.layer_q80_f32  / (double)cpu.flops_q80_f32 / 1e9;

    double total_latency = 0.0f;
    
#if defined(GGML_USE_METAL) || defined(GGML_USE_CUDA)
    total_latency += gpu_latency_per_layer * n_gpu_layers;
    total_latency += cpu_latency_per_layer * (n_layers - n_gpu_layers);
#else
    (void)n_gpu_layers;
    (void)gpu_latency_per_layer;
    total_latency += cpu_latency_per_layer * n_layers;
#endif

    total_latency += (double)n_flops.output_f32_f32 / (double)cpu.flops_f32_f32 / 1e9;
    total_latency += (double)n_flops.output_f16_f32 / (double)cpu.flops_f16_f32 / 1e9;
    total_latency += (double)n_flops.output_q4k_f32 / (double)cpu.flops_q4k_f32 / 1e9;
    total_latency += (double)n_flops.output_q6k_f32 / (double)cpu.flops_q6k_f32 / 1e9;
    total_latency += (double)n_flops.output_q80_f32 / (double)cpu.flops_q80_f32 / 1e9;

    total_latency *= 1000; // convert to ms

    total_latency += n_flops.inp_embd_ms;

    return static_cast<float>(total_latency);
}

// estimate the memory access delay, except for the input embedding because it has been considered in n_flops.inp_embd_ms
static float device_memory_access_delay(struct device_info & dev_info, const struct llama_context_params cparams, int n_layers) {
    struct model_params n_params = dev_info.model_params;
    int n_gpu_layers = cparams.n_gpu_layers;

    int64_t layer_bytes = 
                   n_params.layer_f32 * 4 +
                   n_params.layer_f16 * 2 +
                   n_params.layer_q4k / 2 +
                   n_params.layer_q6k * 3 / 8 +
                   n_params.layer_q80;

    int64_t output_bytes = 
                   n_params.output_f32 * 4 +
                   n_params.output_f16 * 2 +
                   n_params.output_q4k / 2 +
                   n_params.output_q6k * 3 / 8 +
                   n_params.output_q80;

#if defined(GGML_USE_CUDA) || defined(GGML_USE_METAL)
    int64_t vram_bytes = layer_bytes * n_gpu_layers;
    int64_t ram_bytes  = layer_bytes * (n_layers - n_gpu_layers) + output_bytes;

#ifdef GGML_USE_CUDA
    double vram_access_delay = (double)(vram_bytes) / 1e6 / dev_info.gpu_props.cuda_read_vram_bw;
#elif GGML_USE_METAL
    double vram_access_delay = (double)(vram_bytes) / 1e6 / dev_info.gpu_props.metal_read_vram_bw;
#endif

    double ram_access_delay  = (double)(ram_bytes)  / 1e6 / dev_info.memory.cpu_read_ram_bw;
    return static_cast<float>(vram_access_delay + ram_access_delay); // ms

#else
    (void)n_gpu_layers;
    int64_t ram_bytes = layer_bytes * n_layers + output_bytes;
    double ram_access_delay = (double)(ram_bytes) / 1e6 / dev_info.memory.cpu_read_ram_bw;
    return static_cast<float>(ram_access_delay); // ms
#endif
}

static float device_disk_access_delay(struct device_info & dev_info, struct llama_model * model, const struct llama_context_params cparams) {
    auto n_params         = dev_info.model_params;
    int n_layers          = llama_model_n_layers(model);
    int n_gpu_layers      = cparams.n_gpu_layers;

    uint64_t cpu_kv_size;
    uint64_t gpu_kv_size;
    uint64_t cpu_compute_buf;
    uint64_t gpu_compute_buf;

#if defined(GGML_USE_METAL) || defined(GGML_USE_CUDA)
    llama_model_kvcache_size(&cpu_kv_size, &gpu_kv_size, model, cparams, true);
    llama_model_compute_buf_size(&cpu_compute_buf, &gpu_compute_buf, model, cparams, true);
#else
    llama_model_kvcache_size(&cpu_kv_size, &gpu_kv_size, model, cparams, false);
    llama_model_compute_buf_size(&cpu_compute_buf, &gpu_compute_buf, model, cparams, false);
#endif

    double cpu_kv_size_gb     = static_cast<double>(cpu_kv_size) / 1e9;     // convert to GB
    double cpu_compute_buf_gb = static_cast<double>(cpu_compute_buf) / 1e9; // convert to GB

    int64_t cpu_total_bytes =
                   n_params.layer_f32 * 4 +
                   n_params.layer_f16 * 2 +
                   n_params.layer_q4k / 2 +
                   n_params.layer_q6k * 3 / 8 +
                   n_params.layer_q80;

#if defined(GGML_USE_METAL) || defined(GGML_USE_CUDA)
    cpu_total_bytes *= (n_layers - n_gpu_layers);
#else
    (void)n_gpu_layers;
    cpu_total_bytes *= n_layers;
#endif

    cpu_total_bytes += (
                   n_params.output_f32 * 4 +
                   n_params.output_f16 * 2 +
                   n_params.output_q4k / 2 +
                   n_params.output_q6k * 3 / 8 +
                   n_params.output_q80);
    
    float cpu_total_gbytes = (double)cpu_total_bytes / 1e9; // convert to GB
    float cpu_mem_avail = dev_info.memory.available_physical * 1024.0f * 1024.0f * 1024.0f / 1e9; // convert to GB
          cpu_mem_avail -= static_cast<float>(cpu_kv_size_gb);
          cpu_mem_avail -= static_cast<float>(cpu_compute_buf_gb);
          
// #ifdef __linux__
//     float disk_read_bw = dev_info.disk.read_seq_bw; // GB/s
// #else
//     float disk_read_bw = dev_info.disk.read_rnd_bw; // GB/s
// #endif

    float disk_read_bw = dev_info.disk.read_rnd_bw; // GB/s
    return std::max(0.0, static_cast<double>(cpu_total_gbytes - cpu_mem_avail) / disk_read_bw * 1000); // convert to ms
}

void device_print_props(struct device_info * dev_info_set, int n, struct llama_model * model, const struct llama_context_params cparams) {
    LOG_INF("\n-------------------------------------------------------------------------------------------\n");
    LOG_INF("| Property                     ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| Rank %-8d", i);
        GGML_ASSERT((int)dev_info_set[i].rank == i);
    }
    LOG_INF("\n-------------------------------------------------------------------------------------------\n");

    LOG_INF("| Device Name                  ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.10s   ", dev_info_set[i].device_name);
    }
    LOG_INF("\n");

    LOG_INF("| CPU Name                     ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.10s   ", dev_info_set[i].cpu_props.name);
    }
    LOG_INF("\n");

    LOG_INF("| CPU Description              ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.10s   ", dev_info_set[i].cpu_props.description);
    }
    LOG_INF("\n");

    LOG_INF("| Number of CPU cores          ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10u   ", dev_info_set[i].cpu_props.cores);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (F32xF32, GFLOPS)  ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_f32_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (F16xF32, GFLOPS)  ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_f16_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (Q4K x F32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_q4k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (Q6K x F32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_q6k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (Q80 x F32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_q80_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Physical Mem Total (GB)      ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.total_physical);
    }
    LOG_INF("\n");

    LOG_INF("| Physical Mem Available (GiB) ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.available_physical);
    }
    LOG_INF("\n");

    LOG_INF("| Swap Mem Total (GiB)         ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.total_swap);
    }
    LOG_INF("\n");

    LOG_INF("| Swap Mem Available (GiB)     ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.available_swap);
    }
    LOG_INF("\n");

    LOG_INF("| CPU RAM Read BW (GB/s)       ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.cpu_read_ram_bw);
    }
    LOG_INF("\n");

    LOG_INF("| Disk Read Seq Speed (GB/s)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].disk.read_seq_bw);
    }
    LOG_INF("\n");

    LOG_INF("| Disk Write Seq Speed (GB/s)  ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].disk.write_seq_bw);
    }
    LOG_INF("\n");

    LOG_INF("| Disk Read Rnd Speed (GB/s)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].disk.read_rnd_bw);
    }
    LOG_INF("\n");

    LOG_INF("| Disk Write Rnd Speed (GB/s)  ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].disk.write_rnd_bw);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Metal                    ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.metal);
    }
    LOG_INF("\n");

    LOG_INF("| GPU CUDA                     ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.cuda);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Vulkan                   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.vulkan);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Kompute                  ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.kompute);
    }
    LOG_INF("\n");

    LOG_INF("| GPU BLAS                     ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.gpublas);
    }
    LOG_INF("\n");

    LOG_INF("| BLAS                         ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.blas);
    }
    LOG_INF("\n");

    LOG_INF("| SYCL                         ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10d   ", dev_info_set[i].gpu_support.sycl);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Name                     ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.10s   ", dev_info_set[i].gpu_props.name);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Description              ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.10s   ", dev_info_set[i].gpu_props.description);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Mem Free (GiB)           ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].gpu_props.memory_free);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Mem Total (GiB)          ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].gpu_props.memory_total);
    }
    LOG_INF("\n");

    LOG_INF("| Metal VRAM Read BW (GB/s)    ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].gpu_props.metal_read_vram_bw);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (F32xF32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_f32_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (F16xF32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_f16_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (Q4KxF32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_q4k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (Q6KxF32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_q6k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (Q80xF32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_q80_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA VRAM Read BW (GB/s)     ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].gpu_props.cuda_read_vram_bw);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (F32xF32, GFLOPS) ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_f32_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (F16xF32, GFLOPS) ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_f16_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (Q4KxF32, GFLOPS) ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_q4k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (Q6KxF32, GFLOPS) ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_q6k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA flops (Q80xF32, GFLOPS) ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_q80_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Model flops (output F32xF32) ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_f32_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output F16xF32) ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_f16_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output Q4KxF32) ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_q4k_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output Q6KxF32) ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_q6k_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (output Q80xF32) ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.output_q80_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer F32xF32)  ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_f32_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer F16xF32)  ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_f16_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer Q4KxF32)  ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_q4k_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer Q6KxF32)  ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_q6k_f32);
    LOG_INF("\n");

    LOG_INF("| Model flops (layer Q80xF32)  ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_flops.layer_q80_f32);
    LOG_INF("\n");

    LOG_INF("| Model params (input F32)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_f32);
    LOG_INF("\n");

    LOG_INF("| Model params (input F16)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_f16);
    LOG_INF("\n");

    LOG_INF("| Model params (input Q4K)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_q4k);
    LOG_INF("\n");

    LOG_INF("| Model params (input Q6K)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_q6k);
    LOG_INF("\n");

    LOG_INF("| Model params (input Q80)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.input_q80);
    LOG_INF("\n");

    LOG_INF("| Model params (layer F32)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_f32);
    LOG_INF("\n");

    LOG_INF("| Model params (layer F16)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_f16);
    LOG_INF("\n");

    LOG_INF("| Model params (layer Q4K)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_q4k);
    LOG_INF("\n");

    LOG_INF("| Model params (layer Q6K)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_q6k);
    LOG_INF("\n");

    LOG_INF("| Model params (layer Q80)     ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.layer_q80);
    LOG_INF("\n");

    LOG_INF("| Model params (output F32)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_f32);
    LOG_INF("\n");

    LOG_INF("| Model params (output F16)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_f16);
    LOG_INF("\n");

    LOG_INF("| Model params (output Q4K)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_q4k);
    LOG_INF("\n");

    LOG_INF("| Model params (output Q6K)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_q6k);
    LOG_INF("\n");

    LOG_INF("| Model params (output Q80)    ");
    LOG_INF("| %-10" PRId64 "   ", dev_info_set[0].model_params.output_q80);
    LOG_INF("\n");

    // todo: calculate for each device, not only master
    float latency = 0.0f;
    int n_layers  = llama_model_n_layers (model);
    latency += device_compute_delay      (dev_info_set[0], n_layers, cparams);
    latency += device_memory_access_delay(dev_info_set[0], cparams,  n_layers);
    latency += device_disk_access_delay  (dev_info_set[0], model,    cparams); // if physical memory is not enough, some tensor weights will be released from memory and reloaded by mmap later
    
    LOG_INF("| Token latency (ms)           ");
    LOG_INF("| %-10.2f   ", latency);
    LOG_INF("\n");

    LOG_INF("-------------------------------------------------------------------------------------------\n\n");
}


size_t serialize(const struct device_info * dev_info, char ** buffer) {
    // calculate total size for serialized buffer
    size_t device_name_len     = strlen(dev_info->device_name) + 1;
    size_t cpu_name_len        = strlen(dev_info->cpu_props.name) + 1;
    size_t cpu_description_len = strlen(dev_info->cpu_props.description) + 1;
    size_t gpu_name_len        = strlen(dev_info->gpu_props.name) + 1;
    size_t gpu_description_len = strlen(dev_info->gpu_props.description) + 1;

    size_t total_size = sizeof(uint32_t)
                      + sizeof(size_t) * 5  // for lengths of strings
                      + device_name_len
                      + cpu_name_len
                      + cpu_description_len
                      + gpu_name_len
                      + gpu_description_len
                      + sizeof(struct disk_props)
                      + sizeof(uint32_t)    // cpu_props.cores
                      + sizeof(float) * 5    // cpu_props.flops_f32_f32, cpu_props.flops_f16_f32, cpu_props.flops_q4k_f32, cpu_props.flops_q6k_f32, cpu_props.flops_q80_f32
                      + sizeof(struct memory_info)
                      + sizeof(struct gpu_support)
                      + sizeof(float) * 14; // gpu_props.memory_free, gpu_props.memory_total, gpu_props.metal_read_vram_bw, gpu_props.cuda_read_vram_bw,
                                            // gpu_props.metal_flops_f32_f32, gpu_props.metal_flops_f16_f32, gpu_props.metal_flops_q4k_f32, gpu_props.metal_flops_q6k_f32, gpu_props.metal_flops_q80_f32, 
                                            // gpu_props.cuda_flops_f32_f32, gpu_props.cuda_flops_f16_f32, gpu_props.cuda_flops_q4k_f32, gpu_props.cuda_flops_q6k_f32, gpu_props.cuda_flops_q80_f32

    *buffer = (char *)malloc(total_size);
    char * ptr = *buffer;

    // rank
    memcpy(ptr, &dev_info->rank, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    // copy string lengths and string data
    memcpy(ptr, &device_name_len, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, dev_info->device_name, device_name_len);
    ptr += device_name_len;

    memcpy(ptr, &cpu_name_len, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, dev_info->cpu_props.name, cpu_name_len);
    ptr += cpu_name_len;

    memcpy(ptr, &cpu_description_len, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, dev_info->cpu_props.description, cpu_description_len);
    ptr += cpu_description_len;

    memcpy(ptr, &gpu_name_len, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, dev_info->gpu_props.name, gpu_name_len);
    ptr += gpu_name_len;

    memcpy(ptr, &gpu_description_len, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, dev_info->gpu_props.description, gpu_description_len);
    ptr += gpu_description_len;

    // copy the non-string members
    memcpy(ptr, &dev_info->disk, sizeof(struct disk_props));
    ptr += sizeof(struct disk_props);

    memcpy(ptr, &dev_info->cpu_props.cores, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    memcpy(ptr, &dev_info->cpu_props.flops_f32_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_f16_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_q4k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_q6k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_q80_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->memory, sizeof(struct memory_info));
    ptr += sizeof(struct memory_info);

    memcpy(ptr, &dev_info->gpu_support, sizeof(struct gpu_support));
    ptr += sizeof(struct gpu_support);

    memcpy(ptr, &dev_info->gpu_props.memory_free, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.memory_total, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_read_vram_bw, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_f32_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_f16_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_q4k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_q6k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_q80_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_read_vram_bw, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_f32_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_f16_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_q4k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_q6k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_q80_f32, sizeof(float));

    // no need to synchronize model flops and model params
    return total_size;
}

void deserialize(const char * buffer, struct device_info * dev_info) {
    const char * ptr = buffer;

    // rank
    memcpy(&dev_info->rank, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    // device_name
    size_t device_name_len;
    memcpy(&device_name_len, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    dev_info->device_name = (char *)malloc(device_name_len);
    memcpy(const_cast<void*>(static_cast<const void*>(dev_info->device_name)), ptr, device_name_len);
    ptr += device_name_len;

    // cpu_props.name
    size_t cpu_name_len;
    memcpy(&cpu_name_len, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    dev_info->cpu_props.name = (char *)malloc(cpu_name_len);
    memcpy(const_cast<void*>(static_cast<const void*>(dev_info->cpu_props.name)), ptr, cpu_name_len);
    ptr += cpu_name_len;

    // cpu_props.description
    size_t cpu_description_len;
    memcpy(&cpu_description_len, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    dev_info->cpu_props.description = (char *)malloc(cpu_description_len);
    memcpy(const_cast<void*>(static_cast<const void*>(dev_info->cpu_props.description)), ptr, cpu_description_len);
    ptr += cpu_description_len;

    // gpu_props.name
    size_t gpu_name_len;
    memcpy(&gpu_name_len, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    dev_info->gpu_props.name = (char *)malloc(gpu_name_len);
    memcpy(const_cast<void*>(static_cast<const void*>(dev_info->gpu_props.name)), ptr, gpu_name_len);
    ptr += gpu_name_len;

    // gpu_props.description
    size_t gpu_description_len;
    memcpy(&gpu_description_len, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    dev_info->gpu_props.description = (char *)malloc(gpu_description_len);
    memcpy(const_cast<void*>(static_cast<const void*>(dev_info->gpu_props.description)), ptr, gpu_description_len);
    ptr += gpu_description_len;

    // other non-string members
    memcpy(&dev_info->disk, ptr, sizeof(struct disk_props));
    ptr += sizeof(struct disk_props);

    memcpy(&dev_info->cpu_props.cores, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    memcpy(&dev_info->cpu_props.flops_f32_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_f16_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_q4k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_q6k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_q80_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->memory, ptr, sizeof(struct memory_info));
    ptr += sizeof(struct memory_info);

    memcpy(&dev_info->gpu_support, ptr, sizeof(struct gpu_support));
    ptr += sizeof(struct gpu_support);

    memcpy(&dev_info->gpu_props.memory_free, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.memory_total, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_read_vram_bw, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_f32_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_f16_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_q4k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_q6k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_q80_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_read_vram_bw, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_f32_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_f16_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_q4k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_q6k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_q80_f32, ptr, sizeof(float));

    // no need to synchronize model flops and model params
}