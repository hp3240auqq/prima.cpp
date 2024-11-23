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
#elif defined(__APPLE__) && defined(__MACH__)
    #include <sys/sysctl.h>
    #include <mach/mach.h>
    #include <unistd.h>
#endif

#ifdef GGML_USE_METAL
    #include "ggml-metal.h"
#endif

#ifdef GGML_USE_CUDA
    #include "ggml-cuda.h"
#endif

#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <sys/types.h>
#include <vector>

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

static float device_flops(struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t, profiler_backend_type btype, int n_threads) {
    const int n_embd = llama_n_embd(model);
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

    size_t ctx_size = 0;
    ctx_size += 2 * ggml_tensor_overhead(); // tensors

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
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
    {
        struct ggml_init_params params0 = {
            /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
        };
        ctx_cgraph = ggml_init(params0);

        gf = ggml_new_graph(ctx_cgraph);
        struct ggml_tensor * cur = ggml_mul_mat(ctx_cgraph, tensor_a, tensor_b);
        ggml_build_forward_expand(gf, cur);
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

    // warm-up
    ggml_backend_graph_compute(backend, gf);

    const int64_t t_start = ggml_time_us();
    ggml_backend_graph_compute(backend, gf);
    const int64_t t_end = ggml_time_us();

    double elapsed_seconds = ((double)t_end - (double)t_start) / 1e6; // convert to seconds
    double flops = (2.0 * (double)n_embd * (double)n_embd * (double)n_embd) / elapsed_seconds / 1e9; // convert to GFLOPS

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

uint64_t device_physical_memory(bool available) {
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

uint64_t device_swap_memory(bool available) {
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

uint64_t device_disk_read_bw(const char * test_file, size_t buffer_size_mb) {
    uint64_t speed = 0;
    size_t buffer_size = buffer_size_mb * 1024 * 1024; // buffer size in bytes

    try {
        // open a file for reading
        std::ifstream file(test_file, std::ios::binary | std::ios::in);
        if (!file) {
            LOG_ERR("Unable to open the file at path: %s\n", test_file);
            return speed;
        }

        // prepare buffer for reading
        std::vector<char> buffer(buffer_size);

        auto start_time = std::chrono::high_resolution_clock::now();

        // read file into buffer
        file.read(buffer.data(), buffer.size());
        if (!file) {
            LOG_ERR("Failed to read enough data from the test file\n");
            return speed;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end_time - start_time;

        // speed in bytes per second
        if (elapsed_time.count() > 0) {
            speed = static_cast<uint64_t>(buffer.size() / elapsed_time.count());
        }

        buffer.clear();
        buffer.shrink_to_fit();
    } catch (const std::exception &e) {
        LOG_ERR("Exception while calculating disk read speed: %s\n", e.what());
    }

    return speed;
}

uint64_t device_memory_bw(size_t buffer_size_mb) {
    uint64_t speed = 0;
    size_t test_size = buffer_size_mb * 1024 * 1024; // convert MB to bytes

    try {
        // allocate memory for speed test
        std::vector<char> buffer(test_size, 1);

        // measure write speed
        auto start_time = std::chrono::high_resolution_clock::now();
        memset(buffer.data(), 0xAB, buffer.size());
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end_time - start_time;
        double write_speed = static_cast<double>(test_size) / elapsed_time.count();

        // measure read speed
        start_time = std::chrono::high_resolution_clock::now();
        volatile char temp = 0;
        for (size_t i = 0; i < buffer.size(); i += 64) {
            temp += buffer[i]; // read in steps of cache line size to minimize cache thrashing
        }
        end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = end_time - start_time;
        double read_speed = static_cast<double>(test_size) / elapsed_time.count();

        // average speed
        speed = static_cast<uint64_t>((write_speed + read_speed) / 2.0);

        buffer.clear();
        buffer.shrink_to_fit();
    } catch (const std::exception &e) {
        LOG_ERR("Exception while calculating memory speed: %s\n", e.what());
    }

    return speed;
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

void device_print_props(struct device_info * dev_info_set, int n, struct llama_model * model) {
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

    LOG_INF("| CPU flops (F32 x F32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CPU flops (F16 x F16, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].cpu_props.flops_f16);
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

    LOG_INF("| Physical Mem Total (GB)      ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.total_physical);
    }
    LOG_INF("\n");

    LOG_INF("| Physical Mem Available (GB)  ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.available_physical);
    }
    LOG_INF("\n");

    LOG_INF("| Swap Mem Total (GB)          ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.total_swap);
    }
    LOG_INF("\n");

    LOG_INF("| Swap Mem Available (GB)      ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.available_swap);
    }
    LOG_INF("\n");

    LOG_INF("| Mem Bandwidth (GB/s)         ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].memory.bandwidth);
    }
    LOG_INF("\n");

    LOG_INF("| Disk Read Bandwidth (GB/s)   ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].disk_read_bandwidth);
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

    LOG_INF("| GPU Mem Free (GB)            ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].gpu_props.memory_free);
    }
    LOG_INF("\n");

    LOG_INF("| GPU Mem Total (GB)           ");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.2f   ", dev_info_set[i].gpu_props.memory_total);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (F32xF32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Metal flops (F16xF16, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.metal_flops_f16);
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

    LOG_INF("| CUDA  flops (F32xF32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA  flops (F16xF16, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_f16);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA  flops (Q4KxF32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_q4k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| CUDA  flops (Q6KxF32, GFLOPS)");
    for (int i = 0; i < n; ++i) {
        LOG_INF("| %-10.1f   ", dev_info_set[i].gpu_props.cuda_flops_q6k_f32);
    }
    LOG_INF("\n");

    LOG_INF("| Model flops (input)          ");
    LOG_INF("| %-10lu   ", dev_info_set[0].model_flops.input_flops);
    LOG_INF("\n");

    LOG_INF("| Model flops (each layer)     ");
    LOG_INF("| %-10lu   ", dev_info_set[0].model_flops.layer_flops);
    LOG_INF("\n");

    LOG_INF("| Model flops (output)         ");
    LOG_INF("| %-10lu   ", dev_info_set[0].model_flops.output_flops);
    LOG_INF("\n");

    LOG_INF("| Model params (input)         ");
    LOG_INF("| %-10lu   ", dev_info_set[0].model_flops.input_params);
    LOG_INF("\n");

    LOG_INF("| Model params (each layer)    ");
    LOG_INF("| %-10lu   ", dev_info_set[0].model_flops.layer_params);
    LOG_INF("\n");

    LOG_INF("| Model params (output)        ");
    LOG_INF("| %-10lu   ", dev_info_set[0].model_flops.output_params);
    LOG_INF("\n");

    model_flops ffo  = dev_info_set[0].model_flops;
    int64_t total_flops = ffo.input_flops + ffo.output_flops + (ffo.layer_flops * llama_model_n_layers(model));
    double cpu_flops_f16 = dev_info_set[0].cpu_props.flops_f16 * 1e9;

    LOG_INF("| Token latency (ms)           ");
    LOG_INF("| %-10.2f   ", total_flops / cpu_flops_f16 * 1000);
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
                      + sizeof(float)       // disk_read_bandwidth
                      + sizeof(uint32_t)    // cpu_props.cores
                      + sizeof(float) * 4    // cpu_props.flops_f32, cpu_props.flops_f16, cpu_props.flops_q4k_f32, cpu_props.flops_q6k_f32
                      + sizeof(struct memory_info)
                      + sizeof(struct gpu_support)
                      + sizeof(float) * 10; // gpu_props.memory_free, gpu_props.memory_total, 
                                            // gpu_props.metal_flops_f32, gpu_props.metal_flops_f16, gpu_props.metal_flops_q4k_f32, gpu_props.metal_flops_q6k_f32, 
                                            // gpu_props.cuda_flops_f32, gpu_props.cuda_flops_f16, gpu_props.cuda_flops_q8, and gpu_props.cuda_flops_q4k

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
    memcpy(ptr, &dev_info->disk_read_bandwidth, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.cores, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    memcpy(ptr, &dev_info->cpu_props.flops_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_f16, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_q4k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->cpu_props.flops_q6k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->memory, sizeof(struct memory_info));
    ptr += sizeof(struct memory_info);

    memcpy(ptr, &dev_info->gpu_support, sizeof(struct gpu_support));
    ptr += sizeof(struct gpu_support);

    memcpy(ptr, &dev_info->gpu_props.memory_free, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.memory_total, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_f16, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_q4k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.metal_flops_q6k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_f16, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_q4k_f32, sizeof(float));
    ptr += sizeof(float);

    memcpy(ptr, &dev_info->gpu_props.cuda_flops_q6k_f32, sizeof(float));

    // no need to synchronize model flops
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
    memcpy(&dev_info->disk_read_bandwidth, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.cores, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    memcpy(&dev_info->cpu_props.flops_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_f16, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_q4k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->cpu_props.flops_q6k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->memory, ptr, sizeof(struct memory_info));
    ptr += sizeof(struct memory_info);

    memcpy(&dev_info->gpu_support, ptr, sizeof(struct gpu_support));
    ptr += sizeof(struct gpu_support);

    memcpy(&dev_info->gpu_props.memory_free, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.memory_total, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_f16, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_q4k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.metal_flops_q6k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_f16, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_q4k_f32, ptr, sizeof(float));
    ptr += sizeof(float);

    memcpy(&dev_info->gpu_props.cuda_flops_q6k_f32, ptr, sizeof(float));

    // no need to synchronize model flops
}