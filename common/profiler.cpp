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

#include <chrono>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <sys/types.h>
#include <vector>

namespace profiler {

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

// ggml_backend_buffer_type_t llama_dev_buffer_type(const llama_model * model, int device)

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

} // namespace profiler