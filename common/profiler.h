#ifndef PROFILER_H
#define PROFILER_H

#include <string>

#define BUFFER_SIZE_MB 1024

namespace profiler {
    const char * device_name(); 
    uint32_t device_cpu_cores();
    uint64_t device_physical_memory(bool available = true);
    uint64_t device_swap_memory(bool available = true);
    uint64_t device_disk_read_bw(const char * test_file, size_t buffer_size_mb = BUFFER_SIZE_MB);
    uint64_t device_memory_bw(size_t buffer_size_mb = BUFFER_SIZE_MB);
} // namespace profiler

#endif // PROFILER_H