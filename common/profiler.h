#ifndef PROFILER_H
#define PROFILER_H

#include <string>

namespace profiler {
    uint32_t device_cpu_cores();
    uint64_t device_physical_memory(bool available = true);
    uint64_t device_swap_memory(bool available = true);
    uint64_t get_disk_read_speed(const char * test_file, size_t buffer_size_mb = 500);
} // namespace profiler

#endif // PROFILER_H