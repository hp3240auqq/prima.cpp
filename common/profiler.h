#ifndef PROFILER_H
#define PROFILER_H

namespace profiler {
    uint32_t device_cpu_cores();
    uint64_t device_physical_memory(bool available = true);
} // namespace profiler

#endif // PROFILER_H