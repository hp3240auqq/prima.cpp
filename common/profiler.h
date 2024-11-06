#ifndef PROFILER_H
#define PROFILER_H

#include "llama.h"
#include <string>

#define BUFFER_SIZE_MB 1024

namespace profiler {
    const char * device_name(void); 

    uint32_t device_cpu_cores      (void);
    uint64_t device_physical_memory(bool available = true);
    uint64_t device_swap_memory    (bool available = true);
    uint64_t device_disk_read_bw   (const char * test_file, size_t buffer_size_mb = BUFFER_SIZE_MB);
    uint64_t device_memory_bw      (size_t buffer_size_mb = BUFFER_SIZE_MB);
    void     device_get_props      (struct llama_model * model, int device, struct ggml_backend_dev_props * props);

    int device_has_metal(void);
    int device_has_cuda(void);
    int device_has_vulkan(void);
    int device_has_kompute(void);
    int device_has_gpublas(void);
    int device_has_blas(void);
    int device_has_sycl(void);

} // namespace profiler

#endif // PROFILER_H