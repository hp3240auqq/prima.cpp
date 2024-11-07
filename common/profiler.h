#ifndef PROFILER_H
#define PROFILER_H

#include "llama.h"

struct cpu_props {
    const char * name;
    const char * description;
    uint32_t     cores;
};

struct memory_info {
    float        total_physical;      // in GB
    float        available_physical;  // in GB
    float        total_swap;          // in GB
    float        available_swap;      // in GB
    float        bandwidth;           // in GB/s
};

struct gpu_support {
    bool         metal;
    bool         cuda;
    bool         vulkan;
    bool         kompute;
    bool         gpublas;
    bool         blas;
    bool         sycl;
};

struct gpu_props {
    const char * name;
    const char * description;
    float        memory_free;   // in GB
    float        memory_total;  // in GB
};

struct device_info {
    uint32_t           rank;
    const char *       device_name;
    float              disk_read_bandwidth;  // in GB/s
    struct cpu_props   cpu_props;
    struct memory_info memory;
    struct gpu_support gpu_support;
    struct gpu_props   gpu_props;
};

const char * device_name(void); 

uint32_t device_cpu_cores      (void);
uint64_t device_physical_memory(bool available);
uint64_t device_swap_memory    (bool available);
uint64_t device_disk_read_bw   (const char * test_file, size_t buffer_size_mb);
uint64_t device_memory_bw      (size_t buffer_size_mb);
void     device_get_props      (struct llama_model * model, int device, struct ggml_backend_dev_props * props); 
void     device_print_props    (struct device_info * dev_info_set, int n);

int      device_has_metal  (void);
int      device_has_cuda   (void);
int      device_has_vulkan (void);
int      device_has_kompute(void);
int      device_has_gpublas(void);
int      device_has_blas   (void);
int      device_has_sycl   (void);

size_t   serialize  (const struct device_info * dev_info, char ** buffer);
void     deserialize(const char * buffer, struct device_info * dev_info);

#endif // PROFILER_H
