#ifndef PROFILER_H
#define PROFILER_H

#include "ggml.h"
#include "llama.h"

struct cpu_props {
    const char * name;
    const char * description;
    uint32_t     cores;
    float        flops_f32; // in GFLOPS
    float        flops_f16; // in GFLOPS

    cpu_props()
        : name(""), description(""), cores(0), flops_f32(0.0f), flops_f16(0.0f) {}
};

struct memory_info {
    float        total_physical;      // in GB
    float        available_physical;  // in GB
    float        total_swap;          // in GB
    float        available_swap;      // in GB
    float        bandwidth;           // in GB/s

    memory_info()
        : total_physical(0.0f), available_physical(0.0f), total_swap(0.0f), available_swap(0.0f), bandwidth(0.0f) {}
};

struct gpu_support {
    bool         metal;
    bool         cuda;
    bool         vulkan;
    bool         kompute;
    bool         gpublas;
    bool         blas;
    bool         sycl;

    gpu_support()
        : metal(false), cuda(false), vulkan(false), kompute(false), gpublas(false), blas(false), sycl(false) {}
};

struct gpu_props {
    const char * name;
    const char * description;
    float        memory_free;    // in GB
    float        memory_total;   // in GB
    float        metal_flops;    // in GFLOPS
    float        cuda_flops_f32; // in GFLOPS
    float        cuda_flops_f16; // in GFLOPS
    float        cuda_flops_q8;  // in GFLOPS
    float        cuda_flops_q4k; // in GFLOPS

    gpu_props()
        : name(""), description(""), memory_free(0.0f), memory_total(0.0f), metal_flops(0.0f), cuda_flops_f32(0.0f), cuda_flops_f16(0.0f), cuda_flops_q8(0.0f), cuda_flops_q4k(0.0f) {}
};

struct model_flops {
    // model flops
    int64_t input_flops;
    int64_t output_flops;
    int64_t layer_flops;
    
    // model params
    int64_t input_params;
    int64_t output_params;
    int64_t layer_params;

    model_flops()
        : input_flops(0), output_flops(0), layer_flops(0), input_params(0), output_params(0), layer_params(0) {}
};

struct device_info {
    uint32_t           rank;
    const char *       device_name;
    float              disk_read_bandwidth;  // in GB/s
    struct cpu_props   cpu_props;
    struct memory_info memory;
    struct gpu_support gpu_support;
    struct gpu_props   gpu_props;
    struct model_flops model_flops;

    device_info()
        : rank(0), device_name(""), disk_read_bandwidth(0.0f), cpu_props(), memory(), gpu_support(), gpu_props(), model_flops() {}
};

enum profiler_backend_type {
    PROFILER_BACKEND_TYPE_CPU   = 0,
    PROFILER_BACKEND_TYPE_METAL = 1,
    PROFILER_BACKEND_TYPE_CUDA  = 2,
};

const char * device_name(void); 

uint32_t device_cpu_cores      (void);
float    device_cpu_flops      (struct llama_model * model, enum ggml_type dtype, int n_threads);
float    device_metal_flops    (struct llama_model * model, enum ggml_type dtype);
float    device_cuda_flops     (struct llama_model * model, enum ggml_type dtype);
uint64_t device_physical_memory(bool available);
uint64_t device_swap_memory    (bool available);
uint64_t device_disk_read_bw   (const char * test_file, size_t buffer_size_mb);
uint64_t device_memory_bw      (size_t buffer_size_mb);
void     device_get_props      (struct llama_model * model, int device, struct ggml_backend_dev_props * props); 
void     device_print_props    (struct device_info * dev_info_set, int n, struct llama_model * model);

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
