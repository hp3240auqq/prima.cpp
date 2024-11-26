#ifndef PROFILER_H
#define PROFILER_H

#include "ggml.h"
#include "llama.h"

struct cpu_props {
    const char * name;
    const char * description;
    uint32_t     cores;
    float        flops_f32_f32; // in GFLOPS
    float        flops_f16_f32; // in GFLOPS
    float        flops_q4k_f32; // in GFLOPS
    float        flops_q6k_f32; // in GFLOPS
    float        flops_q80_f32; // in GFLOPS

    cpu_props() : 
        name(""), 
        description(""), 
        cores(0), 
        flops_f32_f32(0.0f), 
        flops_f16_f32(0.0f), 
        flops_q4k_f32(0.0f),
        flops_q6k_f32(0.0f),
        flops_q80_f32(0.0f) {}
};

struct memory_info {
    float        total_physical;     // in GB
    float        available_physical; // in GB
    float        total_swap;         // in GB
    float        available_swap;     // in GB
    float        read_bandwidth;     // in GB/s

    memory_info() : 
        total_physical    (0.0f), 
        available_physical(0.0f), 
        total_swap        (0.0f), 
        available_swap    (0.0f), 
        read_bandwidth    (0.0f) {}
};

struct gpu_support {
    bool         metal;
    bool         cuda;
    bool         vulkan;
    bool         kompute;
    bool         gpublas;
    bool         blas;
    bool         sycl;

    gpu_support() : 
        metal  (false), 
        cuda   (false), 
        vulkan (false), 
        kompute(false), 
        gpublas(false), 
        blas   (false), 
        sycl   (false) {}
};

struct gpu_props {
    const char * name;
    const char * description;
    float        memory_free;         // in GB
    float        memory_total;        // in GB
    float        read_bandwidth;      // in GB/s
    float        metal_flops_f32_f32; // in GFLOPS
    float        metal_flops_f16_f32; // in GFLOPS
    float        metal_flops_q4k_f32; // in GFLOPS
    float        metal_flops_q6k_f32; // in GFLOPS
    float        metal_flops_q80_f32; // in GFLOPS
    float        cuda_flops_f32_f32;  // in GFLOPS
    float        cuda_flops_f16_f32;  // in GFLOPS
    float        cuda_flops_q4k_f32;  // in GFLOPS
    float        cuda_flops_q6k_f32;  // in GFLOPS
    float        cuda_flops_q80_f32;  // in GFLOPS

    gpu_props() : 
        name(""), 
        description(""), 
        memory_free        (0.0f), 
        memory_total       (0.0f), 
        read_bandwidth     (1.0f),
        metal_flops_f32_f32(0.0f), 
        metal_flops_f16_f32(0.0f),
        metal_flops_q4k_f32(0.0f),
        metal_flops_q6k_f32(0.0f),
        metal_flops_q80_f32(0.0f),
        cuda_flops_f32_f32 (0.0f), 
        cuda_flops_f16_f32 (0.0f), 
        cuda_flops_q4k_f32 (0.0f), 
        cuda_flops_q6k_f32 (0.0f),
        cuda_flops_q80_f32 (0.0f) {}
};

struct model_flops {
    float   inp_embd_ms;
    int64_t output_f32_f32;
    int64_t output_f16_f32;
    int64_t output_q4k_f32;
    int64_t output_q6k_f32;
    int64_t output_q80_f32;
    int64_t layer_f32_f32;
    int64_t layer_f16_f32;
    int64_t layer_q4k_f32;
    int64_t layer_q6k_f32;
    int64_t layer_q80_f32;

    model_flops() : 
        inp_embd_ms(0.0f),
        output_f32_f32(0), 
        output_f16_f32(0),
        output_q4k_f32(0),
        output_q6k_f32(0), 
        output_q80_f32(0),
        layer_f32_f32 (0),
        layer_f16_f32 (0),
        layer_q4k_f32 (0),
        layer_q6k_f32 (0),
        layer_q80_f32 (0) {}
};

struct model_params {
    int64_t input_f32;
    int64_t input_f16;
    int64_t input_q4k;
    int64_t input_q6k;
    int64_t input_q80;
    int64_t output_f32;
    int64_t output_f16;
    int64_t output_q4k;
    int64_t output_q6k;
    int64_t output_q80;
    int64_t layer_f32;
    int64_t layer_f16;
    int64_t layer_q4k;
    int64_t layer_q6k;
    int64_t layer_q80;

    model_params() :
        input_f32 (0),
        input_f16 (0),
        input_q4k (0),
        input_q6k (0),
        input_q80 (0),
        output_f32(0),
        output_f16(0),
        output_q4k(0),
        output_q6k(0),
        output_q80(0),
        layer_f32 (0),
        layer_f16 (0),
        layer_q4k (0),
        layer_q6k (0),
        layer_q80 (0) {}
};

struct device_info {
    uint32_t            rank;
    const char *        device_name;
    float               disk_read_bandwidth;  // in GB/s
    struct cpu_props    cpu_props;
    struct memory_info  memory;
    struct gpu_support  gpu_support;
    struct gpu_props    gpu_props;
    struct model_flops  model_flops;
    struct model_params model_params;

    device_info() : 
        rank(0), 
        device_name(""), 
        disk_read_bandwidth(0.0f), 
        cpu_props(), 
        memory(), 
        gpu_support(), 
        gpu_props(), 
        model_flops(),
        model_params() {}
};

enum profiler_backend_type {
    PROFILER_BACKEND_TYPE_CPU   = 0,
    PROFILER_BACKEND_TYPE_METAL = 1,
    PROFILER_BACKEND_TYPE_CUDA  = 2,
};

enum profiler_layer_type {
    PROFILER_LAYER_INPUT   = 0,
    PROFILER_LAYER_OUTPUT  = 1,
    PROFILER_LAYER_BACKEND = 2,
};

const char * device_name(void); 

uint32_t device_cpu_cores      (void);
float    device_cpu_flops      (struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t, int n_threads);
float    device_metal_flops    (struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t);
float    device_cuda_flops     (struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t);
float    device_inp_embd_delay (struct llama_model * model, enum ggml_type src0t, int n_tokens, int n_threads);
uint64_t device_physical_memory(bool available);
uint64_t device_swap_memory    (bool available);
uint64_t device_disk_read_bw   (const char * test_file, size_t buffer_size_mb);
float    device_memory_bw      (int n_thread);
float    device_cuda_memory_bw (struct llama_model * model);
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
