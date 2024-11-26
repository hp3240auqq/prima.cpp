#include "common.cuh"
#include "read.cuh"

__global__ void read_vram_f32(
    const float * data, int64_t ne,
    int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
    int64_t ne00, int64_t ne01, int64_t ne02
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ne) return;

    int i = idx % ne00;
    int j = (idx / ne00) % ne01;
    int k = (idx / (ne00 * ne01)) % ne02;

    int64_t offset = i * nb00 + j * nb01 + k * nb02;

    volatile float value = data[offset / sizeof(float)];
    asm volatile("" : : "f"(value) : "memory");
}

void ggml_cuda_read(ggml_tensor * dst) {
    const int64_t ne = ggml_nelements(dst);
    GGML_ASSERT(ggml_nbytes(dst) <= INT_MAX);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne00 = dst->ne[0];
    const int64_t ne01 = dst->ne[1];
    const int64_t ne02 = dst->ne[2];

    const int64_t nb00 = dst->nb[0];
    const int64_t nb01 = dst->nb[1];
    const int64_t nb02 = dst->nb[2];
    const int64_t nb03 = dst->nb[3];

    const char * dst_ddc = (const char *)dst->data;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int num_blocks = (ne + CUDA_READ_BLOCK_SIZE - 1) / CUDA_READ_BLOCK_SIZE;
    read_vram_f32<<<num_blocks, CUDA_READ_BLOCK_SIZE, 0, stream>>>(
        (const float *)dst_ddc, ne, nb00, nb01, nb02, nb03, ne00, ne01, ne02
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}
