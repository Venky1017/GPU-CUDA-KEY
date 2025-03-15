// Main CUDA-based Private Key Search Tool

#include <iostream>
#include <cuda.h>
#include "ecc_math.cuh"
#include "bloom_filter.cuh"

#define THREADS_PER_BLOCK 256
#define BLOOM_SIZE 1000000

__global__ void searchPrivateKeys(uint64_t start, uint64_t range, uint64_t *foundKey, uint8_t *bloomFilter) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < start + range) {
        if (!bloomCheck(bloomFilter, idx) && eccMultiply(idx)) {
            *foundKey = idx;
        }
    }
}

int main() {
    uint64_t start = 0x80000000000000000;
    uint64_t range = 0xFFFFFFFFFFFFFFFFF - start;
    uint64_t *d_foundKey, h_foundKey = 0;
    uint8_t *d_bloomFilter;

    cudaMalloc(&d_foundKey, sizeof(uint64_t));
    cudaMalloc(&d_bloomFilter, BLOOM_SIZE / 8);
    cudaMemset(d_bloomFilter, 0, BLOOM_SIZE / 8);
    cudaMemcpy(d_foundKey, &h_foundKey, sizeof(uint64_t), cudaMemcpyHostToDevice);

    int numBlocks = (range + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    searchPrivateKeys<<<numBlocks, THREADS_PER_BLOCK>>>(start, range, d_foundKey, d_bloomFilter);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_foundKey, d_foundKey, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_foundKey);
    cudaFree(d_bloomFilter);

    if (h_foundKey != 0) {
        std::cout << "Private key found: " << h_foundKey << std::endl;
    } else {
        std::cout << "No key found in range." << std::endl;
    }

    return 0;
}
