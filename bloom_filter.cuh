// Bloom Filter for CUDA-based Private Key Search

#ifndef BLOOM_FILTER_CUH
#define BLOOM_FILTER_CUH

#include <cuda.h>
#include <stdint.h>

#define BLOOM_SIZE 1000000 // Define size of bloom filter

__device__ void bloomAdd(uint8_t *filter, uint64_t key) {
    uint64_t hash = key % BLOOM_SIZE;
    filter[hash / 8] |= (1 << (hash % 8));
}

__device__ bool bloomCheck(uint8_t *filter, uint64_t key) {
    uint64_t hash = key % BLOOM_SIZE;
    return (filter[hash / 8] & (1 << (hash % 8))) != 0;
}

#endif // BLOOM_FILTER_CUH
