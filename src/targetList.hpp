#ifndef TARGET_CONTAINER_CUDA_HPP
#define TARGET_CONTAINER_CUDA_HPP

#include <cuda_runtime.h>

#include "targetf.hpp"
#include "scatteringf.hpp"

class TargetList{
public:

    Target** targets;
    size_t size, capacity;

    __device__ TargetList(Target** targets_, int N, int maxN);

    __device__ TargetList(int capacity_);

    __device__ Target* operator[](int i);

    /**
     * @brief Append a sequence of Targets to targetList, free given sequence
     * 
     * @param additional 
     * @param amount 
     * @return __device__ 
     */
    __device__ void append(Target** additional, size_t amount);

    __device__ void findCollision(const Ray& ray, HitInfo& hit) const;

};

#endif