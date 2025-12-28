#ifndef SCENES_CUDA_HPP
#define SCENES_CUDA_HPP

#include "cuda_runtime.h"

#include "compoundf.hpp"

enum class sceneType {
    EMPTY,
    PLATON,
    LIGHT,
    CSG,
    TEST,
    CSG2
};

namespace Scene{

    __host__ bool parseScene(std::string line, sceneType& type);

    __device__ void testScene(TargetList* list);

    __device__ void empty(TargetList* list);

    __device__ void Platon(TargetList* list);

    __device__ void CSG(TargetList* list);

    __device__ void CSG2(TargetList* list);

    __device__ void light(TargetList* list);
};

#endif