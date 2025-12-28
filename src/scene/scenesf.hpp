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

    __device__ void testScene(TargetList* list, int capacity);

    __device__ void empty(TargetList* list, int capacity);

    __device__ void Platon(TargetList* list, int capacity);

    __device__ void CSG(TargetList* list, int capacity);

    __device__ void CSG2(TargetList* list, int capacity);

    __device__ void light(TargetList* list, int capacity);
};

#endif