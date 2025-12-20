#ifndef SCENES_CUDA_HPP
#define SCENES_CUDA_HPP

#include "cuda_runtime.h"

#include "compoundf.hpp"

namespace Scene{

    __device__ void testScene(TargetList** list, Target** targets, Shape** shapes, int capacity);

    __device__ void empty(TargetList** list, Target** targets, Shape** shapes, int capacity);

    __device__ void Platon(TargetList** list, Target** targets, Shape** shapes, int capacity);

};

#endif