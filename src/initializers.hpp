#ifndef INITIALIZERS_CUDA_HPP
#define INITIALIZERS_CUDA_HPP

#include <cuda_runtime.h>

#include "targetf.hpp"
#include "backgroundsf.hpp"
#include "compoundf.hpp"
#include "auxiliaryf.hpp"
#include "scenesf.hpp"

/*
    printf("\033[31mDEBUG: id\033[0m\n");
    int size = (**list).size;
    printf("Number of objects: %d\n", size);
    for(int i = 0; i < size; i++) {
        Target* t1 = (**list).targets[i];
        Target* t2 = targets[i];
        Shape* s = shapes[i];
        printf("Target from list: %p, from targets: %p and shape: %p\n", t1, t2, s);
    }
*/


namespace init{

__device__ void createTargets(Target** targets, TargetList** list, Shape** shapes, int capacity);

__device__ BackgroundColor* createBackground(int i = 1);

};


#endif