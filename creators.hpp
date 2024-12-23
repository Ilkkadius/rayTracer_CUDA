#ifndef CUDA_CREATOR_METHODS_HPP
#define CUDA_CREATOR_METHODS_HPP

#include <cuda_runtime.h>

#include "targetList.hpp"
#include "backgroundsf.hpp"
#include "scenesf.hpp"


/**
 * @brief Methods for kernels (device only) to use
 * 
 */
namespace Creator{
    __device__ void targetScene(Target** targets, targetList** list, Shape** shapes, int capacity) {
        int N = capacity;
        switch(1) {
            case 1:
                Scene::Platon(list, targets, shapes, N);
                break;
            case 2:
                Scene::testScene(list, targets, shapes, N);
                break;
            default:
                Scene::empty(list, targets, shapes, N);
                break;
        }
    }

    __device__ BackgroundColor* background(int i = 1) {
        switch(i % 2) {
            case 1:
                return new nightTime();
            default:
                return new dayTime();
        }
    }
}


#endif