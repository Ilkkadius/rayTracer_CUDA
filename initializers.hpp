#ifndef INITIALIZERS_CUDA_HPP
#define INITIALIZERS_CUDA_HPP

#include <cuda_runtime.h>

#include "targetf.hpp"
#include "backgroundsf.hpp"

__device__ void createTargets(Target** targets, targetList** list, Shape** shapes) {
    *(shapes) = new Sphere(Vector3D(7,0,0), 1);
    *(shapes + 1) = new Sphere(Vector3D(0,0,-5000), 4999);
    *(shapes + 2) = new Triangle(Vector3D(7,3,0), Vector3D(8,-3, 0), Vector3D(7,-1,5));
    *(targets) = new Target(*shapes);
    *(targets + 1) = new Target(*(shapes + 1));
    *(targets + 2) = new Target(*(shapes + 2));
    *list = new targetList(targets, 3);
}

__device__ BackgroundColor* createDayTime() {
    return new dayTime();
}


#endif