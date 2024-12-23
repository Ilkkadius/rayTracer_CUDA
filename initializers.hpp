#ifndef INITIALIZERS_CUDA_HPP
#define INITIALIZERS_CUDA_HPP

#include <cuda_runtime.h>

#include "targetf.hpp"
#include "backgroundsf.hpp"
#include "compoundf.hpp"
#include "auxiliaryf.hpp"

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


__device__ void createTargets(Target** targets, targetList** list, Shape** shapes, int N) {
    float r = 100000;
    shapes[0] = new Sphere(Vector3D(7,0,0), 1);
    shapes[1] = new Sphere(Vector3D(0,0,-r/2.0f), r-1);
    shapes[2] = new Triangle(Vector3D(7,3,0), Vector3D(8,-3, 0), Vector3D(7,-1,5));
    targets[0] = new Target(shapes[0]);
    targets[1] = new Target(shapes[1]);
    targets[2] = new Target(shapes[2]);
    *list = new targetList(targets, 3, N);

    compoundTest test;
    test.copyToList(*list, shapes);
}

__device__ BackgroundColor* createDayTime() {
    return new dayTime();
}


#endif