#ifndef TARGET_CUDA_HPP
#define TARGET_CUDA_HPP

#include "linearAlgebraf.hpp"
#include "geometria.hpp"

class Target{
public:

Shape* shape;

    __device__ Target(Shape* shape_) : shape(shape_) {}

    __device__ float collision(const Ray& ray) const {
        if(shape == NULL) return -1.0f;

        return shape->rayCollision(ray);
    }

    __device__ Vector3D normal(const Vector3D& point) const {
        return shape->normal(point);
    }


};

class targetList{
public:

    Target** targets;
    size_t size;

    __device__ targetList(Target** targets_, int N) {
        targets = targets_; size = N;
    }

    __device__ targetList() : targets(NULL), size(0) {}

    __device__ Target operator[](int i) {
        return **(targets + i); // or *(*targets + i) ?
    }

};


#endif