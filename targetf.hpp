#ifndef TARGET_CUDA_HPP
#define TARGET_CUDA_HPP

#include "vector3D.hpp"
#include "geometria.hpp"

class Target{
public:

Shape* shape;

    __device__ Target(Shape* shape_) : shape(shape_) {}

    __device__ float collision(const Ray& ray) const {
        return shape->rayCollision(ray);
    }

    __device__ Vector3D normal(const Vector3D& point) const {
        return shape->normal(point);
    }


};

class targetList{
public:

    Target** targets;
    size_t size, capacity;

    __device__ targetList(Target** targets_, int N, int maxN) {
        targets = targets_; size = N, capacity = maxN;
    }

    __device__ targetList() : targets(NULL), size(0) {}

    __device__ Target operator[](int i) {
        return **(targets + i); // NOT *(*targets + i) ?
    }

    /**
     * @brief Append a sequence of Targets to targetList, free given sequence
     * 
     * @param additional 
     * @param amount 
     * @return __device__ 
     */
    __device__ void append(Target** additional, size_t amount) {
        for(size_t i = 0; i < amount; i++) {
            if(size < capacity) {
                targets[size] = additional[i];
                size++;
            } else {
                delete additional[i]->shape;
                delete additional[i];
            }
        }
        delete[] additional;
    }

};


#endif