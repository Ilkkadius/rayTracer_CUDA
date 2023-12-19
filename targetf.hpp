#ifndef TARGET_CUDA_HPP
#define TARGET_CUDA_HPP

#include "vector3D.hpp"
#include "geometria.hpp"

class Target{
public:

Shape* shape;
Vector3D color;
float emissivity;

    __device__ Target(Shape* shape_, const Vector3D& color_, float emissivity_ = 0.0f) : shape(shape_), color(color_), emissivity(emissivity_) {}

    __device__ float collision(const Ray& ray) const {
        return shape->rayCollision(ray);
    }

    __device__ Vector3D normal(const Vector3D& point) const {
        return shape->normal(point);
    }

    __device__ Vector3D centroid() const {
        return shape->centroid();
    }

    __device__ void translate(const Vector3D& vec) {shape->translate(vec);}
    __device__ void translate(float x, float y, float z) {shape->translate(x,y,z);}

    __device__ void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) {
        shape->rotate(angle, axis, axisPos);
    }

    __device__ bool isRadiant() const {
        return emissivity > 0.0f;
    }

    __device__ Vector3D emission() const {
        return emissivity*color;
    }


};

class targetList{
public:

    Target** targets;
    size_t size, capacity;

    __device__ targetList(Target** targets_, int N, int maxN) {
        targets = targets_; size = N; capacity = maxN;
    }

    __device__ targetList(int capacity_) : targets(NULL), size(0), capacity(capacity_) {}

    __device__ Target* operator[](int i) {
        return *(targets + i); // NOT *(*targets + i) ?
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