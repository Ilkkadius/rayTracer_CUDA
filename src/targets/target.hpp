#ifndef TARGET_CUDA_HPP
#define TARGET_CUDA_HPP

#include "vector3D.hpp"
#include "rayf.hpp"
#include "hitInfof.hpp"
#include "rotationf.hpp"

class HitInfo;

class Target{
public:

    Vector3D color = Vector3D(1.0f, 1.0f, 1.0f);
    float emissivity = 0.0f;

    __device__ virtual bool rayCollision(const Ray& ray, HitInfo* hit) const = 0;
    __device__ virtual int allCollisions(const Ray& ray, HitInfo* hitlist) const = 0;

    __device__ virtual Vector3D centroid() const = 0;

    __device__ virtual void translate(const Vector3D& vec) = 0;
    __device__ virtual void translate(float x, float y, float z) = 0;

    __device__ virtual void rotate(float angle, const Vector3D& axis, const Vector3D& axisPos) = 0;

    __device__ virtual Vector3D emission() const {return color;}

    __device__ virtual Vector3D minBox() const = 0;
    __device__ virtual Vector3D maxBox() const = 0;


};


#endif