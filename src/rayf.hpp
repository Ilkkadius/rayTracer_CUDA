#ifndef RAY_CUDA_HPP
#define RAY_CUDA_HPP

#include <cuda_runtime.h>

#include "vector3D.hpp"
#include "auxiliaryf.hpp"

class Ray{
public:

    Vector3D dir, pos;

    __host__ __device__ Ray();

    __host__ __device__ Ray(const Vector3D& direction, const Vector3D& point);

    __host__ __device__ Vector3D at(float t) const;

    __host__ __device__ Vector3D location() const;

    __host__ __device__ Vector3D heading() const;

    __host__ __device__ Vector3D direction() const;

};

class WindowVectors{
public:
    __host__ __device__ WindowVectors(Vector3D starter, Vector3D eye, Vector3D xVec, Vector3D yVec);

    __host__ __device__ WindowVectors(const WindowVectors& window);

    Vector3D starter_, eye_, xVec_, yVec_;
};

__host__ WindowVectors initialRays(const Vector3D& eye, const Vector3D& direction, 
            float windowDistance, const Vector3D& up, int height, int width, float windowHeight);


#endif