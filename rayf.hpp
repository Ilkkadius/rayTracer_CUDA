#ifndef RAY_CUDA_HPP
#define RAY_CUDA_HPP

#include <cuda_runtime.h>

#include "vector3D.hpp"
#include "auxiliaryf.hpp"

class Ray{
public:

    Vector3D dir, pos;

    __host__ __device__ Ray() : dir(Vector3D(1,0,0)), pos(Vector3D(0,0,0)) {}

    __host__ __device__ Ray(const Vector3D& direction, const Vector3D& point) : dir(unitVec(direction)), pos(point) {}

    __host__ __device__ Vector3D at(float t) const {
        return pos + t*dir;
    }

    __host__ __device__ const Vector3D& location() const {
        return pos;
    }

    __host__ __device__ const Vector3D& heading() const {
        return dir;
    }

    __host__ __device__ const Vector3D& direction() const {
        return dir;
    }

};

class WindowVectors{
public:
    __host__ __device__ WindowVectors(Vector3D starter, Vector3D eye, Vector3D xVec, Vector3D yVec)
    : starter_(starter), eye_(eye), xVec_(xVec), yVec_(yVec) {}

    __host__ __device__ WindowVectors(const WindowVectors& window) : starter_(window.starter_), eye_(window.eye_), xVec_(window.xVec_), yVec_(window.yVec_) {}

    Vector3D starter_, eye_, xVec_, yVec_;
};

__host__ WindowVectors initialRays(const Vector3D& eye, const Vector3D& direction, 
            float windowDistance, const Vector3D& up, int height, int width, float windowHeight) {
    // Perpendicular direction to window
    Vector3D unitDirection = unitVec(direction);

    // Make "up" completely perpendicular to "direction" and normalize to unity
    Vector3D unitUp = unitVec(up - Dot(up, unitDirection) * unitDirection);

    Vector3D unitRight = -Cross(unitDirection, unitUp);

    float Ny = height/2.0f;
    if(height % 2 != 0) {
        Ny += 0.5f;
    }
    float Nx = width/2.0f;
    if(width % 2 != 0) {
        Nx += 0.5f;
    }

    float dn = windowHeight/static_cast<float>(height);

    Vector3D starter = unitDirection * windowDistance + (Ny * unitUp - Nx * unitRight) * dn;

    return WindowVectors(starter, eye, dn * unitRight, -dn * unitUp);
}


#endif