#ifndef IMAGE_BACKGROUNDS_CUDA_HPP
#define IMAGE_BACKGROUNDS_CUDA_HPP

#include <cuda_runtime.h>

#include "vector3D.hpp"
#include "rayf.hpp"

/**
 * @brief Defines the color of a ray that escapes without collisions
 * 
 */
class BackgroundColor{
public:
    __device__ virtual Vector3D colorize(const Ray& ray) const {
        return Vector3D();
    }
};

class dayTime : public BackgroundColor{
public:

    __device__ dayTime(const Vector3D& color = Vector3D(0.3f,0.4f,1.0f));

    __device__ Vector3D colorize(const Ray& ray) const;

    Vector3D color_;
};

class nightTime : public BackgroundColor{
public:

    __device__ nightTime(const Vector3D& color = Vector3D(0.6,0.3,0.1));
    
    __device__ Vector3D colorize(const Ray& ray) const;

    Vector3D color_;
};

#endif