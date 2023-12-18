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

    __device__ dayTime(const Vector3D& color = Vector3D(0.3f,0.4f,1.0f)) : color_(color) {}

    __device__ Vector3D colorize(const Ray& ray) const {
        float a = 0.5f*(ray.dir.z + 1.0f);
        return (1.0f-a)*Vector3D(1, 1, 1) + a*color_;
    }

    Vector3D color_;
};

#endif